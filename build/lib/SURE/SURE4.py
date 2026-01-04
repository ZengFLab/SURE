import pyro
import pyro.distributions as dist
from pyro.optim import ExponentialLR
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs
from torch.distributions import constraints
from torch.distributions.transforms import SoftmaxTransform

from .utils.custom_mlp import MLP, Exp, ZeroBiasMLP
from .utils.utils import CustomDataset, CustomDataset2, CustomDataset4, tensor_to_numpy, convert_to_tensor

from .dist.negbinomial import NegativeBinomial as MyNB
from .dist.negbinomial import ZeroInflatedNegativeBinomial as MyZINB

import zuko 
from pyro.contrib.zuko import ZukoToPyro

import os
import argparse
import random
import numpy as np
import datatable as dt
from tqdm import tqdm
from scipy import sparse

import scanpy as sc
from .atac import binarize

from typing import Literal

import warnings
warnings.filterwarnings("ignore")

import dill as pickle
import gzip
from packaging.version import Version
torch_version = torch.__version__


def set_random_seed(seed):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA, set the seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups.
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for Pyro
    pyro.set_rng_seed(seed)

class SURE4(nn.Module):
    """SUccinct REpresentation of single-omics cells

    Parameters
    ----------
    inpute_size
        Number of features (e.g., genes, peaks, proteins, etc.) per cell.
    codebook_size
        Number of metacells.
    cov_size
        Number of cell-level factors. 
    z_dim
        Dimensionality of latent states and metacells. 
    hidden_layers
        A list give the numbers of neurons for each hidden layer.
    loss_func
        The likelihood model for single-cell data generation. 
        
        One of the following: 
        * ``'negbinomial'`` -  negative binomial distribution (default)
        * ``'poisson'`` - poisson distribution
        * ``'multinomial'`` - multinomial distribution
    z_dist
        The distribution model for latent states. 
        
        One of the following:
        * ``'normal'`` - normal distribution
        * ``'laplacian'`` - Laplacian distribution
        * ``'studentt'`` - Student-t distribution. 
    use_cuda
        A boolean option for switching on cuda device. 

    Examples
    --------
    >>>
    >>>
    >>>

    """
    def __init__(self,
                 input_size: int,
                 codebook_size: int = 200,
                 cov_size: int = 0,
                 transforms: int = 1,
                 z_dim: int = 50,
                 z_dist: Literal['normal','studentt','laplacian','cauchy','gumbel'] = 'studentt',
                 loss_func: Literal['negbinomial','poisson','multinomial','bernoulli'] = 'poisson',
                 dispersion: float = 10.0,
                 use_zeroinflate: bool = True,
                 hidden_layers: list = [500],
                 hidden_layer_activation: Literal['relu','softplus','leakyrelu','linear'] = 'relu',
                 flow_hidden_layers = [64,64,64],
                 nn_dropout: float = 0.1,
                 post_layer_fct: list = ['layernorm'],
                 post_act_fct: list = None,
                 config_enum: str = 'parallel',
                 use_cuda: bool = True,
                 seed: int = 42,
                 dtype = torch.float32, # type: ignore
                 ):
        super().__init__()

        self.input_size = input_size
        self.cov_size = cov_size
        self.dispersion = dispersion
        self.latent_dim = z_dim
        self.hidden_layers = hidden_layers
        self.decoder_hidden_layers = hidden_layers[::-1]
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.loss_func = loss_func
        self.options = None
        self.code_size=codebook_size
        self.latent_dist = z_dist
        self.dtype = dtype
        self.use_zeroinflate=use_zeroinflate
        self.nn_dropout = nn_dropout
        self.post_layer_fct = post_layer_fct
        self.post_act_fct = post_act_fct
        self.hidden_layer_activation = hidden_layer_activation
        self.transforms = transforms
        self.flow_hidden_layers = flow_hidden_layers
        
        self.codebook_weights = None

        set_random_seed(seed)
        self.setup_networks()

    def setup_networks(self):
        latent_dim = self.latent_dim
        hidden_sizes = self.hidden_layers

        nn_layer_norm, nn_batch_norm, nn_layer_dropout = False, False, False
        na_layer_norm, na_batch_norm, na_layer_dropout = False, False, False

        if self.post_layer_fct is not None:
            nn_layer_norm=True if ('layernorm' in self.post_layer_fct) or ('layer_norm' in self.post_layer_fct) else False
            nn_batch_norm=True if ('batchnorm' in self.post_layer_fct) or ('batch_norm' in self.post_layer_fct) else False
            nn_layer_dropout=True if 'dropout' in self.post_layer_fct else False

        if self.post_act_fct is not None:
            na_layer_norm=True if ('layernorm' in self.post_act_fct) or ('layer_norm' in self.post_act_fct) else False
            na_batch_norm=True if ('batchnorm' in self.post_act_fct) or ('batch_norm' in self.post_act_fct) else False
            na_layer_dropout=True if 'dropout' in self.post_act_fct else False

        if nn_layer_norm and nn_batch_norm and nn_layer_dropout:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout),nn.BatchNorm1d(layer.module.out_features), nn.LayerNorm(layer.module.out_features))
        elif nn_layer_norm and nn_layer_dropout:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout), nn.LayerNorm(layer.module.out_features))
        elif nn_batch_norm and nn_layer_dropout:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout), nn.BatchNorm1d(layer.module.out_features))
        elif nn_layer_norm and nn_batch_norm:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.BatchNorm1d(layer.module.out_features), nn.LayerNorm(layer.module.out_features))
        elif nn_layer_norm:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.LayerNorm(layer.module.out_features)
        elif nn_batch_norm:
            post_layer_fct = lambda layer_ix, total_layers, layer:nn.BatchNorm1d(layer.module.out_features)
        elif nn_layer_dropout:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Dropout(self.nn_dropout)
        else:
            post_layer_fct = lambda layer_ix, total_layers, layer: None

        if na_layer_norm and na_batch_norm and na_layer_dropout:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout),nn.BatchNorm1d(layer.module.out_features), nn.LayerNorm(layer.module.out_features))
        elif na_layer_norm and na_layer_dropout:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout), nn.LayerNorm(layer.module.out_features))
        elif na_batch_norm and na_layer_dropout:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout), nn.BatchNorm1d(layer.module.out_features))
        elif na_layer_norm and na_batch_norm:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.BatchNorm1d(layer.module.out_features), nn.LayerNorm(layer.module.out_features))
        elif na_layer_norm:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.LayerNorm(layer.module.out_features)
        elif na_batch_norm:
            post_act_fct = lambda layer_ix, total_layers, layer:nn.BatchNorm1d(layer.module.out_features)
        elif na_layer_dropout:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Dropout(self.nn_dropout)
        else:
            post_act_fct = lambda layer_ix, total_layers, layer: None

        if self.hidden_layer_activation == 'relu':
            activate_fct = nn.ReLU
        elif self.hidden_layer_activation == 'softplus':
            activate_fct = nn.Softplus
        elif self.hidden_layer_activation == 'leakyrelu':
            activate_fct = nn.LeakyReLU
        elif self.hidden_layer_activation == 'linear':
            activate_fct = nn.Identity

        self.encoder_n = MLP(
                [self.latent_dim] + hidden_sizes + [self.code_size],
                activation=activate_fct,
                output_activation=None,
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )

        self.encoder_zn = MLP(
            [self.input_size] + hidden_sizes + [[latent_dim, latent_dim]],
            activation=activate_fct,
            output_activation=[None, Exp],
            post_layer_fct=post_layer_fct,
            post_act_fct=post_act_fct,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        if self.cov_size>0:
            self.covariate_effect = ZeroBiasMLP(
                [self.latent_dim + self.cov_size] + self.decoder_hidden_layers + [self.latent_dim],
                activation=activate_fct,
                output_activation=None,
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )
            
        self.decoder_log_mu = MLP(
                [self.latent_dim] + self.decoder_hidden_layers + [self.input_size],
                activation=activate_fct,
                output_activation=None,
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )

        if self.latent_dist == 'studentt':
            self.codebook = MLP(
                [self.code_size] + hidden_sizes + [[latent_dim,latent_dim]],
                activation=activate_fct,
                output_activation=[Exp,None],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )
        else:
            self.codebook = MLP(
                [self.code_size] + hidden_sizes + [latent_dim],
                activation=activate_fct,
                output_activation=None,
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )
            
        self.decoder_x = zuko.flows.NSF(features=self.input_size, context=self.latent_dim, 
                                         transforms=self.transforms, hidden_features=self.flow_hidden_layers)
        
        if self.use_cuda:
            self.cuda()

    def get_device(self):
        return next(self.parameters()).device

    def cutoff(self, xs, thresh=None):
        eps = torch.finfo(xs.dtype).eps
        
        if not thresh is None:
            if eps < thresh:
                eps = thresh

        xs = xs.clamp(min=eps)

        if torch.any(torch.isnan(xs)):
            xs[torch.isnan(xs)] = eps

        return xs

    def softmax(self, xs):
        #xs = SoftmaxTransform()(xs)
        xs = dist.Multinomial(total_count=1, logits=xs).mean
        return xs
    
    def sigmoid(self, xs):
        #sigm_enc = nn.Sigmoid()
        #xs = sigm_enc(xs)
        #xs = clamp_probs(xs)
        xs = dist.Bernoulli(logits=xs).mean
        return xs

    def softmax_logit(self, xs):
        eps = torch.finfo(xs.dtype).eps
        xs = self.softmax(xs)
        xs = torch.logit(xs, eps=eps)
        return xs

    def logit(self, xs):
        eps = torch.finfo(xs.dtype).eps
        xs = torch.logit(xs, eps=eps)
        return xs

    def dirimulti_param(self, xs):
        xs = self.dirimulti_mass * self.sigmoid(xs)
        return xs

    def multi_param(self, xs):
        xs = self.softmax(xs)
        return xs

    def model(self, xs, us=None):
        pyro.module('sure', self)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        self.options = dict(dtype=xs.dtype, device=xs.device)
        
        if self.loss_func=='negbinomial':
            dispersion = pyro.param("dispersion", self.dispersion *
                                     xs.new_ones(self.input_size), constraint=constraints.positive)
            
        if self.use_zeroinflate:
            gate_logits = pyro.param("dropout_rate", xs.new_zeros(self.input_size))
            
        acs_scale = pyro.param("codebook_scale", xs.new_ones(self.latent_dim), constraint=constraints.positive)

        I = torch.eye(self.code_size)
        if self.latent_dist=='studentt':
            acs_dof,acs_loc = self.codebook(I)
        else:
            acs_loc = self.codebook(I)

        with pyro.plate('data'):
            prior = torch.zeros(batch_size, self.code_size, **self.options)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=prior))

            zn_loc = torch.matmul(ns,acs_loc)
            #zn_scale = torch.matmul(ns,acs_scale)
            zn_scale = acs_scale

            if self.latent_dist == 'studentt':
                prior_dof = torch.matmul(ns,acs_dof)
                zns = pyro.sample('zn', dist.StudentT(df=prior_dof, loc=zn_loc, scale=zn_scale).to_event(1))
            elif self.latent_dist == 'laplacian':
                zns = pyro.sample('zn', dist.Laplace(zn_loc, zn_scale).to_event(1))
            elif self.latent_dist == 'cauchy':
                zns = pyro.sample('zn', dist.Cauchy(zn_loc, zn_scale).to_event(1))
            elif self.latent_dist == 'normal':
                zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))
            elif self.latent_dist == 'gumbel':
                zns = pyro.sample('zn', dist.Gumbel(zn_loc, zn_scale).to_event(1))

            if (self.cov_size>0) and (us is not None):
                zus = self.covariate_effect([zns,us])
                zs = zns+zus
            else:
                zs = zns

            pyro.sample('x', ZukoToPyro(self.decoder_x(zs)), obs=xs)

    def guide(self, xs, us=None):
        with pyro.plate('data'):
            zn_loc, zn_scale = self.encoder_zn(xs)
            zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            alpha = self.encoder_n(zns)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=alpha))

    def _get_codebook(self):
        I = torch.eye(self.code_size, **self.options)
        if self.latent_dist=='studentt':
            _,cb = self.codebook(I)
        else:
            cb = self.codebook(I)
        return cb
    
    def get_codebook(self):
        """
        Return the mean part of metacell codebook
        """
        cb = self._get_codebook()
        cb = tensor_to_numpy(cb)
        return cb

    def _get_cell_embedding(self, xs):           
        zns, _ = self.encoder_zn(xs)
        return zns 
    
    def get_cell_embedding(self, 
                             xs, 
                             batch_size: int = 1024,
                             show_progress: bool = True):
        """
        Return cells' latent representations

        Parameters
        ----------
        xs
            Single-cell expression matrix. It should be a Numpy array or a Pytorch Tensor.
        batch_size
            Size of batch processing.
        show_progress
            Verbose on or off
        """
        xs = self.preprocess(xs)
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        Z = []
        with tqdm(total=len(dataloader), disable=not show_progress, desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                zns = self._get_cell_embedding(X_batch)
                Z.append(tensor_to_numpy(zns))
                pbar.update(1)

        Z = np.concatenate(Z)
        return Z
    
    def _code(self, xs):
        #zns,_ = self.encoder_zn(xs)
        zns = self._get_cell_embedding(xs)
        alpha = self.encoder_n(zns)
        return alpha
    
    def code(self, xs, batch_size=1024, show_progress=True):
        xs = self.preprocess(xs)
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        A = []
        with tqdm(total=len(dataloader), disable=not show_progress, desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                a = self._code(X_batch)
                A.append(tensor_to_numpy(a))
                pbar.update(1)

        A = np.concatenate(A)
        return A
    
    def _soft_assignments(self, xs):
        alpha = self._code(xs)
        alpha = self.softmax(alpha)
        return alpha
    
    def soft_assignments(self, xs, batch_size=1024, show_progress=True):
        """
        Map cells to metacells and return the probabilistic values of metacell assignments
        """
        xs = self.preprocess(xs)
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        A = []
        with tqdm(total=len(dataloader), disable=not show_progress, desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                a = self._soft_assignments(X_batch)
                A.append(tensor_to_numpy(a))
                pbar.update(1)

        A = np.concatenate(A)
        return A
    
    def _hard_assignments(self, xs):
        alpha = self._code(xs)
        res, ind = torch.topk(alpha, 1)
        ns = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ns
    
    def hard_assignments(self, xs, batch_size=1024, show_progress=True):
        """
        Map cells to metacells and return the assigned metacell identities.
        """
        xs = self.preprocess(xs)
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        A = []
        with tqdm(total=len(dataloader), disable=not show_progress, desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                a = self._hard_assignments(X_batch)
                A.append(tensor_to_numpy(a))
                pbar.update(1)

        A = np.concatenate(A)
        return A
    
    def preprocess(self, xs, threshold=0): 
        if sparse.issparse(xs):
            xs = xs.toarray()
        return xs 
    
    def fit(self, xs, 
            us = None, 
            num_epochs: int = 100, 
            learning_rate: float = 0.0001, 
            batch_size: int = 256, 
            algo: Literal['adam','rmsprop','adamw'] = 'adam', 
            beta_1: float = 0.9, 
            weight_decay: float = 0.005, 
            decay_rate: float = 0.9,
            config_enum: str = 'parallel',
            threshold: int = 0,
            use_jax: bool = True,
            show_progress: bool = True):
        """
        Train the SURE model.

        Parameters
        ----------
        xs
            Single-cell experssion matrix. It should be a Numpy array or a Pytorch Tensor. Rows are cells and columns are features.
        us
            cell-level factor matrix. 
        num_epochs
            Number of training epochs.
        learning_rate
            Parameter for training.
        batch_size
            Size of batch processing.
        algo
            Optimization algorithm.
        beta_1
            Parameter for optimization.
        weight_decay
            Parameter for optimization.
        decay_rate 
            Parameter for optimization.
        use_jax
            If toggled on, Jax will be used for speeding up. CAUTION: This will raise errors because of unknown reasons when it is called in
            the Python script or Jupyter notebook. It is OK if it is used when runing SURE in the shell command.
        """
        xs = self.preprocess(xs, threshold=threshold)
        xs = convert_to_tensor(xs, dtype=self.dtype, device=self.get_device())
        if us is not None:
            us = convert_to_tensor(us, dtype=self.dtype, device=self.get_device())

        if us is None:
            dataset = CustomDataset2(xs, xs)
        else:
            dataset = CustomDataset2(xs, us)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # setup the optimizer
        optim_params = {'lr': learning_rate, 'betas': (beta_1, 0.999), 'weight_decay': weight_decay}

        if algo.lower()=='rmsprop':
            optimizer = torch.optim.RMSprop
        elif algo.lower()=='adam':
            optimizer = torch.optim.Adam
        elif algo.lower() == 'adamw':
            optimizer = torch.optim.AdamW
        else:
            raise ValueError("An optimization algorithm must be specified.")
        scheduler = ExponentialLR({'optimizer': optimizer, 'optim_args': optim_params, 'gamma': decay_rate})

        pyro.clear_param_store()

        # set up the loss(es) for inference, wrapping the guide in config_enumerate builds the loss as a sum
        # by enumerating each class label form the sampled discrete categorical distribution in the model
        Elbo = JitTraceEnum_ELBO if use_jax else TraceEnum_ELBO
        elbo = Elbo(max_plate_nesting=1, strict_enumeration_warning=False)
        guide = config_enumerate(self.guide, config_enum, expand=True)
        loss_basic = SVI(self.model, guide, scheduler, loss=elbo)

        # build a list of all losses considered
        losses = [loss_basic]
        num_losses = len(losses)

        with tqdm(total=num_epochs, disable=not show_progress, desc='Training', unit='epoch') as pbar:
            for epoch in range(num_epochs):
                epoch_losses = [0.0] * num_losses
                for batch_x, batch_u, _ in dataloader:
                    for loss_id in range(num_losses):
                        if us is None:
                            new_loss = losses[loss_id].step(batch_x)
                        else:
                            new_loss = losses[loss_id].step(batch_x, batch_u)
                        epoch_losses[loss_id] += new_loss

                avg_epoch_losses_ = map(lambda v: v / len(dataloader), epoch_losses)
                avg_epoch_losses = map(lambda v: "{:.4f}".format(v), avg_epoch_losses_)

                # store the loss
                str_loss = " ".join(map(str, avg_epoch_losses))

                # Update progress bar
                pbar.set_postfix({'loss': str_loss})
                pbar.update(1)

    @classmethod
    def save_model(cls, model, file_path, compression=False):
        """Save the model to the specified file path."""
        file_path = os.path.abspath(file_path)

        model.eval()
        if compression:
            with gzip.open(file_path, 'wb') as pickle_file:
                pickle.dump(model, pickle_file)
        else:
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(model, pickle_file)

        print(f'Model saved to {file_path}')

    @classmethod
    def load_model(cls, file_path):
        """Load the model from the specified file path and return an instance."""
        print(f'Model loaded from {file_path}')

        file_path = os.path.abspath(file_path)
        if file_path.endswith('gz'):
            with gzip.open(file_path, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
        else:
            with open(file_path, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
        
        return model

        
EXAMPLE_RUN = (
    "example run: SURE --help"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="SURE\n{}".format(EXAMPLE_RUN))

    parser.add_argument(
        "--cuda", action="store_true", help="use GPU(s) to speed up training"
    )
    parser.add_argument(
        "--jit", action="store_true", help="use PyTorch jit to speed up training"
    )
    parser.add_argument(
        "-n", "--num-epochs", default=200, type=int, help="number of epochs to run"
    )
    parser.add_argument(
        "-enum",
        "--enum-discrete",
        default="parallel",
        help="parallel, sequential or none. uses parallel enumeration by default",
    )
    parser.add_argument(
        "-data",
        "--data-file",
        default=None,
        type=str,
        help="the data file",
    )
    parser.add_argument(
        "-cf",
        "--cell-factor-file",
        default=None,
        type=str,
        help="the file for the record of cell-level factors",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=1000,
        type=int,
        help="number of cells to be considered in a batch",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.0001,
        type=float,
        help="learning rate for Adam optimizer",
    )
    parser.add_argument(
        "-cs",
        "--codebook-size",
        default=100,
        type=int,
        help="size of vector quantization codebook",
    )
    parser.add_argument(
        "--z-dist",
        default='gumbel',
        type=str,
        choices=['normal','laplacian','studentt','cauchy','gumbel'],
        help="distribution model for latent representation",
    )
    parser.add_argument(
        "-zd",
        "--z-dim",
        default=10,
        type=int,
        help="size of the tensor representing the latent variable z variable",
    )
    parser.add_argument(
        "-likeli",
        "--likelihood",
        default='negbinomial',
        type=str,
        choices=['negbinomial', 'multinomial', 'poisson', 'bernoulli'],
        help="specify the distribution likelihood function",
    )
    parser.add_argument(
        "-zi",
        "--zeroinflate",
        action="store_true",
        help="use zeroinflation",
    )
    parser.add_argument(
        "-id",
        "--inverse-dispersion",
        default=10.0,
        type=float,
        help="inverse dispersion prior for negative binomial",
    )
    parser.add_argument(
        "-hl",
        "--hidden-layers",
        nargs="+",
        default=[500],
        type=int,
        help="a tuple (or list) of MLP layers to be used in the neural networks "
        "representing the parameters of the distributions in our model",
    )
    parser.add_argument(
        "-hla",
        "--hidden-layer-activation",
        default='relu',
        type=str,
        choices=['relu','softplus','leakyrelu','linear'],
        help="activation function for hidden layers",
    )
    parser.add_argument(
        "-plf",
        "--post-layer-function",
        nargs="+",
        default=['layernorm'],
        type=str,
        help="post functions for hidden layers, could be none, dropout, layernorm, batchnorm, or combination, default is 'dropout layernorm'",
    )
    parser.add_argument(
        "-paf",
        "--post-activation-function",
        nargs="+",
        default=['none'],
        type=str,
        help="post functions for activation layers, could be none or dropout, default is 'none'",
    )
    parser.add_argument(
        "-dr",
        "--decay-rate",
        default=0.9,
        type=float,
        help="decay rate for Adam optimizer",
    )
    parser.add_argument(
        "--layer-dropout-rate",
        default=0.1,
        type=float,
        help="droput rate for neural networks",
    )
    parser.add_argument(
        "-b1",
        "--beta-1",
        default=0.95,
        type=float,
        help="beta-1 parameter for Adam optimizer",
    )
    parser.add_argument(
        "-64",
        "--float64",
        action="store_true",
        help="use double float precision",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for controlling randomness in this example",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        type=str,
        help="path to save model for prediction",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert (
        (args.data_file is not None) and (
            os.path.exists(args.data_file))
    ), "data file must be provided"

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.float64:
        dtype = torch.float64
        torch.set_default_dtype(torch.float64)
    else:
        dtype = torch.float32
        torch.set_default_dtype(torch.float32)

    xs = dt.fread(file=args.data_file, header=True).to_numpy()
    us = None 
    if args.cell_factor_file is not None:
        us = dt.fread(file=args.cell_factor_file, header=True).to_numpy()

    input_size = xs.shape[1]
    cov_size = 0 if us is None else us.shape[1]

    ###########################################
    sure = SURE4(
        input_size=input_size,
        cov_size=cov_size,
        dispersion=args.dispersion,
        z_dim=args.z_dim,
        hidden_layers=args.hidden_layers,
        hidden_layer_activation=args.hidden_layer_activation,
        use_cuda=args.cuda,
        config_enum=args.enum_discrete,
        use_zeroinflate=args.zeroinflate,
        loss_func=args.likelihood,
        nn_dropout=args.layer_dropout_rate,
        post_layer_fct=args.post_layer_function,
        post_act_fct=args.post_activation_function,
        codebook_size=args.codebook_size,
        z_dist = args.z_dist,
        dtype=dtype,
    )

    sure.fit(xs, us=us, 
             num_epochs=args.num_epochs,
             learning_rate=args.learning_rate,
             batch_size=args.batch_size,
             beta_1=args.beta_1,
             decay_rate=args.decay_rate,
             use_jax=args.jit,
             config_enum=args.enum_discrete,
             )

    if args.save_model is not None:
        if args.save_model.endswith('gz'):
            SURE4.save_model(sure, args.save_model, compression=True)
        else:
            SURE4.save_model(sure, args.save_model)
    


if __name__ == "__main__":

    main()