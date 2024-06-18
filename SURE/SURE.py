from pyro.optim import Adam, ExponentialLR
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.contrib.examples.util import print_and_log
import pyro.distributions as dist
import pyro
from utils.scdata_cached import mkdir_p, setup_data_loader, SingleCellCached
from utils.custom_mlp import MLP, Exp
from torch.nn.modules.linear import Linear
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs
from torch.distributions import constraints
from torch.distributions.transforms import SoftmaxTransform
import torch.nn.functional as ft
import torch.nn as nn
import torch
import argparse
import os
import time as tm
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class SURE(nn.Module):
    def __init__(self,
                 input_size=2000,
                 undesired_size=2,
                 inverse_dispersion=10.0,
                 z_dim=50,
                 hidden_layers=(500,),
                 hidden_layer_activation='relu',
                 use_undesired=True,
                 undesired_factor=None,
                 config_enum=None,
                 use_cuda=False,
                 dist_model='dmm',
                 use_zeroinflate=False,
                 use_exact_zeroinflate=False,
                 gate_prior=0.7,
                 delta=0.5,
                 loss_func='multinomial',
                 dirimulti_mass=1,
                 nn_dropout = 1e-3,
                 post_layer_fct=None,
                 post_act_fct=None,
                 effect_size_estimator='linear',
                 use_laplacian=True,
                 code_size=20,
                 ):
        super().__init__()

        self.input_size = input_size
        self.undesired_size = undesired_size
        self.inverse_dispersion = inverse_dispersion
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.use_undesired = use_undesired
        self.undesired_factor = undesired_factor
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.dist_model = dist_model
        self.use_zeroinflate = use_zeroinflate
        self.use_exact_zeroinflate = use_exact_zeroinflate
        self.delta = delta
        self.loss_func = loss_func
        self.dirimulti_mass = dirimulti_mass
        self.effect_size_estimator = effect_size_estimator
        self.use_laplacian = use_laplacian
        self.options = None
        self.code_size=code_size

        if use_exact_zeroinflate:
            self.use_zeroinflate = True

        if gate_prior < 1e-5:
            gate_prior = 1e-5
        elif gate_prior == 1:
            gate_prior = 1-1e-5
        self.gate_prior = np.log(gate_prior) - np.log(1-gate_prior)

        self.nn_dropout = nn_dropout
        self.post_layer_fct = post_layer_fct
        self.post_act_fct = post_act_fct
        self.hidden_layer_activation = hidden_layer_activation

        self.setup_networks()

    def setup_networks(self):
        z_dim = self.z_dim
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
            [self.z_dim] + hidden_sizes + [self.code_size],
            activation=activate_fct,
            output_activation=None,
            post_layer_fct=post_layer_fct,
            post_act_fct=post_act_fct,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        self.encoder_zn = MLP(
            [self.input_size] + hidden_sizes + [[z_dim, z_dim]],
            activation=activate_fct,
            output_activation=[None, Exp],
            post_layer_fct=post_layer_fct,
            post_act_fct=post_act_fct,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        if self.use_undesired:
            if self.loss_func in ['gaussian','lognormal']:
                self.decoder_concentrate = MLP(
                    [self.undesired_size + self.z_dim] + hidden_sizes + [[self.input_size, self.input_size]],
                    activation=activate_fct,
                    output_activation=[None, Exp],
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )
            else:
                self.decoder_concentrate = MLP(
                    [self.undesired_size + self.z_dim] + hidden_sizes + [self.input_size],
                    activation=activate_fct,
                    output_activation=None,
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )

            self.encoder_gate = MLP(
                [self.undesired_size + self.z_dim] + hidden_sizes + [[self.input_size, 1]],
                activation=activate_fct,
                output_activation=[None, Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )
        else:
            if self.loss_func in ['gaussian','lognormal']:
                self.decoder_concentrate = MLP(
                    [self.z_dim] + hidden_sizes + [[self.input_size, self.input_size]],
                    activation=activate_fct,
                    output_activation=[None, Exp],
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )
            else:
                self.decoder_concentrate = MLP(
                    [self.z_dim] + hidden_sizes + [self.input_size],
                    activation=activate_fct,
                    output_activation=None,
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )

            self.encoder_gate = MLP(
                [self.z_dim] + hidden_sizes + [[self.input_size, 1]],
                activation=activate_fct,
                output_activation=[None, Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            ) 

        self.codebook = MLP(
            [self.code_size] + hidden_sizes + [[z_dim,z_dim]],
            activation=activate_fct,
            output_activation=[None,Exp],
            post_layer_fct=post_layer_fct,
            post_act_fct=post_act_fct,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        if self.use_cuda:
            self.cuda()

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
        xs = SoftmaxTransform()(xs)
        return xs
    
    def sigmoid(self, xs):
        sigm_enc = nn.Sigmoid()
        xs = sigm_enc(xs)
        xs = clamp_probs(xs)
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

    def model(self, xs, ks2=None):
        pyro.module('sure', self)

        total_count = pyro.param("inverse_dispersion", self.inverse_dispersion *
                                 xs.new_ones(self.input_size), constraint=constraints.positive)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        self.options = dict(dtype=xs.dtype, device=xs.device)

        I = torch.eye(self.code_size)
        acs_loc,acs_scale = self.codebook(I)

        with pyro.plate('data'):
            prior = torch.zeros(batch_size, self.code_size, **self.options)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=prior))

            prior_loc = torch.matmul(ns,acs_loc)
            #prior_scale = torch.ones_like(prior_loc)
            prior_scale = torch.matmul(ns,acs_scale)

            if self.use_laplacian:
                zns = pyro.sample('zn', dist.Laplace(prior_loc, prior_scale).to_event(1))
            else:
                zns = pyro.sample('zn', dist.Normal(prior_loc, prior_scale).to_event(1))

            if self.use_undesired:
                zs = [ks2, zns]
            else:
                zs = zns

            if self.loss_func == 'gaussian':
                concentrate_loc, concentrate_scale = self.decoder_concentrate(zs)
                concentrate = concentrate_loc
            else:
                concentrate = self.decoder_concentrate(zs)

            if self.dist_model == 'dmm':
                concentrate = self.dirimulti_param(concentrate)
                theta = dist.DirichletMultinomial(total_count=1, concentration=concentrate).mean
            elif self.dist_model == 'mm':
                probs = self.multi_param(concentrate)
                theta = dist.Multinomial(total_count=1, probs=probs).mean

            if self.use_zeroinflate:
                gate_loc = self.gate_prior * torch.ones(batch_size, self.input_size, **self.options)
                gate_scale = torch.ones(batch_size, self.input_size, **self.options)
                gate_logits = pyro.sample('gate_logit', dist.Normal(gate_loc, gate_scale).to_event(1))
                gate_probs = self.sigmoid(gate_logits)

                if self.use_exact_zeroinflate:
                    if self.loss_func == 'multinomial':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)
                else:
                    if self.loss_func != 'gaussian':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **self.options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one', dist.Binomial(probs=1-gate_probs).to_event(1), obs=ones)

            if self.loss_func == 'negbinomial':
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count, probs=theta),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.NegativeBinomial(total_count=total_count, probs=theta).to_event(1), obs=xs)
            elif self.loss_func == 'poisson':
                rate = xs.sum(1).unsqueeze(-1) * theta
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Exponential(rate=rate).to_event(1), obs=xs)
            elif self.loss_func == 'multinomial':
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta), obs=xs)
            elif self.loss_func == 'gaussian':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Normal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Normal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)
            elif self.loss_func == 'lognormal':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.LogNormal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)

    def guide(self, xs, ks2=None):
        with pyro.plate('data'):
            zn_loc, zn_scale = self.encoder_zn(xs)

            if self.use_laplacian:
                zns = pyro.sample('zn', dist.Laplace(zn_loc, zn_scale).to_event(1))
            else:
                zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            alpha = self.encoder_n(zns)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=alpha))
            
            if self.use_zeroinflate:
                if self.use_undesired:
                    zs=[ks2,zns]
                else:
                    zs=zns
                loc, scale = self.encoder_gate(zs)
                scale = self.cutoff(scale)
                gates_logit = pyro.sample('gate_logit', dist.Normal(loc, scale).to_event(1))

    def get_metacell_coordinates(self):
        I = torch.eye(self.code_size, **self.options)
        cb,_ = self.codebook(I)
        return cb
    
    def get_metacell_expressions(self):
        cbs = self.get_metacell_coordinates()
        concentrate = self._expression(cbs)
        return concentrate
    
    def get_metacell_counts(self, total_counts_per_item=1e6):
        concentrate = self.get_metacell_expressions()
        theta = self._count(concentrate, total_counts_per_cell=total_counts_per_item)
        return theta

    def get_cell_coordinates(self, xs, is_query=False, use_soft=False):
        if is_query:
            cb = self.get_metacell_coordinates()
            if use_soft:
                A = self.soft_assignments(xs)
            else:
                A = self.hard_assignments(xs)
            zns = torch.matmul(A, cb)
        else:
            zns, _ = self.encoder_zn(xs)
        return zns
    
    def _code(self, xs):
        zns,_ = self.encoder_zn(xs)
        alpha = self.encoder_n(zns)
        return alpha
    
    def soft_assignments(self, xs):
        alpha = self._code(xs)
        alpha = self.softmax(alpha)
        return alpha
    
    def hard_assignments(self, xs):
        alpha = self._code(xs)
        res, ind = torch.topk(alpha, 1)
        ns = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ns
    
    def _expression(self,zns):
        if self.use_undesired:
            ks2 = torch.zeros(zns.shape[0], self.undesired_size, **self.options)
            zs=[ks2,zns]
        else:
            zs=zns

        if not (self.loss_func in ['gaussian','lognormal']):
            concentrate = self.decoder_concentrate(zs)
        else:
            concentrate,_ = self.decoder_concentrate(zs)

        return concentrate
    
    def _count(self,concentrate,total_counts_per_cell=1e6):
        if self.dist_model == 'dmm':
            concentrate = self.dirimulti_param(concentrate)
            theta = dist.DirichletMultinomial(total_count=1, concentration=concentrate).mean
        elif self.dist_model == 'mm':
            probs = self.multi_param(concentrate)
            theta = dist.Multinomial(total_count=1, probs=probs).mean

        theta = theta * total_counts_per_cell
        return theta

    def get_cell_expressions(self, xs, is_query=False):
        zns = self.get_cell_coordinates(xs, is_query=is_query)
        concentrate = self._expression(zns)
        return concentrate
    
    def get_cell_counts(self, xs, total_counts_per_cell=1e6, is_query=False):
        concentrate = self.get_cell_expressions(xs, is_query=is_query)
        theta = self._count(concentrate,total_counts_per_cell)
        return theta
    



def run_inference_for_epoch(data_loader, losses, use_cuda=True):
    num_losses = len(losses)
    batches = len(data_loader)
    epoch_losses = [0.0] * num_losses
    sup_iter = iter(data_loader)

    for i in range(batches):
        (xs, ks2) = next(sup_iter)

        for loss_id in range(num_losses):
            new_loss = losses[loss_id].step(xs, ks2)
            epoch_losses[loss_id] += new_loss

    return epoch_losses




def main(args):
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)

    if args.float64:
        torch.set_default_dtype(torch.float64)

    data_loader = None
    data_num = 0
    if args.data_file is not None:
        data_loader = setup_data_loader(SingleCellCached, args.data_file, args.undesired_factor_file, 
                                        args.log_transform, args.expm1, args.cuda, args.float64, args.batch_size)
        data_num = len(data_loader)

    undesired_size = data_loader.dataset.num_undesired
    input_size = data_loader.dataset.data.shape[1]
    undesired_factor = data_loader.dataset.undesired_factor

    dist_model = 'mm'
    if args.use_dirichlet:
        dist_model = 'dmm'

    use_undesired = True
    if args.undesired_factor_file is None:
        use_undesired = False

    use_zeroinflate=False
    use_exact_zeroinflate=False
    if args.zero_inflation=='exact':
        use_exact_zeroinflate=True
    elif args.zero_inflation=='inexact':
        use_zeroinflate=True

    proj_name=''
    if args.save_model is not None:
        proj_name=Path(args.save_model).stem

    if args.save_intermediate_model:
        tmp_dir='./{}_tmp_models'.format(proj_name)

    sure = SURE(
        input_size=input_size,
        undesired_size=undesired_size,
        inverse_dispersion=args.inverse_dispersion,
        z_dim=args.z_dim,
        hidden_layers=args.hidden_layers,
        hidden_layer_activation=args.hidden_layer_activation,
        use_undesired=use_undesired,
        undesired_factor=undesired_factor,
        use_cuda=args.cuda,
        config_enum=args.enum_discrete,
        dist_model=dist_model,
        use_zeroinflate=use_zeroinflate,
        use_exact_zeroinflate=use_exact_zeroinflate,
        gate_prior=args.gate_prior,
        delta=args.delta,
        loss_func=args.likelihood,
        dirimulti_mass=args.dirichlet_mass,
        nn_dropout=args.layer_dropout_rate,
        post_layer_fct=args.post_layer_function,
        post_act_fct=args.post_activation_function,
        effect_size_estimator=args.effect_size_estimator,
        use_laplacian = args.laplace,
        code_size=args.code_dim,
    )

    adam_params = {'lr': args.learning_rate, 'betas': (args.beta_1, 0.999), 'weight_decay': 0.005}
    optimizer = torch.optim.Adam
    decayRate = args.decay_rate
    scheduler = ExponentialLR({'optimizer': optimizer, 'optim_args': adam_params, 'gamma': decayRate})

    pyro.clear_param_store()

    guide = config_enumerate(sure.guide, args.enum_discrete, expand=True)
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=1, strict_enumeration_warning=False)
    loss_basic = SVI(sure.model, guide, scheduler, loss=elbo)

    losses = [loss_basic]

    try:
        logger = open(args.logfile, 'w') if args.logfile else None

        tr_start = tm.time()
        for i in range(0, args.num_epochs):
            ep_tr_start = tm.time()

            epoch_losses = run_inference_for_epoch(data_loader, losses, args.cuda)

            avg_epoch_losses_ = map(lambda v: v / data_num, epoch_losses)
            avg_epoch_losses = map(lambda v: "{:.4f}".format(v), avg_epoch_losses_)

            str_loss = " ".join(map(str, avg_epoch_losses))

            str_print = "{} epoch: avg losses {}".format(
                i+1, str_loss
            )

            ep_tr_time = tm.time() - ep_tr_start
            str_print += " elapsed {:.4f} seconds".format(ep_tr_time)

            if i % args.decay_epochs == 0:
                scheduler.step()

            if True:
                if args.save_intermediate_model:
                    if not os.path.exists(tmp_dir):
                        mkdir_p(tmp_dir)
                    torch.save(sure, '{}/{}_{}.pth'.format(tmp_dir,proj_name,i))

            if (i+1) == args.num_epochs:
                if args.save_model is not None:
                    torch.save(sure, args.save_model)

            print_and_log(logger, str_print)

        tr_time = tm.time()-tr_start
        if args.runtime:
            print('running time: {} secs'.format(tr_time))

    finally:
        if args.logfile:
            logger.close()


EXAMPLE_RUN = (
    "example run: python SURE.py --help"
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SURE\n{}".format(EXAMPLE_RUN))

    parser.add_argument(
        "--cuda", action="store_true", help="use GPU(s) to speed up training"
    )
    parser.add_argument(
        "--jit", action="store_true", help="use PyTorch jit to speed up training"
    )
    parser.add_argument(
        "-n", "--num-epochs", default=40, type=int, help="number of epochs to run"
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
        "-undesired",
        "--undesired-factor-file",
        default=None,
        type=str,
        help="the file for the record of undesired factors",
    )
    parser.add_argument(
        "-ese",
        "--effect-size-estimator",
        default='linear',
        type=str,
        choices=['linear', 'nonlinear'],
        help="specify method for effect size estimation",
    )
    parser.add_argument(
        "-delta",
        "--delta",
        default=0.0,
        type=float,
        help="penalty weight for zero inflation loss",
    )
    parser.add_argument(
        "-64",
        "--float64",
        action="store_true",
        help="use double float precision",
    )
    parser.add_argument(
        "-lt",
        "--log-transform",
        action="store_true",
        help="run log-transform on count data",
    )
    parser.add_argument(
        "-la",
        "--laplace",
        action="store_true",
        help="use laplace distribution for latent representation",
    )
    parser.add_argument(
        "-cd",
        "--code-dim",
        default=100,
        type=int,
        help="size of the latent sparse measurement for compressed sensing",
    )
    parser.add_argument(
        "-zd",
        "--z-dim",
        default=10,
        type=int,
        help="size of the tensor representing the latent variable z "
        "variable (handwriting style for our MNIST dataset)",
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
        "-id",
        "--inverse-dispersion",
        default=10.0,
        type=float,
        help="inverse dispersion prior for negative binomial",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.0001,
        type=float,
        help="learning rate for Adam optimizer",
    )
    parser.add_argument(
        "-dr",
        "--decay-rate",
        default=0.9,
        type=float,
        help="decay rate for Adam optimizer",
    )
    parser.add_argument(
        "-de",
        "--decay-epochs",
        default=20,
        type=int,
        help="decay learning rate every #epochs",
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
        "-bs",
        "--batch-size",
        default=1000,
        type=int,
        help="number of images (and labels) to be considered in a batch",
    )
    parser.add_argument(
        "-expm1",
        "--expm1",
        action="store_true",
        help="turn on exponential transformation",
    )
    parser.add_argument(
        "-gp",
        "--gate-prior",
        default=0.6,
        type=float,
        help="gate prior for zero-inflated model",
    )
    parser.add_argument(
        "-likeli",
        "--likelihood",
        default='negbinomial',
        type=str,
        choices=['negbinomial', 'multinomial', 'poisson', 'gaussian','lognormal'],
        help="specify the distribution likelihood function",
    )
    parser.add_argument(
        "-dirichlet",
        "--use-dirichlet",
        action="store_true",
        help="use Dirichlet distribution over gene frequency",
    )
    parser.add_argument(
        "-mass",
        "--dirichlet-mass",
        default=1,
        type=float,
        help="mass param for dirichlet model",
    )
    parser.add_argument(
        "-zi",
        "--zero-inflation",
        default='none',
        type=str,
        choices=['none','exact','inexact'],
        help="use zero-inflated estimation",
    )
    parser.add_argument(
        "-rt",
        "--runtime",
        action="store_true",
        help="print running time",
    )
    parser.add_argument(
        "-log",
        "--logfile",
        default="./tmp.log",
        type=str,
        help="filename for logging the outputs",
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
    parser.add_argument(
        "--save-intermediate-model",
        action="store_true",
        help="path to save intermediate model in the tmp_models directory",
    )
    args = parser.parse_args()

    assert (
        (args.data_file is not None) and (
            os.path.exists(args.data_file))
    ), "data file must be provided"

    main(args)
