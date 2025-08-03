import torch
from torch.utils.data import DataLoader

import pyro
import pyro.distributions as dist

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors
import scanpy as sc

from tqdm import tqdm

from ..utils import convert_to_tensor, tensor_to_numpy
from ..utils import CustomDataset2


def codebook_generate(sure_model, n_samples):
    code_weights = convert_to_tensor(sure_model.codebook_weights, dtype=sure_model.dtype, device=sure_model.get_device())
    ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])

    codebook_loc, codebook_scale = sure_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=sure_model.dtype, device=sure_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=sure_model.dtype, device=sure_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()
    return tensor_to_numpy(zs), tensor_to_numpy(ns)


def codebook_sample(sure_model, xs, n_samples, even_sample=False, filter=True):
    xs = convert_to_tensor(xs, dtype=sure_model.dtype, device=sure_model.get_device())
    assigns = sure_model.soft_assignments(xs)
    code_assigns = np.argmax(assigns, axis=1)
    
    if even_sample:        
        repeat = n_samples // assigns.shape[1]
        remainder = n_samples % assigns.shape[1]
        ns_id = np.repeat(np.arange(1, assigns.shape[1] + 1), repeat)
        # 补充剩余元素（将前 `remainder` 个数字各多重复1次）
        if remainder > 0:
            ns_id = np.concatenate([ns_id, np.arange(1, remainder + 1)])
        ns_id -= 1
        
        ns = LabelBinarizer().fit_transform(ns_id)
        ns = convert_to_tensor(ns, dtype=sure_model.dtype, device=sure_model.get_device())
    else:
        code_weights = codebook_weights(assigns)
        code_weights = convert_to_tensor(code_weights, dtype=sure_model.dtype, device=sure_model.get_device())
        ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])
        ns_id = np.argmax(tensor_to_numpy(ns), axis=1)

    codebook_loc, codebook_scale = sure_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=sure_model.dtype, device=sure_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=sure_model.dtype, device=sure_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()

    xs_zs = sure_model.get_cell_coordinates(xs)
    #xs_zs = convert_to_tensor(xs_zs, dtype=sure_model.dtype, device=sure_model.get_device())
    #xs_dist = torch.cdist(zs, xs_zs)
    #idx = xs_dist.argmin(dim=1)
    
    nbrs = NearestNeighbors(n_jobs=-1, n_neighbors=1)
    nbrs.fit(tensor_to_numpy(xs_zs))
    idx = nbrs.kneighbors(tensor_to_numpy(zs), return_distance=False)
    idx_ = idx.flatten()
    #idx = [idx_[i] for i in np.arange(n_samples) if np.array_equal(code_assigns[idx_[i]], ns[i])]
    df = pd.DataFrame({'idx':idx_,
                       'to':code_assigns[idx_],
                       'from':ns_id})
    if filter:
        filtered_df = df[df['from'] != df['to']]
    else:
        filtered_df = df
    idx = filtered_df.loc[:,'idx'].values
    ns_id = filtered_df.loc[:,'from'].values

    return tensor_to_numpy(xs[idx]), tensor_to_numpy(idx), ns_id


def codebook_sketch(sure_model, xs, n_samples, even_sample=False):
    return codebook_sample(sure_model, xs, n_samples, even_sample)

def codebook_bootstrap_sketch(sure_model, xs, n_samples, n_neighbors=8, aggregate_means='mean', pval=1e-12, even_sample=False, filter=True):
    xs = convert_to_tensor(xs, dtype=sure_model.dtype, device=sure_model.get_device())
    xs_zs = sure_model.get_cell_coordinates(xs)
    xs_zs = tensor_to_numpy(xs_zs)

    # generate samples that follow the metacell distribution of the given data
    assigns = sure_model.soft_assignments(xs)
    code_assigns = np.argmax(assigns,axis=1)
    if even_sample:
        repeat = n_samples // assigns.shape[1]
        remainder = n_samples % assigns.shape[1]
        ns_id = np.repeat(np.arange(1, assigns.shape[1] + 1), repeat)
        # 补充剩余元素（将前 `remainder` 个数字各多重复1次）
        if remainder > 0:
            ns_id = np.concatenate([ns_id, np.arange(1, remainder + 1)])
        ns_id -= 1
        
        ns = LabelBinarizer().fit_transform(ns_id)
        ns = convert_to_tensor(ns, dtype=sure_model.dtype, device=sure_model.get_device())
    else:
        code_weights = codebook_weights(assigns)
        code_weights = convert_to_tensor(code_weights, dtype=sure_model.dtype, device=sure_model.get_device())
        ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])
        ns_id = np.argmax(tensor_to_numpy(ns), axis=1)

    codebook_loc, codebook_scale = sure_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=sure_model.dtype, device=sure_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=sure_model.dtype, device=sure_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()
    zs = tensor_to_numpy(zs)

    # find the neighbors of sample data in the real data space
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(xs_zs)
        
    xs_list = []
    ns_list = []
    distances, ids = nbrs.kneighbors(zs, return_distance=True)
    dist_pdf = gaussian_kde(distances.flatten())

    xs = tensor_to_numpy(xs)    
    sketch_cells = dict()
    with tqdm(total=n_samples, desc='Sketching', unit='sketch') as pbar:
        for i in np.arange(n_samples):
            cells_i_ = ids[i, dist_pdf(distances[i]) > pval]
            #cells_i = [c for c in cells_i_ if np.array_equal(code_assigns[c],ns[i])]
            df = pd.DataFrame({'idx':cells_i_,
                               'to': code_assigns[cells_i_],
                               'from': [ns_id[i]] * len(cells_i_)})
            if filter:
                filtered_df = df[df['from'] != df['to']]
            else:
                filtered_df = df
            cells_i = filtered_df.loc[:,'idx'].values
            ns_i = filtered_df.loc[:,'from'].unique()

            if len(cells_i)>0:
                xs_i = xs[cells_i]
                if aggregate_means == 'mean':
                    xs_i = np.mean(xs_i, axis=0, keepdims=True)
                elif aggregate_means == 'median':
                    xs_i = np.median(xs_i, axis=0, keepdims=True)
                elif aggregate_means == 'sum':
                    xs_i = np.sum(xs_i, axis=0, keepdims=True)

                xs_list.append(xs_i)
                ns_list.extend(ns_i)
                sketch_cells[i] = cells_i 

            pbar.update(1)

    return np.vstack(xs_list),sketch_cells,ns_list



def codebook_summarize_(assigns, xs):
    assigns = convert_to_tensor(assigns)
    xs = convert_to_tensor(xs)
    results = torch.matmul(assigns.T, xs)
    results = results / torch.sum(assigns.T, dim=1, keepdim=True)
    return tensor_to_numpy(results)


def codebook_summarize(assigns, xs, batch_size=1024):
    assigns = convert_to_tensor(assigns)
    xs = convert_to_tensor(xs)

    dataset = CustomDataset2(assigns, xs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    R = None
    W = None
    with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
        for A_batch, X_batch, _ in dataloader:
            r = torch.matmul(A_batch.T, X_batch)
            w = torch.sum(A_batch.T, dim=1, keepdim=True)
            if R is None:
                R = r 
                W = w 
            else:
                R += r 
                W += w 
            pbar.update(1)

    results = R / W
    return tensor_to_numpy(results)

def codebook_aggregate(assigns, xs, batch_size=1024):
    assigns = convert_to_tensor(assigns)
    xs = convert_to_tensor(xs)

    dataset = CustomDataset2(assigns, xs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    R = None
    with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
        for A_batch, X_batch, _ in dataloader:
            r = torch.matmul(A_batch.T, X_batch)
            if R is None:
                R = r 
            else:
                R += r 
            pbar.update(1)

    results = R
    return tensor_to_numpy(results)


def codebook_weights(assigns):
    assigns = convert_to_tensor(assigns)
    results = torch.sum(assigns, dim=0)
    results = results / torch.sum(results)
    return tensor_to_numpy(results)

