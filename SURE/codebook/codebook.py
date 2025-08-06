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

import multiprocessing as mp
from tqdm import tqdm

from ..utils import convert_to_tensor, tensor_to_numpy
from ..utils import CustomDataset2

from typing import Literal, List, Tuple, Dict
from functools import partial


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
    
    #nbrs = NearestNeighbors(n_jobs=-1, n_neighbors=1)
    #nbrs.fit(tensor_to_numpy(xs_zs))
    #idx = nbrs.kneighbors(tensor_to_numpy(zs), return_distance=False)
    #idx_ = idx.flatten()
    #idx = [idx_[i] for i in np.arange(n_samples) if np.array_equal(code_assigns[idx_[i]], ns[i])]
    #df = pd.DataFrame({'idx':idx_,
    #                   'to':code_assigns[idx_],
    #                   'from':ns_id})
    #if filter:
    #    filtered_df = df[df['from'] != df['to']]
    #else:
    #    filtered_df = df
    #idx = filtered_df.loc[:,'idx'].values
    #ns_id = filtered_df.loc[:,'from'].values

    nbrs = NearestNeighbors(n_neighbors=50, n_jobs=-1)
    nbrs.fit(tensor_to_numpy(xs_zs))
        
    distances, ids = nbrs.kneighbors(tensor_to_numpy(zs), return_distance=True)
        
    idx,ns_list = [],[]
    with tqdm(total=n_samples, desc='Sketching', unit='sketch') as pbar:
        for i in np.arange(n_samples):
            distances_i = distances[i]
            weights_i = distance_to_softmax_weights(distances_i)
            cell_i_ = weighted_sample(ids[i], weights_i, sample_size=1, replace=False)
            
            df = pd.DataFrame({'idx':[cell_i_],
                               'to': [code_assigns[cell_i_]],
                               'from': [ns_id[i]]})
            if filter:
                filtered_df = df[df['from'] != df['to']]
            else:
                filtered_df = df
            cells_i = filtered_df.loc[:,'idx'].values
            ns_i = filtered_df.loc[:,'from'].unique()

            idx.extend(cells_i)
            ns_list.extend(ns_i)

            pbar.update(1)
            
    return tensor_to_numpy(xs[idx].squeece()), tensor_to_numpy(idx.squeeze()), ns_list


def codebook_sketch(sure_model, xs, n_samples, even_sample=False):
    return codebook_sample(sure_model, xs, n_samples, even_sample)

def codebook_bootstrap_sketch(sure_model, xs, n_samples, n_neighbors=8, 
                              aggregate_fun: Literal['mean','sum'] = 'mean', 
                              even_sample=False, replace=True, filter=True):
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
    nbrs = NearestNeighbors(n_neighbors=50, n_jobs=-1)
    nbrs.fit(xs_zs)
        
    xs_list = []
    ns_list = []
    distances, ids = nbrs.kneighbors(zs, return_distance=True)
    #dist_pdf = gaussian_kde(distances.flatten())

    xs = tensor_to_numpy(xs)    
    sketch_cells = dict()
    with tqdm(total=n_samples, desc='Sketching', unit='sketch') as pbar:
        for i in np.arange(n_samples):
            #cells_i_ = ids[i, dist_pdf(distances[i]) > pval]
            #cells_i = [c for c in cells_i_ if np.array_equal(code_assigns[c],ns[i])]
            distances_i = distances[i]
            weights_i = distance_to_softmax_weights(distances_i)
            cells_i_ = weighted_sample(ids[i], weights_i, sample_size=n_neighbors, replace=replace)
            
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
                if aggregate_fun == 'mean':
                    xs_i = np.mean(xs_i, axis=0, keepdims=True)
                elif aggregate_fun == 'median':
                    xs_i = np.median(xs_i, axis=0, keepdims=True)
                elif aggregate_fun == 'sum':
                    xs_i = np.sum(xs_i, axis=0, keepdims=True)

                xs_list.append(xs_i)
                ns_list.extend(ns_i)
                sketch_cells[i] = cells_i 

            pbar.update(1)

    return np.vstack(xs_list),sketch_cells,ns_list


def process_chunk(chunk_indices: np.ndarray,
                 zs: np.ndarray,
                 xs_zs: np.ndarray,
                 xs: np.ndarray,
                 code_assigns: np.ndarray,
                 ns_id: np.ndarray,
                 n_neighbors: int,
                 replace: bool,
                 filter: bool,
                 aggregate_fun: str) -> Tuple[List[np.ndarray], Dict[int, np.ndarray], List[int]]:
    """
    处理一个chunk的样本
    
    参数:
        chunk_indices: 当前chunk包含的样本索引
        其他参数与主函数相同
        
    返回:
        (xs_chunk, sketch_cells_chunk, ns_list_chunk)
    """
    xs_chunk = []
    sketch_cells_chunk = {}
    ns_list_chunk = []
    
    # 每个chunk创建自己的NearestNeighbors实例，避免多进程冲突
    nbrs = NearestNeighbors(n_neighbors=50, n_jobs=1)  # 单线程模式
    nbrs.fit(xs_zs)
    
    for i in chunk_indices:
        distances, ids = nbrs.kneighbors(zs[i:i+1], return_distance=True)
        distances_i = distances[0]
        weights_i = distance_to_softmax_weights(distances_i)
        cells_i_ = weighted_sample(ids[0], weights_i, sample_size=n_neighbors, replace=replace)
        
        df = pd.DataFrame({
            'idx': cells_i_,
            'to': code_assigns[cells_i_],
            'from': [ns_id[i]] * len(cells_i_)
        })
        
        if filter:
            filtered_df = df[df['from'] != df['to']]
        else:
            filtered_df = df
            
        cells_i = filtered_df.loc[:, 'idx'].values
        ns_i = filtered_df.loc[:, 'from'].unique()
        
        if len(cells_i) > 0:
            xs_i = xs[cells_i]
            
            if aggregate_fun == 'mean':
                xs_i = np.mean(xs_i, axis=0, keepdims=True)
            elif aggregate_fun == 'median':
                xs_i = np.median(xs_i, axis=0, keepdims=True)
            elif aggregate_fun == 'sum':
                xs_i = np.sum(xs_i, axis=0, keepdims=True)
                
            xs_chunk.append(xs_i)
            sketch_cells_chunk[i] = cells_i
            ns_list_chunk.extend(ns_i)
    
    return xs_chunk, sketch_cells_chunk, ns_list_chunk

def codebook_bootstrap_sketch_parallel(
    sure_model, 
    xs, 
    n_samples, 
    n_neighbors=8, 
    aggregate_fun: Literal['mean','sum'] = 'mean', 
    even_sample=False, 
    replace=True, 
    filter=True,
    n_processes: int = None,
    chunk_size: int = 100
) -> Tuple[np.ndarray, Dict[int, np.ndarray], List[int]]:
    """
    基于chunk的并行版本
    
    新增参数:
        n_processes: 并行进程数，None表示使用所有CPU核心
        chunk_size: 每个chunk包含的样本数
    """
    # 转换输入数据 (与原始版本相同)
    xs = convert_to_tensor(xs, dtype=sure_model.dtype, device=sure_model.get_device())
    xs_zs = sure_model.get_cell_coordinates(xs)
    xs_zs = tensor_to_numpy(xs_zs)

    # 生成样本 (与原始版本相同)
    assigns = sure_model.soft_assignments(xs)
    code_assigns = np.argmax(assigns, axis=1)
    
    if even_sample:
        repeat = n_samples // assigns.shape[1]
        remainder = n_samples % assigns.shape[1]
        ns_id = np.repeat(np.arange(1, assigns.shape[1] + 1), repeat)
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

    # 获取codebook (与原始版本相同)
    codebook_loc, codebook_scale = sure_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=sure_model.dtype, device=sure_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=sure_model.dtype, device=sure_model.get_device())

    # 生成zs (与原始版本相同)
    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()
    zs = tensor_to_numpy(zs)

    # 转换为numpy数组 (与原始版本相同)
    xs = tensor_to_numpy(xs)

    # 准备结果容器
    xs_list = []
    sketch_cells = {}
    ns_list = []

    # 分割样本为chunks
    chunks = [np.arange(i, min(i + chunk_size, n_samples)) 
              for i in range(0, n_samples, chunk_size)]
    
    # 创建进程池
    with mp.Pool(processes=n_processes) as pool:
        # 使用partial固定参数
        worker = partial(
            process_chunk,
            zs=zs,
            xs_zs=xs_zs,
            xs=xs,
            code_assigns=code_assigns,
            ns_id=ns_id,
            n_neighbors=n_neighbors,
            replace=replace,
            filter=filter,
            aggregate_fun=aggregate_fun
        )
        
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(worker, chunks),
            total=len(chunks),
            desc='Processing chunks',
            unit='chunk'
        ))
    
    # 合并结果
    for xs_chunk, sketch_cells_chunk, ns_list_chunk in results:
        xs_list.extend(xs_chunk)
        sketch_cells.update(sketch_cells_chunk)
        ns_list.extend(ns_list_chunk)

    return (
        np.vstack(xs_list) if xs_list else np.empty((0, xs.shape[1])),
        sketch_cells,
        ns_list
    )


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


def distance_to_softmax_weights(distances):
    """使用softmax将距离列表转换为概率权重
    
    参数:
        distances: 距离列表，距离越小权重应该越大
        
    返回:
        概率权重数组，和为1
    """
    distances = np.array(distances)
    # 取负数使得距离越小值越大
    negative_distances = -distances
    # 计算softmax
    exp_dist = np.exp(negative_distances - np.max(negative_distances))  # 数值稳定性处理
    softmax = exp_dist / np.sum(exp_dist)
    return softmax

def weighted_sample(items, weights, sample_size=1, replace=True):
    """根据权重进行采样
    
    参数:
        items: 待采样的列表
        weights: 对应的概率权重列表
        sample_size: 采样数量
        replace: 是否允许重复采样
        
    返回:
        采样结果列表
    """
    return np.random.choice(
        a=items,
        size=sample_size,
        p=weights,
        replace=replace
    )
    
def split_evenly(n, m):
    """使用NumPy将n分成m个尽可能平均的数"""
    arr = np.full(m, n // m)
    arr[:n % m] += 1
    return arr.tolist()