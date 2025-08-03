import os 
import tempfile
import subprocess 
import scanpy as sc
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.stats import gaussian_kde

if sp.__version__ < '1.14.0':
    from scipy.integrate import cumtrapz
else:
    from scipy.integrate import cumulative_trapezoid as cumtrapz

import pandas as pd
from functools import reduce

import datatable as dt
from tqdm import tqdm
import umap 
import faiss 
from sklearn.neighbors import NearestNeighbors
#from cuml.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,LabelBinarizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pyro.distributions as dist

from ..SURE import SURE
from .assembly import assembly,get_data,get_subdata,batch_encoding,get_uns
from ..codebook import codebook_summarize_,codebook_generate,codebook_sketch
from ..utils import convert_to_tensor, tensor_to_numpy
from ..utils import CustomDataset
from ..utils import pretty_print, Colors
from ..utils import PriorityQueue

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import dill as pickle
import gzip 
from packaging.version import Version
torch_version = torch.__version__

from typing import Literal

import warnings
warnings.filterwarnings("ignore")

class SingleOmicsAtlas(nn.Module):
    """
    Compressed Cell Atlas

    Parameters
    ----------
    atlas_name
        Name of the built atlas.
    hvgs
        Highly variable genes.
    eps
        Low bound.
    """
    def __init__(self, 
                 atlas_name: str = 'Atlas', 
                 hvgs: list = None, 
                 eps: float = 1e-12):
        super().__init__()
        self.atlas_name = atlas_name
        self.model = None 
        self.sure_models_list = None 
        self.hvgs = hvgs 
        self.adata = None
        self.sample_adata = None
        self.layer = None
        self.n_sure_models = None
        self.umap_metric='euclidean'
        self.umap = None
        self.adj = None
        self.subatlas_list = None 
        self.n_subatlas = 0
        self.pheno_keys = None
        self.nearest_neighbor_engine = None
        self.knn_k = 5
        self.network = None
        self.network_pos = None
        self.sample_network = None 
        self.sample_network_pos = None
        self.eps=eps

    def fit(self, adata_list_, 
            batch_key: str = None, 
            pheno_keys: list = None, 
            preprocessing: bool = True, 
            hvgs: list = None,
            n_top_genes: int = 5000, 
            hvg_method: Literal['seurat','seurat_v3','cell_ranger'] ='seurat', 
            layer: str = 'counts', 
            cuda_id: int = 0, 
            use_jax: bool = False,
            codebook_size: int = 800, 
            codebook_size_per_adata: int = 500, 
            min_gamma=20,
            min_cell=1000,
            learning_rate: float = 0.0001,
            batch_size: int = 200, 
            batch_size_per_adata: int = 100, 
            n_epochs: int = 200, 
            latent_dist: Literal['normal','laplacian','studentt','cauchy'] = 'studentt',
            use_dirichlet: bool = False, 
            use_dirichlet_per_adata: bool = False, 
            zero_inflation: bool = False, 
            zero_inflation_per_adata: bool = False,
            likelihood: Literal['negbinomial','poisson','multinomial','gaussian'] = 'multinomial', 
            likelihood_per_adata: Literal['negbinomial','poisson','multinomial','gaussian'] = 'multinomial', 
            #n_samples_per_adata: int = 10000, 
            #total_samples: int = 500000, 
            #summarize: bool = True,
            sketch_func: Literal['mean','sum','simulate','sample','bootstrap_sample'] = 'mean',
            n_bootstrap_neighbors: int = 8,
            #sketching: bool = True, 
            #even_sketch: bool = True,
            #bootstrap_sketch: bool = False, 
            #n_sketch_neighbors: int = 10, 
            n_workers: int = 1,
            mute: bool = True,
            edge_thresh: float = 0.001,
            metric: Literal['euclidean','correlation','cosine'] ='euclidean',
            knn_k: int = 5):
        """
        Fit the input list of AnnData datasets.

        Parameters
        ----------
        adata_list
            A list of AnnData datasets.
        batch_key
            Undesired factor. 
        pheno_keys
            A list of phenotype factors, of which the information should be retained in the built atlas.
        preprocessing
            If toggled on, the input datasets will go through the standard Scanpy proprocessing steps including normalization and log1p transformation.
        hvgs
            If a list of highly variable genes is given, the subsequent steps will rely on these genes.
        n_top_genes
            Parameter for Scanpy's highly_variable_genes
        hvg_method
            Parameter for Scanpy's highly_variable_genes
        layer
            Data used for building the atlas.
        cuda_id
            Cuda device.
        use_jax
            If toggled on, Jax will be used for speeding.
        codebook_size
            Size of metacells in the built atlas.
        codebook_size_per_adata
            Size of metacells for each adata.
        learning_rate
            Parameter for optimization.
        batch_size
            Parameter for building the atlas.
        batch_size_per_adata
            Parameter for calling metacells within each adata.
        n_epochs
            Number of epochs.
        latent_dist
            Distribution for latent representations.
        use_dirichlet
            Use Dirichlet model for building the atlas.
        use_dirichlet_per_adata
            Use Dirichlet model for calling metacells within each adata.
        zero_inflation
            Use zero-inflated model for building the atlas.
        zero_inflation_per_adata
            Use zero-inflated model for calling metacells within each adata.
        likelihood
            Data generation model for building the atlas.
        likelihood_per_adata
            Data generation model for calling metacells within each adata.
        n_samples_per_adata
            Number of samples drawn from each adata for building the atlas.
        total_samples
            Total number of samples for building the atlas.
        sketching
            If toggled on, sketched cells will be used for building the atlas.
        bootstrap_sketch
            If toggled on, bootstraped sketching will be used instead of simple sketching.
        n_sketch_neighbors
            Parameter for bootstraped sketching.
        edge_thresh
            Parameter for building network.
        metric
            Parameter for UMAP.
        knn_k
            Parameter for K-nearest-neighbor machine.
        """
        
        print(Colors.YELLOW + 'Create A Distribution-Preserved Single-Cell Omics Atlas' + Colors.RESET)
        adata_list = [ad for ad in adata_list_ if ad.shape[0]>min_cell]
        n_adatas = len(adata_list)
        self.layer = layer
        self.n_sure_models = n_adatas
        self.umap_metric = metric
        self.pheno_keys = pheno_keys
        #zero_inflation = True if sketching else zero_inflation

        # assembly
        print(f'{n_adatas} adata datasets are given')
        self.model,self.submodels,self.hvgs = assembly(adata_list, batch_key, 
                 preprocessing, hvgs, n_top_genes, hvg_method, layer, cuda_id, use_jax,
                 codebook_size, codebook_size_per_adata, min_gamma, learning_rate,
                 batch_size, batch_size_per_adata, n_epochs, latent_dist,
                 use_dirichlet, use_dirichlet_per_adata,
                 zero_inflation, zero_inflation_per_adata,
                 likelihood, likelihood_per_adata,
                 #n_samples_per_adata, total_samples, summarize,
                 sketch_func,n_bootstrap_neighbors,
                 #sketching, even_sketch, bootstrap_sketch, n_sketch_neighbors, 
                 n_workers, mute)
        
        # summarize expression
        X,W,adj = None,None,None
        with tqdm(total=n_adatas, desc=f'Summarize data in {layer}', unit='adata') as pbar:
            for i in np.arange(n_adatas):
                #print(f'Adata {i+1} / {n_adatas}: Summarize data in {layer}')
                adata_i = adata_list[i][:,self.hvgs].copy()
                adata_i_ = adata_list[i].copy()

                xs_i = get_data(adata_i, layer).values
                xs_i_ = get_data(adata_i_, layer).values
                ws_i_sup = self.model.soft_assignments(xs_i)
                xs_i_sup = codebook_summarize_(ws_i_sup, xs_i_)

                if X is None:
                    X = xs_i_sup
                    W = np.sum(ws_i_sup.T, axis=1, keepdims=True)

                    a = convert_to_tensor(ws_i_sup)
                    a_t = a.T / torch.sum(a.T, dim=1, keepdim=True)
                    adj = torch.matmul(a_t, a)
                else:
                    X += xs_i_sup
                    W += np.sum(ws_i_sup.T, axis=1, keepdims=True)

                    a = convert_to_tensor(ws_i_sup)
                    a_t = a.T / torch.sum(a.T, dim=1, keepdim=True)
                    adj += torch.matmul(a_t, a)
                    
                pbar.update(1)
        X = X / W 
        self.adata = sc.AnnData(X)
        self.adata.obs_names = [f'MC{x}' for x in self.adata.obs_names]
        self.adata.var_names = adata_i_.var_names

        adj = tensor_to_numpy(adj) / self.n_sure_models
        self.adj = (adj + adj.T) / 2
        n_nodes = adj.shape[0]
        self.adj[np.arange(n_nodes), np.arange(n_nodes)] = 0

        # summarize phenotypes
        if pheno_keys is not None:
            self._summarize_phenotypes_from_adatas(adata_list, pheno_keys)

        # COMMENT OUT 2025.7.4
        # compute visualization position for the atlas
        #print('Compute the reference position of the atlas')
        #n_samples = np.max([n_samples_per_adata * self.n_sure_models, 50000])
        #n_samples = np.min([n_samples, total_samples])
        #self.instantiation(n_samples)
        #
        # create nearest neighbor indexing
        #self.build_nearest_neighbor_engine(knn_k)
        #self.knn_k = knn_k
        #
        #self.build_network(edge_thresh=edge_thresh)
        # END OF COMMENT OUT 2025.7.4
        
        # the distribution of cell-to-metacell distances
        metacells = self.model.get_metacell_coordinates()
        self.nearest_metacell_engine = NearestNeighbors(n_neighbors=knn_k,n_jobs=-1)
        self.nearest_metacell_engine.fit(metacells)
        
        cell2metacell_distances = []
        with tqdm(total=n_adatas, desc='Build cell-to-metacell distance distribution', unit='adata') as pbar:
            for i in np.arange(n_adatas):
                #print(f'Adata {i+1} / {n_adatas}: Build cell-to-metacell distance distribution')
                adata_i = adata_list[i][:,self.hvgs].copy()

                xs_i = get_data(adata_i, layer).values
                zs_i = self.model.get_cell_coordinates(xs_i)

                dd_i,_ = self.nearest_metacell_engine.kneighbors(zs_i, n_neighbors=1)
                cell2metacell_distances.extend(dd_i.flatten())
                
                pbar.update(1)
            
        self.cell2metacell_dist = gaussian_kde(cell2metacell_distances)
        
        print(Colors.YELLOW + f'A distribution-preserved atlas has been built from {n_adatas} adata datasets.' + Colors.RESET)
        
    def detect_outlier(self, adata_query, thresh:float = 1e-2, batch_size:int = 1024):
        
        def batch_p_values_greater(kde, x_vector):
            """
            批量计算向量x中每个元素的P(X > x)
            
            参数:
                kde: 已拟合的gaussian_kde对象
                x_vector: 包含多个x值的数组
                
            返回:
                与x_vector形状相同的P(X > x)数组
            """
            # 创建高分辨率的CDF查找表
            x_min = min(kde.dataset.min(), x_vector.min()) - 3 * np.std(kde.dataset)
            x_max = max(kde.dataset.max(), x_vector.max()) + 3 * np.std(kde.dataset)
            grid_points = max(20000, len(kde.dataset))  # 确保足够密集
            x_grid = np.linspace(x_min, x_max, grid_points)
            
            # 计算PDF和CDF
            pdf = kde(x_grid)
            cdf = np.cumsum(pdf)
            cdf /= cdf[-1]  # 归一化
            
            # 使用插值查找每个x对应的P(X > x) = 1 - CDF(x)
            p_values = 1 - np.interp(x_vector, x_grid, cdf)
            
            # 处理边界外的值
            p_values[x_vector < x_min] = 1.0
            p_values[x_vector > x_max] = 0.0
            
            return p_values
        
        adata_query = adata_query.copy()
        X_query = get_subdata(adata_query, self.hvgs, self.layer).values
        Z_map = self.model.get_cell_coordinates(X_query, batch_size=batch_size)
        
        dd,_ = self.nearest_metacell_engine.kneighbors(Z_map, n_neighbors=1)
        #pp = self.cell2metacell_dist(dd.flatten())
        pp = batch_p_values_greater(self.cell2metacell_dist,dd)
        outliers = pp < thresh
        
        print(f'{np.sum(outliers)} outliers found.')
        
        return outliers
        
    def map(self, adata_query, 
            batch_size: int = 1024):
        """
        Map query data to the atlas.

        Parameters
        ----------
        adata_query
            Query data. It should be an AnnData object.
        batch_size
            Size of batch processing.
        """
        adata_query = adata_query.copy()
        X_query = get_subdata(adata_query, self.hvgs, self.layer).values

        Z_map = self.model.get_cell_coordinates(X_query, batch_size=batch_size)
        A_map = self.model.soft_assignments(X_query)

        return Z_map, A_map
    
    def phenotype_density_estimation(self, adata_list, pheno_key):
        n_adatas = len(adata_list)
        PH_MC_list = []
        with tqdm(total=n_adatas, desc=f'Estimate {pheno_key} density', unit='adata') as pbar:
            for i in np.arange(n_adatas):
                ad = adata_list[i][:,self.hvgs].copy()
                xs = get_data(ad, layer=self.layer).values
                X_MC_mat = self.model.soft_assignments(xs)
                lb = LabelBinarizer().fit(ad.obs[pheno_key].astype(str))
                X_PH_mat = lb.transform(ad.obs[pheno_key].astype(str))
                PH_MC_mat = np.matmul(X_PH_mat.T, X_MC_mat)
                
                p_PH = np.sum(PH_MC_mat, axis=1, keepdims=True)
                p_PH[p_PH==0] = 1
                p_MC_PH = PH_MC_mat / p_PH
                PH_MC_mat = p_MC_PH * p_PH
                p_MC = np.sum(PH_MC_mat, axis=0, keepdims=True)
                p_MC[p_MC==0] = 1
                p_PH_MC = PH_MC_mat / p_MC
                
                PH_MC_df = pd.DataFrame(p_PH_MC, 
                                        columns = [f'MC{x}' for x in np.arange(X_MC_mat.shape[1])],
                                        index = lb.classes_)
                PH_MC_list.append(PH_MC_df)
                pbar.update(1)
        
        if n_adatas>1:
            PH_MC_DF = aggregate_dataframes(PH_MC_list)
            PH_MC_DF = PH_MC_DF.div(n_adatas)
        else:
            PH_MC_DF = PH_MC_list[0]
            
        self.adata.uns[f'{pheno_key}_density'] = PH_MC_DF
    
    def summarize_phenotypes(self, adata_list=None, pheno_keys=None):
        self._summarize_phenotypes_from_adatas(adata_list, pheno_keys)

    def _summarize_phenotypes_from_adatas(self, adata_list, pheno_keys):
        n_adatas = len(adata_list)
        for pheno in pheno_keys:
            Y = list()
            with tqdm(total=n_adatas, desc=f'Summarize data in {pheno}', unit='adata') as pbar:
                for i in np.arange(n_adatas):
                    if pheno in adata_list[i].obs.columns:
                        #print(f'Adata {i+1} / {n_adatas}: Summarize data in {pheno}')
                        adata_i = adata_list[i][:,self.hvgs].copy()

                        xs_i = get_data(adata_i, self.layer).values
                        ws_i_sup = self.model.soft_assignments(xs_i)
                        #ys_i = batch_encoding(adata_i, pheno)
                        #columns_i = ys_i.columns.tolist()
                        #ys_i = codebook_summarize_(ws_i_sup, ys_i.values)
                    
                        ws_i = np.argmax(ws_i_sup, axis=1)
                        adata_i.obs['metacell'] = [f'MC{x}' for x in ws_i]
                        df = adata_i.obs[['metacell',pheno]].value_counts().unstack(fill_value=0)

                        Y.append(df)
                        pbar.update(1)

            Y_df_ = aggregate_dataframes(Y)
            Y_df = pd.DataFrame(0, index=[f'MC{x}' for x in np.arange(self.adata.shape[0])], columns=Y_df_.columns)
            Y_df = Y_df.add(Y_df_, fill_value=0)
            
            #Y = Y_df.values
            #Y[Y<self.eps] = 0
            #Y = Y / Y.sum(axis=1, keepdims=True)

            self.adata.uns[pheno] = Y_df
            #self.adata.uns[f'{pheno}_columns'] = Y_df.columns
            Y_hat_ = Y_df.idxmax(axis=1)
            Y_hat = pd.DataFrame(pd.NA, index=[f'MC{x}' for x in np.arange(self.adata.shape[0])], columns=['id'])
            Y_hat.loc[Y_hat_.index.tolist(),'id'] = Y_hat_.tolist()
            self.adata.obs[pheno] = Y_hat['id'].tolist()

    def phenotype_predict(self, adata_query, pheno_key, batch_size=1024):
        _,ws = self.map(adata_query, batch_size)
        A = matrix_dotprod(ws, self.adata.uns[f'{pheno_key}_density'].values)
        A = pd.DataFrame(A, columns=self.adata.uns[f'{pheno_key}_density'].columns)
        return A.idxmax(axis=1).tolist()
    
    @classmethod
    def save_model(cls, atlas, file_path, compression=False):
        """Save the model to the specified file path."""
        file_path = os.path.abspath(file_path)

        atlas.sample_adata = None
        atlas.eval()

        if compression:
            with gzip.open(file_path, 'wb') as pickle_file:
                pickle.dump(atlas, pickle_file)
        else:
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(atlas, pickle_file)

        print(f'Model saved to {file_path}')

    @classmethod
    def load_model(cls, file_path, n_samples=10000):
        """Load the model from the specified file path and return an instance."""
        print(f'Model loaded from {file_path}')

        file_path = os.path.abspath(file_path)
        if file_path.endswith('gz'):
            with gzip.open(file_path, 'rb') as pickle_file:
                atlas = pickle.load(pickle_file)
        else:
            with open(file_path, 'rb') as pickle_file:
                atlas = pickle.load(pickle_file)
        
        #xs = atlas.sample(n_samples)
        #atlas.sample_adata = sc.AnnData(xs)
        #atlas.sample_adata.var_names = atlas.hvgs
        #
        #zs = atlas.model.get_cell_coordinates(xs)
        #ws = atlas.model.soft_assignments(xs)
        #atlas.sample_adata.obsm['X_umap'] = atlas.umap.transform(zs)
        #atlas.sample_adata.obsm['X_sure'] = zs 
        #atlas.sample_adata.obsm['weight'] = ws
        #
        return atlas




#def aggregate_dataframes(df_list):
#    n_dfs = len(df_list)
#    all_columns = set(df_list[0].columns)
#    for i in np.arange(n_dfs-1):
#        all_columns = all_columns.union(set(df_list[i+1].columns))
#        
#    all_indexs = set(df_list[0].index.tolist())
#    for i in np.arange(n_dfs-1):
#        all_indexs = all_indexs.union(set(df_list[i+1].index.tolist()))
#        
#    for col in all_columns:
#        for i in np.arange(n_dfs):
#            if col not in df_list[i]:
#                df_list[i][col] = 0  
#
#    df = pd.DataFrame(0, index=all_indexs, columns=all_columns)
#    df = df_list[0]
#    for i in np.arange(n_dfs-1):
#        df += df_list[i+1]
#
#    #df /= n_dfs
#    return df

def aggregate_dataframes(df_list):
    all_index = reduce(lambda x, y: x.union(y), [df.index for df in df_list], pd.Index([]))
    all_index_sorted = all_index.sort_values()
    all_columns = reduce(lambda x, y: x.union(y), [df.columns for df in df_list], pd.Index([]))
    result = pd.DataFrame(0, index=all_index_sorted, columns=all_columns)
    
    for df in df_list:
        result = result.add(df, fill_value=0)
        
    return result

def smooth_y_over_x(xs, ys, knn_k):
    n = xs.shape[0]
    nbrs = NearestNeighbors(n_neighbors=knn_k, n_jobs=-1)
    nbrs.fit(xs)
    ids = nbrs.kneighbors(xs, return_distance=False)
    ys_smooth = np.zeros_like(ys)
    for i in np.arange(knn_k):
        ys_smooth += ys[ids[:,i]]
    ys_smooth -= ys
    ys_smooth /= knn_k-1
    return ys_smooth

def matrix_dotprod(A, B, dtype=torch.float32):
    A = convert_to_tensor(A, dtype=dtype)
    B = convert_to_tensor(B, dtype=dtype)
    AB = torch.matmul(A, B)
    return tensor_to_numpy(AB)

def matrix_elemprod(A, B):
    A = convert_to_tensor(A)
    B = convert_to_tensor(B)
    AB = A * B
    return tensor_to_numpy(AB)

def cdf(density, xs, initial=0):
    CDF = cumtrapz(density, xs, initial=initial)
    CDF /= CDF[-1]
    return CDF



class FaissKNeighbors:
    def __init__(self, n_neighbors=5):
        self.index = None
        self.k = n_neighbors

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return distances, indices
    

