import errno
import os
import numpy as np
from scipy.io import mmread
from pandas import NA, read_csv
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import normalize
from pyro.contrib.examples.util import get_data_directory

import datatable as dt

# transformations for single cell data
def fn_x_scdata(x, log_trans = False, exp_trans=False, use_cuda = True, use_float64 = False):
    if use_float64:
        xp = x.double()
    else:
        xp = x.float()

    if log_trans:
        xp = torch.log1p(xp)

    if exp_trans:
        xp = torch.expm1(xp)

    xp = torch.round(xp)

    # 
    # send the data to GPU(s)
    if use_cuda:
        xp = xp.cuda()
    #
    return xp


def fn_k_scdata(k, use_cuda = True, use_float64 = False):
    if use_float64:
        kp = k.double()
    else:
        kp = k.float()
           
    # send the data to GPU(s)
    if use_cuda:
        kp = kp.cuda()

    return kp


# use one-hot encoding
class SingleCellCached(Dataset):
    def __init__(self, data_file, undesired_file = None, mode = 'data', log_trans = False, exp_trans=False, use_cuda = False, use_float64 = False):
        super(SingleCellCached).__init__()
        #super().__init__(**kwargs)

        self.data = dt.fread(file=data_file, header=True).to_numpy()

        if undesired_file is None:
            self.undesired = None
            self.undesired_factor = None
            self.num_undesired = None
        else:
            self.undesired = dt.fread(undesired_file, header=True).to_pandas().astype(dtype='float')
            self.undesired_factor = self.undesired.columns.tolist()
            self.undesired = self.undesired.values
            self.num_undesired = len(self.undesired_factor)
        
        self.data = torch.from_numpy(self.data)
        if self.undesired is not None:
            self.undesired = torch.from_numpy(self.undesired)
        self.use_cuda = use_cuda
        self.mode = mode

        # transformations on single cell data (normalization and one-hot conversion for labels)
        def transform(x):
            return fn_x_scdata(x, log_trans, exp_trans, use_cuda, use_float64)

        def condition_transform(k):
            return fn_k_scdata(k, use_cuda, use_float64)

        self.data = transform(self.data)
        if self.undesired is not None:
            self.undesired = condition_transform(self.undesired)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        xs = self.data[index]

        if self.undesired is not None:
            ks2 = self.undesired[index]
        else:
            ks2 = ""

        return xs, ks2

def setup_data_loader(
    dataset, 
    data_file, undesired_file,
    log_trans, exp_trans, use_cuda, use_float64,
    batch_size, **kwargs
):
    """
    helper function for setting up pytorch data loaders for a semi-supervised dataset

    :param dataset: the data to use
    :param data_file: the mtx file of single cell data
    :param label_file: the file of class labels
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param kwargs: other params for the pytorch data loader
    :return: three data loader
    """
    # instantiate the dataset as training/testing sets
    if "num_workers" not in kwargs:
        kwargs = {"num_workers": 0, "pin_memory": False}

    cached_data = dataset(
        data_file = data_file, undesired_file = undesired_file,
        log_trans = log_trans, exp_trans = exp_trans, use_cuda = use_cuda, use_float64 = use_float64
    )
    
    loader = DataLoader(
        cached_data, batch_size = batch_size, shuffle = True, **kwargs
    )

    return loader


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


EXAMPLE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
DATA_DIR = os.path.join(EXAMPLE_DIR, "data")
RESULTS_DIR = os.path.join(EXAMPLE_DIR, "results")


