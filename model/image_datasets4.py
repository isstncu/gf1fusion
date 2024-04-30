from mpi4py import MPI
import numpy as np
import torch
import imageio
from natsort import natsorted
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_data(*, data_dir, batch_size, class_cond=False, deterministic=False):
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = DataSet(data_dir)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )
    # while True:
    # yield from loader
    return loader


class DataSet(Dataset):
    def __init__(self, data_dir):
        super(DataSet, self).__init__()
        self.extension = '.tif'
        self.input1_dir = data_dir
        self.allnames = self._get_pair_path()
        # assert len(self.allnames_list[0]) == len(self.input2_lists)  

    def __getitem__(self, index):
        allimage, cond = self._load_image_pair(index)
        # patch.append(filename)
        return allimage, cond

    def __len__(self):
        return len(self.allnames[0])

    def _get_pair_path(self):
        names_input1 = natsorted(glob.glob(os.path.join(self.input1_dir, '*' + self.extension)))
       
        
        
        allnames = []
        allnames.append(names_input1)
        

        return allnames

    def _load_image_pair(self, idx):
        allimage1 = []
        out_dict = {}
        for i in range(1):
            image = np.float32(imageio.imread(self.allnames[i][idx]))  
            if len(image.shape) == 2:  
                image = np.expand_dims(image, axis=-1)
            image = np.ascontiguousarray(image.transpose((2, 0, 1)))  
            image = torch.div(torch.from_numpy(image), 10000.0)
            allimage1.append(image)
        return allimage1, out_dict
