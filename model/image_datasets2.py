from mpi4py import MPI
import numpy as np
import torch
import imageio
from natsort import natsorted
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def load_data(*, data_dir, batch_size, image_size, class_cond=False, deterministic=False):
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = DataSet(data_dir)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader
class DataSet(Dataset):
    def __init__(self,train_dir):
        super(DataSet, self).__init__()
        self.root_dir = train_dir
        self.extension = '.tif'
        self.input1_dir = os.path.join(self.root_dir, 'predicted pansharpen')
        self.input2_dir = os.path.join(self.root_dir, 'predicted WFV')
        self.input3_dir = os.path.join(self.root_dir, 'Reference Pan')
        self.input4_dir = os.path.join(self.root_dir, 'Reference Ms')
        self.allnames = self._get_pair_path()
        # assert len(self.allnames_list[0]) == len(self.input2_lists)  

    def __getitem__(self, index):
        allimage,cond= self._load_image_pair(index)
        # patch.append(filename)
        return allimage,cond

    def __len__(self):
        return len(self.allnames[0])

    def _get_pair_path(self):
        names_input1 = natsorted(glob.glob(os.path.join(self.input1_dir, '*' + self.extension)))
        names_input2 = natsorted(glob.glob(os.path.join(self.input2_dir, '*' + self.extension)))
        names_input3 = natsorted(glob.glob(os.path.join(self.input3_dir, '*' + self.extension)))
        names_input4 = natsorted(glob.glob(os.path.join(self.input4_dir, '*' + self.extension)))
        
        allnames=[]
        allnames.append(names_input1[MPI.COMM_WORLD.Get_rank():][::MPI.COMM_WORLD.Get_size()])
        allnames.append(names_input2[MPI.COMM_WORLD.Get_rank():][::MPI.COMM_WORLD.Get_size()])
        allnames.append(names_input3[MPI.COMM_WORLD.Get_rank():][::MPI.COMM_WORLD.Get_size()])
        allnames.append(names_input4[MPI.COMM_WORLD.Get_rank():][::MPI.COMM_WORLD.Get_size()])
        return allnames

    def _load_image_pair(self, idx):
        allimage1=[]
        out_dict = {}
        for i in range(4):
            image = np.float32(imageio.imread(self.allnames[i][idx]))
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            image = np.ascontiguousarray(image.transpose((2, 0, 1)))
            image=torch.div(torch.from_numpy(image),10000.0)
            allimage1.append(image)
        return allimage1,out_dict
