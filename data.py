import datetime
import time
import argparse
import os
import glob
import blobfile as bf
import numpy as np
import torch as th
import torch
import imageio as imageio
from natsort import natsorted
import torch.distributed as dist
from osgeo import gdal, osr
from torchvision.transforms.functional import crop
from image_datasets4 import load_data
import torch.multiprocessing
import scipy.io as scio
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
path="weightMatrix2.mat"
mask=scio.loadmat(path)
weightMatrix = mask['weightMatrix']
def read_image(path):
    image = np.float32(imageio.imread(path))
    if image.shape[0]!=4 and len(image.shape)==3:
             image=image.transpose((2, 0, 1))  
    
    if len(image.shape) == 2: 
             image = np.expand_dims(image, axis=-1)
             image = np.ascontiguousarray(image.transpose((2, 0, 1))) 
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    image = torch.div(torch.from_numpy(image), 10000.0)
    return image
def overlapping_grid_indices(x,input_size,r=None):
    _,c,h,w=x.shape
    r=16 if r is None else r
    h_list = [i for i in range(0,h-input_size+1,r)]
    if (((h-input_size)%r)!=0):
        h_list.append(h-input_size)
    
    w_list = [i for i in range(0,w-input_size+1,r)]
    if (((w-input_size)%r)!=0):
        w_list.append(w-input_size)
    corners=[(i,j) for i in h_list for j in w_list]
    return corners
def write_data(pan_patch,filepath):    
    for i,data in enumerate(pan_patch):
        datatype = gdal.GDT_Float32
        _,band_nums, height, width = data.shape  
        driver = gdal.GetDriverByName("GTiff")
        sample = th.div(data, 0.0001)
        filename = os.path.join(filepath, "image"  + str(i)+'.tif')
        dataset = driver.Create(filename, width, height, band_nums, datatype)
        print(filename)
        for num in range(band_nums):
              dataset.GetRasterBand(num + 1).WriteArray(sample[0][num].cpu().numpy())
        del dataset 
def Crop_Image(crop_image,save_dir):
    image_m=read_image(crop_image)
    print(image_m.shape)
    corners=overlapping_grid_indices(image_m,128,r=32)
    pan_patch=[]
    for (hi, wi) in corners:
        patch=crop(image_m, hi, wi, 128, 128)
        pan_patch.append(patch)
    write_data(pan_patch,save_dir)
def image_together(image,corners,filepath,data):
    image_output = torch.zeros_like(image)
    print(image_output.shape)
    x_grid_mask = torch.zeros_like(image)
    for (hi, wi) in corners:        
        x_grid_mask[:, :, hi:hi + 128, wi:wi + 128] += weightMatrix
    for idx, (hi, wi) in enumerate(corners): 
            image_output[0, :, hi:hi + 128, wi:wi +128] = image_output[0, :, hi:hi + 128, wi:wi +128]+data.dataset[idx][0][0]*weightMatrix
    x_grid_mask[x_grid_mask<1/32] = 1
    image_output = torch.div(image_output, x_grid_mask)
    datatype = gdal.GDT_Int16
    _,band_nums, height, width = image_output.shape
    driver = gdal.GetDriverByName("GTiff")
    sample = th.div(image_output, 0.0001)
    filename = os.path.join(filepath, "image_together"+'.tif')
    print(filename)
    dataset = driver.Create(filename, width, height, band_nums, datatype)
    for num in range(band_nums):
              dataset.GetRasterBand(num + 1).WriteArray(sample[0][num].cpu().numpy())
    del dataset

def main():
    #Image to be cropped
    crop_image ="t2_1_WFV.tif"
    save_dir ="test"
    Crop_Image(crop_image,save_dir) 
    
    #images together
    sameshape_image="t2_1_WFV.tif"
    image_m=read_image(sameshape_image)
    corners=overlapping_grid_indices(image_m,128,r=32)
    save_dir1="test"
    data = load_data(
        data_dir=save_dir1,
        batch_size=1,
        class_cond=False
    )
    image_together(image_m,corners,save_dir1,data)
    
    
    
if __name__ == "__main__":
    main()   

  
    
    
    
    
    
    
    
    
    
    