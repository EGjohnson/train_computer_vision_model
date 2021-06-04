#%%
# import from pytorch
import torch
from torch import nn
# to load data into the model 
from torch.utils.data import DataLoader # wraps datasets
from torchvision import datasets # reads in data
# to transform images
from torchvision.transforms import ToTensor, Lambda, Compose 

# %%
# LOAD DATASETs
training_data = datasets.FashionMNIST(root="data",
                                      train=True,
                                      download=True,
                                      transform=ToTensor())
# %%
test_data = datasets.FashionMNIST(root="data", 
                                  train=False,
                                  download=True,
                                  transform=ToTensor())


# PASS DATASET TO DATALOADER (BATCHES / LOADS)
train_dataloader = DataLoader(training_data,
                              batch_size)