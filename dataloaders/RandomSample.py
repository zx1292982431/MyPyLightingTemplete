import sys
sys.path.append('/home/lizixuan/workspace/projects/MylightingTemplete')
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning import LightningDataModule
from typing import *
from dataloaders.utils.collate_func import default_collate_func
from dataloaders.utils.my_distributed_sampler import MyDistributedSampler
import random

class RandomSampleDataset(Dataset):
    def __init__(self):
        self.name = 'Randomsample Dataset'
    def __len__(self):
        return 200
    def __getitem__(self,index):
        x = torch.rand(1,16000)
        y = torch.rand(1,1,16000)
        paras = {
            'samplerate':16000,
            'index':index
        }
        return x,y,paras

class RandomSampleDataModule(LightningDataModule):
    def __init__(
            self,
            name: str,   # name of this data module
            batch_size: List[int] = [1, 1],
            num_workers: int = 0, # num workers of dataloader
            pin_memory: bool = True,
            persistent_workers:bool = True,
            seeds: Tuple[Optional[int], int, int] = [None, 2, 3],
    ):
        super().__init__()
        self.name = name
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size[0]
        self.batch_size_val = batch_size[1]
        self.batch_size_test = 1
        if len(batch_size) > 2:
            self.batch_size_test = batch_size[2]
        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))
    
    def setup(self, stage=None):
        self.trainset = RandomSampleDataset()
        self.evalset = RandomSampleDataset()
        self.testset = RandomSampleDataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=default_collate_func,
                sampler=MyDistributedSampler(self.trainset,seed=self.seeds[0],shuffle=True)
            )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
                self.evalset,
                batch_size=self.batch_size_val,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=default_collate_func,
                sampler=MyDistributedSampler(self.trainset,seed=self.seeds[1],shuffle=False)
            )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
                self.testset,
                batch_size=self.batch_size_test,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=default_collate_func,
                sampler=MyDistributedSampler(self.testset,seed=self.seeds[2],shuffle=False)
            )
    
if __name__ == '__main__':
    data_module = RandomSampleDataModule(
        name='test' ,
        num_workers=4
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        x,y,paras = batch
        print(x.shape)