from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning import LightningDataModule
from typing import List

class RandomSampleDataset(Dataset):
    def __init__(self):
        self.name = 'Randomsample Dataset'
    def __len__(self):
        return 1000
    def __getitem__(self,index):
        x = torch.rand(1,16000)
        y = torch.rand(1,1,16000)
        paras = {'samplerate':16000}
        return x,y,paras

class RandomSampleDataModule(LightningDataModule):
    def __init__(
            self,
            name: str,   # name of this data module
            batch_size: List[int] = [1, 1],
            num_workers: int = 0, # num workers of dataloader
            pin_memory: bool = True,
            persistent_workers:bool = True
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
                persistent_workers=self.persistent_workers
            )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
                self.evalset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers
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