import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import NetflixTripletDataset2, EvalDataset
from utils import collate_fn_eval


class RSMLDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=4096, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=collate_fn_eval, pin_memory=True)
