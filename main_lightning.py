import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils import set_seed, preprocessing
from dataset import NetflixTripletDataset2, EvalDataset
from lit_model import RSMLModule
from lit_datamodule import RSMLDataModule
from collections import defaultdict
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')

    config = parser.parse_args()
    pl.seed_everything(42)
    DATA_PATH = "./dataset"
    #### hyper parameters #####
    HIDDEN_SIZE = config.hidden_size
    BATCH_SIZE = config.batch_size
    LR = config.lr
    ###########################
    TEST_RATIO = 0.2
    K = 10
    EPOCHS = 100

    interactions, user_map, item_map, user_interactions = preprocessing(DATA_PATH)
    num_users = len(user_map)
    num_items = len(item_map)

    random.shuffle(interactions)
    split = int(len(interactions) * (1 - TEST_RATIO))
    train = interactions[:split]
    val = interactions[split:]

    train_map = defaultdict(set)
    for u, i in train:
        train_map[u].add(i)
    val_map = defaultdict(set)
    for u, i in val:
        val_map[u].add(i)

    train_dataset = NetflixTripletDataset2(train, num_items, train_map)
    val_dataset = EvalDataset(val_map)

    model = RSMLModule(num_users, num_items, hidden_size=HIDDEN_SIZE, lr=LR, k=K)
    datamodule = RSMLDataModule(train_dataset, val_dataset, batch_size=BATCH_SIZE)

    tensorboard_logger = TensorBoardLogger("tb_logs", name="rsml_model")
    callbacks = [
        ModelCheckpoint(monitor="val_hitrate", mode="max", save_top_k=1, filename="best-{epoch}-{val_hitrate:.4f}"),
        EarlyStopping(monitor="val_hitrate", mode="max", patience=10)
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=EPOCHS,
        logger=tensorboard_logger,
        callbacks=callbacks,
        log_every_n_steps=10
    )

    trainer.fit(model, datamodule=datamodule)