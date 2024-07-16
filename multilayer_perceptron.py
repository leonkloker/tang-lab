import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import sys
import numpy as np

from data import Single_cell_dataset

class SimpleFeedForward(pl.LightningModule):
    def __init__(self, dims=[17, 64, 64, 1]):
        super(SimpleFeedForward, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i-1], dims[i]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.layers[-1](x))
        #x = torch.round(x)
        x = torch.mean(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[0]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('train_mse', loss, on_epoch=True)
        self.log('train_mae', mae, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[0]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('val_mse', loss, on_epoch=True)
        self.log('val_mae', mae, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-0)
        scheduler = OneCycleLR(optimizer, max_lr=1e-0, steps_per_epoch=len(train_loader), epochs=100)
        return [optimizer], [scheduler]

if __name__ == '__main__':
    torch.manual_seed(42)

    train_set = Single_cell_dataset('./data/25_populations_antiIge.pickle', max_combs=1)
    means, stds = train_set.get_normalization()
    val_set = Single_cell_dataset('./data/25_populations_antiIge.pickle', max_combs=1, means=means, stds=stds)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True)

    dims = [17, 64, 64, 64, 1]
    model = SimpleFeedForward(dims=dims)

    # loggers
    tb_logger = TensorBoardLogger('logs/', name='FNN_{}'.format('_'.join(map(str, dims))))

    # Create a PyTorch Lightning trainer with desired configurations
    trainer = pl.Trainer(max_epochs=100, accelerator='mps', logger=tb_logger, log_every_n_steps=1)

    # Train the model
    trainer.fit(model, train_loader, val_loader)
