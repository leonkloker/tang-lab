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

class RelativeL2Loss(nn.Module):
    def __init__(self):
        super(RelativeL2Loss, self).__init__()

    def forward(self, prediction, ground_truth, avg_dim=[0]):
        dims = set([i for i in range(len(prediction.shape))])
        for dim in avg_dim:
            dims.remove(dim)
        dims = tuple(dims)
        
        l2_error = torch.linalg.vector_norm(prediction - ground_truth, ord=2, dim=dims)
        l2_norm = torch.linalg.vector_norm(ground_truth, ord=2, dim=dims)
        loss = torch.mean(l2_error / l2_norm)
        return loss

class Autoencoder(pl.LightningModule):
    def __init__(self, dims=[17, 8, 4]):
        super(Autoencoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.rel_mse = RelativeL2Loss()
        for i in range(1, len(dims)):
            self.encoder.append(nn.Linear(dims[i-1], dims[i]))
            self.decoder.append(nn.Linear(dims[-i], dims[-1-i]))

    def forward(self, x):
        for layer in self.encoder:
            x = torch.relu(layer(x))

        for layer in self.decoder[:-1]:
            x = torch.relu(layer(x))
        x = self.decoder[-1](x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_rec = self(x)
        loss = F.mse_loss(x, x_rec)
        rel_mse = self.rel_mse(x_rec, x)
        self.log('train_mse', loss, on_epoch=True)
        self.log('train_rel_mse', rel_mse, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_rec = self(x)
        mse = F.mse_loss(x, x_rec)
        rel_mse = self.rel_mse(x_rec, x)
        self.log('val_mse', mse, on_epoch=True)
        self.log('val_rel_mse', rel_mse, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=100)
        return [optimizer], [scheduler]

if __name__ == '__main__':
    torch.manual_seed(42)

    train_set = Single_cell_dataset('./data/25_populations_antiIge.pickle', max_combs=1000)
    means, stds = train_set.get_normalization()
    val_set = Single_cell_dataset('./data/11_populations_antiIge.pickle', max_combs=11, means=means, stds=stds)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=5, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True)

    dims = [17, 8, 4]
    model = Autoencoder.load_from_checkpoint('logs/AE_17_8_4/version_8/checkpoints/epoch=50-step=102000.ckpt')

    # loggers
    tb_logger = TensorBoardLogger('logs/', name='AE_{}'.format('_'.join(map(str, dims))))

    # Create a PyTorch Lightning trainer with desired configurations
    trainer = pl.Trainer(max_epochs=100, accelerator='mps', logger=tb_logger, log_every_n_steps=100)

    # Train the model
    trainer.fit(model, train_loader, val_loader)