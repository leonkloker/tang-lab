import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

import data

class PopulationLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data = 
        self.batch_size = batch_size
        self.train_data, self.val_data = train_test_split(data, test_size=0.2)

    def train_dataloader(self):
        return DataLoader(TensorDataset(*self.train_data), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(TensorDataset(*self.val_data), batch_size=self.batch_size)

class SimpleFeedForward(pl.LightningModule):
    def __init__(self):
        super(SimpleFeedForward, self).__init__()
        self.layer_1 = nn.Linear(10, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1).float())
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1).float())
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=10)
        return [optimizer], [scheduler]

# Create a PyTorch Lightning model
model = SimpleFeedForward()

# loggers
tb_logger = TensorBoardLogger('logs/', name='my_model')

# Create a PyTorch Lightning trainer with desired configurations
trainer = pl.Trainer(max_epochs=10, accelerator='mps', logger=tb_logger)

# Train the model
trainer.fit(model, train_loader, val_loader)