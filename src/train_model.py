from torch import optim, nn, Tensor
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import prune

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelPruning
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ray.tune.integration.pytorch_lightning import TuneReportCallback


from src import feature_engineering


# load and define dataset and dataloader
X, y = feature_engineering.load_images()

training_indices: Tensor = torch.randperm(len(X))[: int(len(X) * 0.8)]


train_dataset = TensorDataset(X[training_indices], y[training_indices])
test_dataset = TensorDataset(X[~training_indices], y[~training_indices])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])

# define the model to be used
class Model(nn.Module):
    """A convolutional neural network for brain tumor detection.

    The model architecture consists of:
        - First convolutional layer (1->5 channels, 5x5 kernel, stride 5) with max pooling
        - Second convolutional layer (5->10 channels, 5x5 kernel, stride 5) with max pooling
        - Two fully connected layers (40->20->10->1) with sigmoid activation

    The network takes grayscale brain scan images as input and outputs a binary
    classification probability indicating the presence of a tumor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=(5, 5), stride=(5, 5))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(5, 5), stride=(5, 5))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(40, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.conv1(x)
        x: Tensor = self.pool1(x)
        x: Tensor = self.conv2(x)
        x: Tensor = self.pool2(x)
        x: Tensor = torch.flatten(x, 1)
        x: Tensor = F.relu(self.fc1(x))
        x: Tensor = F.dropout(x, p=0.1)
        x: Tensor = F.relu(self.fc2(x))
        x: Tensor = F.dropout(x, p=0.1)
        x: Tensor = F.sigmoid(self.fc3(x))
        return x


# define the lightning module
class LightningModel(L.LightningModule):
    """A PyTorch Lightning module wrapping the brain tumor detection CNN model.

    This module handles the training process including forward pass, loss calculation,
    optimization, and metric logging. It uses binary cross entropy loss for training
    and tracks both loss and accuracy metrics.

    The underlying model architecture consists of:
        - Two convolutional layers with max pooling
        - A final fully connected layer with sigmoid activation
        - Binary classification output
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = Model()

    def training_step(self, batch, batch_idx) -> Tensor:
        """Executes a single training step.

        Performs forward pass through model, calculates binary cross entropy loss,
        computes accuracy, and logs metrics.

        Args:
            batch: A tuple of (images, labels) where images are brain scan tensors
                  and labels are binary tumor indicators
            batch_idx: Index of current batch (unused)

        Returns:
            Tensor: The calculated loss value for backpropagation
        """
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy(y_hat.squeeze(), y)
        accuracy = ((y_hat.squeeze() > 0.5) == y).float().mean()

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        """Executes a single validation step.

        Performs forward pass through model, calculates binary cross entropy loss,
        computes accuracy, and logs validation metrics.

        Args:
            batch: A tuple of (images, labels) where images are brain scan tensors
                  and labels are binary tumor indicators
            batch_idx: Index of current batch (unused)

        Returns:
            Tensor: The calculated validation loss value
        """
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy(y_hat.squeeze(), y)
        accuracy = ((y_hat.squeeze() > 0.5) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)


model = LightningModel()
logger = TensorBoardLogger("logs", name="Brain_Tumor_Detection")


def pruning_schedule(epoch):
    if epoch == 150:
        return 0.2
    elif epoch == 200:
        return 0.1
    elif epoch == 250:
        return 0.1
    else:
        return 0.0
    

# define the trainer
trainer = L.Trainer(
    max_epochs=700,
    accelerator="gpu",
    devices=1,
    logger=logger,
    log_every_n_steps=100,
    callbacks=[
        ModelPruning(pruning_fn='l1_unstructured', amount=pruning_schedule)],
)

# train the model
trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=test_loader,
)
