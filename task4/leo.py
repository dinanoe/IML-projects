import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tqdm import tqdm
from torch.utils.data import DataLoader

torch.manual_seed(2022)
np.random.seed(2022)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset():
    """
        Load the data from the zip files, return the torch tensors    
    """
    # Load the dataset
    X_pretrain_csv = pd.read_csv('pretrain_features.csv.zip', compression='zip', index_col="Id")
    y_pretrain_csv = pd.read_csv('pretrain_labels.csv.zip', compression='zip', index_col="Id")
    X_train_csv = pd.read_csv('train_features.csv.zip', compression='zip', index_col="Id")
    y_train_csv = pd.read_csv('train_labels.csv.zip', compression='zip', index_col="Id")
    X_test_csv = pd.read_csv('test_features.csv.zip', compression='zip', index_col="Id")
    # sample = pd.read_csv('/sample.csv')

    X_pretrain = X_pretrain_csv.drop(columns="smiles").to_numpy()
    y_pretrain = y_pretrain_csv["lumo_energy"].to_numpy()
    X_train = X_train_csv.drop(columns="smiles").to_numpy()
    y_train = y_train_csv["homo_lumo_gap"].to_numpy()
    X_test = X_test_csv.drop(columns="smiles").to_numpy()

    X_pretrain = torch.from_numpy(X_pretrain).float()
    y_pretrain = torch.from_numpy(y_pretrain).float()
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()

    return X_pretrain, y_pretrain, X_train, y_train, X_test


class Regressor(nn.Module):
    """
        Define the regressor model
    """
    def __init__(self, X, y):
        super(Regressor, self).__init__()
        
        # Setup the model
        self.fc = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.Hardswish(),
            nn.Linear(1000, 1000),
            nn.Hardswish(),
            nn.Linear(1000, 1000),
            nn.Hardswish(),
            nn.Linear(1000, 1)
        )

        # Save the datasets
        self.X = X
        self.y = y

        # Move the model to GPU
        self.to(DEVICE)
        self.criterion = nn.MSELoss().to(DEVICE)
        self.optim = torch.optim.Adam(self.parameters(), lr = 1e-5)

        # Setup for later
        self.train_losses = []
        self.test_losses = []
    
    def forward(self, x):
        return self.fc(x)
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def plot(self):
        """
            Plot the losses
        """
        plt.plot(np.asarray(self.train_losses), 'b')
        plt.title('Total train and test loss')
        plt.xlabel('Batches')
        plt.ylabel('Train Loss')
        plt.show()

    def freeze_layers(self, count):
        """
            Freeeze the first count layers
        """
        for i, (_, param) in enumerate(self.named_parameters()):
            param.requires_grad = i >= count

    def evaluate(self, X, y):
        with torch.no_grad():
            # If avaiable move data to GPU
            X = X.float().to(DEVICE)

            # Apply the regressor and calculating the total loss
            predictions = self(X)
            return self.criterion(predictions, y.unsqueeze(1))    
        
    def _step(self, X, y):
        pass

    def train_model(self, epochs=5, batch_size=0, encoder=None, l2_lambda = 1e-5):
        """
        """
        X_device = self.X.to(DEVICE)
        y_device = self.y.to(DEVICE)

        if encoder is None:
            encoder = self

        if batch_size:
            pretrain_loader = DataLoader(torch.cat((self.X.to("cpu"), self.y.unsqueeze(1).to('cpu')), dim=1), batch_size=batch_size, shuffle=True, num_workers=2)

            for epoch in range(epochs):
                for batch in tqdm(pretrain_loader, desc='Train Epoch {}'.format(epoch)):
                    batch = batch.to(DEVICE)

                    X, y = batch[:, :1000], batch[:, 1000]

                    self.optim.zero_grad()

                    # Apply Autoencoder
                    predictions = encoder(X)

                    # Calculating the loss
                    l2_norm = sum(param.pow(2.0).sum() for param in self.parameters())
                    train_loss = self.criterion(predictions, y.unsqueeze(1)) + l2_lambda * l2_norm

                    # backward pass and gradient step
                    train_loss.backward()
                    self.optim.step()

                train_loss = self.evaluate(X_device, y_device)
                self.train_losses.append(train_loss.item())
        else:
            for epoch in tqdm(range(epochs)):
                self.optim.zero_grad()

                # Apply Autoencoder
                predictions = encoder(X_device) - self(X_device)
        
                # Calculating the loss
                l2_norm = sum(param.pow(2.0).sum() for param in self.parameters())
                train_loss = self.criterion(predictions, y_device.unsqueeze(1)) + l2_lambda * l2_norm

                # backward pass and gradient step
                train_loss.backward()
                self.optim.step()

                self.train_losses.append(train_loss.item())

                if (epoch+1) % 500 == 0:
                    print(f'Total train loss: {train_loss:.4f}')

if __name__ == "__main__":
    X_pretrain, y_pretrain, X_train, y_train, X_test = load_dataset()

    #
    lumo = Regressor(X_pretrain, y_pretrain)
    # lumo.train_model(epochs=5, batch_size=64) # 5, 64
    # lumo.plot()
    # lumo.save("lumo.pt")
    lumo.load("Regressor_LUMO.pt")

    # Create the homo regressor, with parameters starting from the lumo and freeze the first 4 layers
    homo = Regressor(X_train, y_train)
    homo.load("Regressor_LUMO.pt")
    homo.freeze_layers(count=4)

    homo.train_model(epochs=20000, encoder=lumo)
    homo.plot()

    # Calculate the solutions with the 2 models and save it
    with torch.no_grad():
        solution = pd.read_csv('sample.csv')

        X_L = lumo(X_test.to(DEVICE)).squeeze(1)
        X_H = homo(X_test.to(DEVICE)).squeeze(1)

        predictions = X_L - X_H

        solution.iloc[:,1] = predictions[:].to('cpu')

        solution.to_csv('solution.csv')