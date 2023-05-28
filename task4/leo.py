import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(1707)
np.random.seed(1707)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA = 1e-5

def load_dataset():
    """
        Load the data from the zip files, return the torch tensors    
    """

    # Load the dataset
    X_pretrain_csv = pd.read_csv('data/pretrain_features.csv.zip', compression='zip', index_col="Id")
    y_pretrain_csv = pd.read_csv('data/pretrain_labels.csv.zip', compression='zip', index_col="Id")
    X_train_csv = pd.read_csv('data/train_features.csv.zip', compression='zip', index_col="Id")
    y_train_csv = pd.read_csv('data/train_labels.csv.zip', compression='zip', index_col="Id")
    X_test_csv = pd.read_csv('data/test_features.csv.zip', compression='zip', index_col="Id")
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
    def __init__(self, X, y, fc=None, test=False):
        super(Regressor, self).__init__()
        
        # Setup the model
        if fc is None:
            self.fc = nn.Sequential(
                nn.Linear(1000, 800),
                nn.Hardswish(),
                nn.Linear(800, 500),
                nn.Hardswish(),
                nn.Linear(500, 300),
                nn.Hardswish(),
                nn.Linear(300, 1)
            )
        else:
            self.fc = fc

        # If the test option has been set, divide the dataset into test and train
        self.has_test = bool(test)
        if self.has_test:
            self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=test, random_state=42)
            self.X_test = self.X_test.to(DEVICE)
        else:
            self.X, self.y = X, y

        # Move the model to GPU
        self.to(DEVICE)
        self.criterion = nn.MSELoss().to(DEVICE)
        self.optim = torch.optim.Adam(self.parameters(), lr = 1e-5)

        # Setup for later
        self.losses, self.errors = [], []
        self.best_epoch, self.best_error = False, False
    
    def forward(self, x):
        return self.fc(x)
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def plot(self, show=True, save=False):
        """
            Plot the losses
        """
        plt.clf()
        plt.plot(np.asarray(self.losses[int(len(self.losses) * 0.1):]), 'b')
        plt.plot(np.asarray(self.errors[int(len(self.errors) * 0.1):]), 'r')
        plt.title('Total train and test loss')
        if self.best_epoch and self.best_error:
            plt.legend([f"Best epoch: {self.best_epoch}", f"Best error: {self.best_error}"])
        plt.xlabel('Batches')
        plt.ylabel('Train Loss')

        if show:
            plt.show()
        if save:
            plt.savefig(save)

    def freeze_layers(self, count):
        """
            Freeeze the first count layers
        """
        for i, (_, param) in enumerate(self.named_parameters()):
            param.requires_grad = i >= count

    def evaluate(self, X, y):
        X = X.float().to(DEVICE)

        # Apply the regressor and computer the loss
        with torch.no_grad():
            predictions = self(X)
            return self.criterion(predictions, y.unsqueeze(1))    
        
    def _step(self, X, y, encoder = None):
        self.optim.zero_grad()

        # Apply Autoencoder
        if encoder is None:
            predictions = self(X)
            if self.has_test:
                y_test = self(self.X_test)
        else:
            predictions = encoder(X) - self(X)
            if self.has_test:
                y_test = encoder(self.X_test) - self(self.X_test)
        
        # Calculating the error
        error = mean_squared_error(self.y_test, y_test.cpu().detach().numpy(), squared=False) if self.has_test else 0

        # Calculating the loss
        norm = sum(param.pow(2.0).sum() for param in self.parameters())
        train_loss = self.criterion(predictions, y.unsqueeze(1)) + LAMBDA * norm

        # backward pass and gradient step
        train_loss.backward()
        self.optim.step()

        return train_loss, error


    def train_model(self, epochs=5, batch_size=0, encoder=None):
        """
        """
        X_device = self.X.to(DEVICE)
        y_device = self.y.to(DEVICE)

        if batch_size:
            pretrain_loader = DataLoader(torch.cat((self.X, self.y.unsqueeze(1)), dim=1), batch_size=batch_size, shuffle=True, num_workers=2)

            for _ in (pbar := tqdm(range(epochs))):
                for batch in tqdm(pretrain_loader, leave=False):
                    batch = batch.to(DEVICE)

                    X, y = batch[:, :1000], batch[:, 1000]
                    self._step(X, y, encoder)

                train_loss = self.evaluate(X_device, y_device)
                self.losses.append(train_loss.item())
                pbar.set_description(f'Total train loss: {train_loss:.4f}')

        else:
            self.best_error, self.best_epoch = 100, 0
            for epoch in (pbar := tqdm(range(epochs))): 
                train_loss, error = self._step(X_device, y_device, encoder)
                self.losses.append(train_loss.item())
                self.errors.append(error)

                pbar.set_description(f'Total train loss: {train_loss:.4f}, error {error:.4f}')

                # Save the one with the smallest error
                if self.has_test and self.best_error > error:
                    self.best_epoch = epoch
                    self.best_error = error
                    best = self.state_dict()
            
            if self.has_test:
                print("best_error:", self.best_error, ", best_epoch", self.best_epoch)
                self.load_state_dict(best)


plots_data = {}

def test_model(name, model):
    global plots_data

    X_pretrain, y_pretrain, X_train, y_train, X_test = load_dataset()
    filename = f"models/{name}.pt"

    lumo = Regressor(X_pretrain, y_pretrain, fc=copy.deepcopy(model), test=True)
    lumo.train_model(epochs=30, batch_size=128) # 5, 64
    lumo.plot(show=False, save=f"plots/{name}_lumo.png")
    lumo.save(filename)

    homo = Regressor(X_train, y_train, fc=copy.deepcopy(model), test=True)
    homo.load(filename)
    homo.freeze_layers(count=4)
    homo.train_model(epochs=20000, encoder=lumo)
    homo.plot(show=False, save=f"plots/{name}_homo.png")

    plots_data[name] = homo.errors
    plt.clf()
    for value in plots_data.values():
        plt.plot(value[int(len(value) * 0.1):])
    plt.legend(list(plots_data.keys()))
    plt.savefig("global.png")

    with torch.no_grad():
        solution = pd.read_csv('data/sample.csv', index_col="Id")

        X_L = lumo(X_test.to(DEVICE)).squeeze(1)
        X_H = homo(X_test.to(DEVICE)).squeeze(1)

        predictions = X_L - X_H

        solution.y = predictions[:].to('cpu')
        solution.to_csv(f'results/{name}.csv')
    

if __name__ == "__main__":
    # from models import models
    # for key, value in models.items():
    #     print("Starting:", key)
    #     test_model(key, value)

    X_pretrain, y_pretrain, X_train, y_train, X_test = load_dataset()
    filename = "models/model29.pt"
    
    lumo = Regressor(X_pretrain, y_pretrain, test=0.3)
    # lumo.train_model(epochs=50, batch_size=256) # 5, 64
    # lumo.plot()
    # lumo.save(filename)
    lumo.load(filename)

    # Create the homo regressor, with parameters starting from the lumo and freeze the first 4 layers
    homo = Regressor(X_train, y_train)
    homo.load(filename)
    homo.freeze_layers(count=4)
    homo.train_model(epochs=15110, encoder=lumo)
    homo.plot()

    # Calculate the solutions with the 2 models and save it
    with torch.no_grad():
        solution = pd.read_csv('data/sample.csv', index_col="Id")

        X_L = lumo(X_test.to(DEVICE)).squeeze(1)
        X_H = homo(X_test.to(DEVICE)).squeeze(1)

        predictions = X_L - X_H

        solution.y = predictions[:].to('cpu')
        solution.to_csv('solution_29.csv')