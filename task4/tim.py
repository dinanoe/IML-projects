# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import torch.optim as optim
import torch.nn.functional as F
import tqdm
from sklearn.svm import SVR

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data
        # and then used to extract features from the training and test data.

        self.linear1 = nn.Linear(1000, 500)
        self.linear2 = nn.Linear(500, 250)
        self.linear3 = nn.Linear(250, 10)
        self.linear4 = nn.Linear(10, 1)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture
        # defined in the constructor.

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x
    
def make_feature_extractor(x, y, batch_size=20, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net()
    model.train()
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.

    train_dataset = TensorDataset(x_tr, y_tr)

    training_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    n_epochs = 1
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # I Chose the Adam optimizer
    loss_function = nn.MSELoss()  # Mean-squared error loss
    train_batch_losses = []
    val_batch_losses = []
    train_epoch_losses = []
    val_epoch_losses = []

    for epochs in range(n_epochs):
        for data, label in tqdm.tqdm(training_loader,colour='green', desc='Train Epoch {}'.format(epochs), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            #print(data.shape)
            model.train()
            optimizer.zero_grad()
            output = model(data)
            label = label.float()
            #print(output.shape)
            loss = loss_function(output, label.unsqueeze(1))
            train_batch_losses.append(loss)
            loss.backward()
            optimizer.step()

        train_epoch_loss = sum(train_batch_losses) / len(train_batch_losses)
        train_epoch_losses.append(train_epoch_loss.detach().numpy())
        print('Training loss {}'.format(train_epoch_loss))

        with torch.no_grad():
            for data, label in tqdm.tqdm(validation_loader, desc='Validation Epoch {}'.format(epochs), colour = 'blue', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                model.eval()
                output = model(data)
                label = label.float()
                loss = loss_function(output, label.unsqueeze(1))
                val_batch_losses.append(loss)


        val_epoch_loss = sum(val_batch_losses)/len(val_batch_losses)
        val_epoch_losses.append(val_epoch_loss)
        print('Validation loss {}'.format(val_epoch_loss))

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.

        number_of_features = 10
        length = len(x)

        # initializing the embedding matrix
        features_mat = np.zeros((length, number_of_features))

        model.linear4 = torch.nn.Identity()

        x_train = torch.tensor(x, dtype=torch.float)

        train_dataset = TensorDataset(x_train)

        training_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=1)

        # encoder loop that saves the batch embeddings in the embedding matrix
        with torch.no_grad():
            print('start check')
            for i, features in enumerate(training_loader):
                # print(features)
                batch_embedding = model(features[0])[0]
                # print(batch_idx)
                batch_embedding_np = batch_embedding.detach().numpy()
                # print(batch_embedding_np[0])

                for j in range(len(batch_embedding)):
                    features_mat[i][j] = batch_embedding_np[j]

        return features_mat

    return make_features, train_epoch_losses, val_epoch_losses, n_epochs

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
    pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))
    pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))

    pipelines = [x[1] for x in pipelines]

    class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, models):
            self.models = models

        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            self.models_ = [clone(x) for x in self.models]

            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)

            return self

        # Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1)

    return AveragingModels(models=(pipelines))

def plot_loss(train_epoch_losses, val_epoch_losses, n_epochs):

    x = np.linspace(0,n_epochs, n_epochs)
    train_loss = train_epoch_losses
    val_loss = val_epoch_losses

    plt.plot(x, train_loss, color='r', label='train_loss')
    plt.plot(x, val_loss, color='g', label='val_loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation loss")

    plt.legend()

    plt.show()

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor, train_epoch_loss, val_epoch_loss, epochs = make_feature_extractor(x_pretrain, y_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    plot_loss(train_epoch_loss, val_epoch_loss, epochs)
    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    pipe = Pipeline([('Scaler', StandardScaler()),("pretrain", PretrainedFeatureClass(feature_extractor='pretrain')), ('regression', regression_model)])

    pipe.fit(x_train, y_train)
    x_test_values = x_test.iloc[:,:].values
    y_pred = pipe.predict(x_test_values)


    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")