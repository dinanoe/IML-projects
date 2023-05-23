# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.preprocessing import normalize
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """

    # Automatic transformation via pre_trained weights of the model
    resnet18(weights=ResNet18_Weights.DEFAULT)
    weights = ResNet18_Weights.DEFAULT
    train_transforms = transforms.Compose([transforms.ToTensor(), weights.transforms()])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=False,
                              pin_memory=True, num_workers=4)

    # I chose the model resnet18
    model = torchvision.models.resnet18(pretrained=True)
    # setting the model to eval mode
    model.eval()

    batch_size = 16
    samples = 10000
    features = 512

    # initializing the embedding matrix
    embeddings_mat = np.zeros((samples, features))

    # set last layer "FC" to the identity layer so the last layer is "avgpool" that gives the 512 features vector
    model.fc = torch.nn.Identity()

    # encoder loop that saves the batch embeddings in the embedding matrix
    counter = 0
    with torch.no_grad():
        print('start check')
        for batch_idx, (features, _) in enumerate(train_loader):
            batch_embedding = model(features)
            batch_embedding_np = batch_embedding.detach().numpy()
            for i in range(batch_size):
                embeddings_mat[batch_size * batch_idx + i] = batch_embedding_np[i]

            print(counter)

            counter += 1

            # if counter == 10:
            # break

    print(embeddings_mat.shape)
    # returns embeddings with 512 features in a vector

    np.save('dataset/embeddings.npy', embeddings_mat)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('\\')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')

    embeddings = normalize(embeddings, axis=1, norm='l2')

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)

    if train == True:

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=1)

        #print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

        return X_train, X_val, y_train, y_val   # Dimensions: X_train(117839, 1536), y_train(117839,),
                                                # X_val(1191, 1536), y_val(1191,)
    else:
        return X,y


def create_loader_from_np(X, X_val=None, y=None, y_val=None, train=True, batch_size=16, shuffle=True, num_workers=4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float),
                                torch.from_numpy(y).type(torch.long))

        val_dataset = TensorDataset(torch.from_numpy(X_val).type(torch.float),
                                torch.from_numpy(y_val).type(torch.long))

        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True, num_workers=num_workers)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                pin_memory=True, num_workers=num_workers)

        return loader, val_loader

    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))

        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True, num_workers=num_workers)

        return loader


class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """

        # the model has to have an input of 1536 in the first layer and output of 1 in the last layer

        # Simple model with 3 linear layers with relu layers
        super().__init__()
        self.fc1 = nn.Linear(1536, 768)
        self.fc2 = nn.Linear(768, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def train_model(train_loader, val_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.to(device)
    n_epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # I Chose the Adam optimizer
    loss_function = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
    train_batch_losses = []
    val_batch_losses = []
    train_epoch_losses = []
    val_epoch_losses = []


    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.
    for epochs in range(n_epochs):
        for data, label in tqdm.tqdm(train_loader,colour='green', desc='Train Epoch {}'.format(epochs), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
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
            for data, label in tqdm.tqdm(val_loader, desc='Validation Epoch {}'.format(epochs), colour = 'blue', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                model.eval()
                output = model(data)
                label = label.float()
                loss = loss_function(output, label.unsqueeze(1))
                val_batch_losses.append(loss)


        val_epoch_loss = sum(val_batch_losses)/len(val_batch_losses)
        val_epoch_losses.append(val_epoch_loss)
        print('Validation loss {}'.format(val_epoch_loss))

    return model, train_epoch_losses, val_epoch_losses, n_epochs

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

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if os.path.exists('dataset/embeddings.npy') == False:
        generate_embeddings()

    # load the training and testing data
    X_train, X_val, y_train, y_val = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    #print(X_test.shape)

    # Create data loaders for the training and testing data
    train_loader, val_loader = create_loader_from_np(X_train, X_val, y_train, y_val, train = True, batch_size=32)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # define a model and train it
    model, train_epoch_losses, val_epoch_losses, n_epochs = train_model(train_loader, val_loader)

    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")

    # plot losses
    plot_loss(train_epoch_losses, val_epoch_losses, n_epochs)
