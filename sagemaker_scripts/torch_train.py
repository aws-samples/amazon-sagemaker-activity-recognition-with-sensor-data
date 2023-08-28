
""" SageMaker Training Script """

# Sagemaker imports
import argparse
import os

# other imports
import logging
from numpy import (
    arange,
    argmax,
    count_nonzero,
    dstack,
    zeros
)
from pandas import read_csv
from torch import (
    cuda as torch_cuda,
    device as torch_device,
    no_grad as torch_no_grad,
    Tensor
)
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from torch.jit import (
    save as torch_jit_save,
    script as torch_jit_script
)
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMClassifier

# setup logging
logging.basicConfig(level=logging.INFO)

def load_file(filepath):
    """
    Loads a single file as a numpy array

    Args:
        filepath (string): Local path to file

    Returns:
        numpy array: Array containing data from file
    """
    print('Reading from file: ', filepath)
    df = read_csv(filepath, header=None, delim_whitespace=True)
    return df.values

def load_file_group(filenames, prefix=''):
    """
    Loads a list of files and return as a 3d numpy array

    Args:
        filenames (list of strings): List of filenames containing input data
        prefix (string): Optional prefix for local filepath

    Returns:
        numpy array: 3d numpy array containing data from files
    """
    data_group = []
    for filename in filenames:
        data = load_file(prefix + filename)
        data_group.append(data)

    return dstack(data_group)

def load_dataset_type(dataset_type, prefix=''):
    """
    Loads a dataset type, such as train or test

    Args:
        dataset_type (string): Group name such as 'train' or 'test'
        prefix (string): Optional prefix for local filepath

    Returns:
        numpy array: Features
        numpy array: Labels
    """
    data_folder_path = prefix + dataset_type + '/Inertial Signals/'
    filenames = []
    for signal_type in ['total_acc', 'body_acc', 'body_gyro']:
        for axis in ['x', 'y', 'z']:
            filenames.append(f'{signal_type}_{axis}_{dataset_type}.txt') 

    X = load_file_group(filenames = filenames, prefix = data_folder_path)
    y = load_file(prefix + dataset_type + '/y_'+dataset_type+'.txt')
    return X, y

def one_hot_encode(inputs):
    """
    One hot encodes a Numpy array

    Args:
        inputs (numpy array): 1d numpy array containing class indices

    Returns:
        numpy array: 2d numpy array with one hot encoded class indices
    """
    outputs = zeros((inputs.size, inputs.max() + 1))
    outputs[arange(inputs.size), inputs] = 1
    return outputs

def load_dataset(prefix=''):
    """
    Loads the dataset

    Args:
        prefix (string): Optional prefix for local filepath

    Returns:
        numpy array: Features for train dataset
        numpy array: Labels for train dataset
        numpy array: Features for test dataset
        numpy array: Labels for test dataset
    """
    # load train dataset
    train_X, train_y = load_dataset_type(dataset_type = 'train', prefix = prefix + '/')
    # load test dataset
    test_X, test_y = load_dataset_type(dataset_type = 'test', prefix = prefix + '/')

    # offset class values to start at zero
    train_y = train_y - 1
    test_y = test_y - 1

    # one-hot encode labels
    train_y = train_y.reshape(len(train_y)) # array of single-label arrays -> array of labels
    test_y = test_y.reshape(len(test_y)) 
    train_y = one_hot_encode(train_y) # one hot encode train labels for neural network
    return train_X, train_y, test_X, test_y

def train(model, train_X, train_y, test_X, test_y, num_epochs = 20, batch_size = 64, learning_rate = 0.01):
    """
    Trains the model on the given dataset and evaluates model after each epoch

    Args:
        model (PyTorch NN module): Model to be trained
        train_X (numpy array): Features for train dataset
        train_y (numpy array): Labels for train dataset
        test_X (numpy array): Features for test dataset
        test_y (numpy array): Labels for test dataset
        num_epochs (int): Optional, specifies number of epochs to train for, default=20
        batch_size (int): Optional, specifies batch size for training, default=64
        learning_rate (float): Optional, specifies initial learning rate for training, default=0.01

    Returns:
        PyTorch NN module: Trained model
    """
    device = torch_device('cuda' if torch_cuda.is_available() else 'cpu')
    model.to(device) # pass model to GPU if present

    train_X_tensor = Tensor(train_X)
    train_y_tensor = Tensor(train_y)
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor) 
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size)

    cat_loss = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for i in range(num_epochs):
        model.train()
        for batch, labels in train_dataloader:
            batch, labels = batch.to(device), labels.to(device) # pass data to GPU if present
            optimizer.zero_grad()
            outputs = model(batch)
            loss = cat_loss(outputs, labels)
            loss.backward()
            optimizer.step()
  
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step() # update the learning rate

        # evaluate after each epoch
        model.eval()
        with torch_no_grad():
            test_X_tensor = Tensor(test_X)
            test_X_tensor = test_X_tensor.to(device)
            test_outputs = model(test_X_tensor)
            predictions = argmax(test_outputs.cpu().detach().numpy(), axis=-1)
            accuracy = count_nonzero(predictions == test_y) / len(test_y)

        logging.info(f'Epoch {i + 1} of {num_epochs}, Loss: {loss.item()}, Test Accuracy: {accuracy}, LR: {curr_lr}')

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA')) # usually this is split into train and test
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    # load data
    train_X, train_y, test_X, test_y = load_dataset(prefix=args.data)
    n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]

    # fit model
    model = LSTMClassifier(n_features, 100, n_outputs)
    trained_model = train(model, train_X, train_y, test_X, test_y, args.epochs, args.batch_size, args.learning_rate)

    # pickle and save the model
    model_path = os.path.join('/opt/ml/model', 'model.pt')
    trained_model = torch_jit_script(trained_model)
    torch_jit_save(trained_model, model_path)
