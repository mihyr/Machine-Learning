from numpy.lib.function_base import flip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None,
    batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
    """
    This function either trains or evaluates a model.

    training mode: the model is trained and evaluated on a validation set, if provided.
                   If no validation set is provided, the training is performed for a fixed
                   number of epochs.
                   Otherwise, the model should be evaluted on the validation set
                   at the end of each epoch and the training should be stopped based on one
                   of these two conditions (whichever happens first):
                   1. The validation loss stops improving.
                   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs:

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
    learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model
    loss: dictionary with keys 'train' and 'valid'
          The value of each key is a list of loss values. Each loss value is the average
          of training/validation loss over one epoch.
          If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
         The value of each key is a list of accuracies (percentage of correctly classified
         samples in the dataset). Each accuracy value is the average of training/validation
         accuracies over one epoch.
         If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set.
    accuracy: percentage of correctly classified samples in the testing set.

    Summary of the operations this function should perform:
    1. Use the DataLoader class to generate trainin, validation, or test data loaders
    2. In the training mode:
       - define an optimizer (we use SGD in this homework)
       - call the train function (see below) for a number of epochs untill a stopping
         criterion is met
       - call the test function (see below) with the validation data loader at each epoch
         if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results

    """

    if running_mode == 'train':
        epochs = 0
        past_loss = 1000000
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []
        optimize = optim.SGD(model.parameters(), lr=learning_rate)

        for i in range(n_epochs):
            #if epochs< n_epochs:
            #  return model, loss_dict, acc_dict

            load_set = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = shuffle)
            model, train_loss, train_accuracy = _train(model, load_set, optimize)
            loss_train.append(train_loss)
            acc_train.append(train_accuracy)
            
            if valid_set != None:
                load_vset = torch.utils.data.DataLoader(valid_set, batch_size = batch_size, shuffle = shuffle)
                present_valid_loss, present_valid_accuracy = _test(model, load_vset)
                loss_valid.append(present_valid_loss)
                acc_valid.append(present_valid_accuracy)

                valid_loss_diff = past_loss - present_valid_loss

                if valid_loss_diff>=stop_thr:
                    past_loss = present_valid_loss
                else:
                    return model, {"train": loss_train, "valid": loss_valid}, {"train": acc_train, "valid": acc_valid}
                epochs +=1
            else:
                epochs +=1
        return model, {"train": loss_train, "valid": loss_valid}, {"train": acc_train, "valid": acc_valid}

    elif running_mode == 'test':
        load_set = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = shuffle)
        test_loss, test_accuracy = _test(model, load_set)
        return test_loss, test_accuracy


def _train(model,data_loader,optimizer,device=torch.device('cpu')):

    """
    This function implements ONE EPOCH of training a neural network on a given dataset.
    Example: training the Digit_Classifier on the MNIST dataset
    Use nn.CrossEntropyLoss() for the loss function


    Inputs:
    model: the neural network to be trained
    data_loader: for loading the netowrk input and targets from the training dataset
    optimizer: the optimiztion method, e.g., SGD
    device: we run everything on CPU in this homework

    Outputs:
    model: the trained model
    train_loss: average loss value on the entire training dataset
    train_accuracy: average accuracy on the entire training dataset
    """
    
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    asserted_true = 0
    count = 0
    present_loss = 0
    for index, elem in enumerate(data_loader):

        ip, labels = elem
        ip = ip.float()
        optimizer.zero_grad()
        op = model(ip)

        loss = loss_fn(op, labels)
        loss.backward()
        loss_assert  = loss.item()
        #print(loss_assert)
        present_loss += loss_assert

        optimizer.step() 

        x, predict = torch.max(op.data, 1)
        #x, predict = torch.exp(op.data, 1)
        #increment = labels.size(0)
        increment = np.size(labels,0)
        count += increment
        pt_list = predict == labels
        #print(pt_list.sum())
        pt_bool = sum(pt_list)
        assert_increment = pt_bool.item()*100
        asserted_true += assert_increment

        

    return model, present_loss/count, asserted_true/count
    


def _test(model, data_loader, device=torch.device('cpu')):
    """
    This function evaluates a trained neural network on a validation set
    or a testing set.
    Use nn.CrossEntropyLoss() for the loss function

    Inputs:
    model: trained neural network
    data_loader: for loading the netowrk input and targets from the validation or testing dataset
    device: we run everything on CPU in this homework

    Output:
    test_loss: average loss value on the entire validation or testing dataset
    test_accuracy: percentage of correctly classified samples in the validation or testing dataset
    """

    present_loss = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    asserted_true = 0
    count = 0

    #with torch.no_grad():
    for index, elem in enumerate(data_loader):
        ip, labels = elem
        ip = ip.float()
        op = model(ip)

        loss = loss_fn(op, labels)
        #loss.backward()
        loss_assert  = loss.item()
        #print(loss_assert)
        present_loss += loss_assert

        
        x, predict = torch.max(op.data, 1)
        increment = np.size(labels,0)
        count += increment
        pt_list = predict == labels
        #print(pt_list.sum())
        pt_bool = sum(pt_list)
        
        assert_increment = pt_bool.item()*100 
        asserted_true += assert_increment


    return present_loss/count, asserted_true/count 

