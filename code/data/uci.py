import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import os

uci_datasets = [
    'elevators', 
    'bike', 
    'keggdirected', 
    'keggundirected',
    'ink40k', 
    'protein',
    '3droad'
]


class UCI(Dataset):
    def __init__(self, root, data_split, transform = None, target_transform = None):
        """
        Args:
            root(str): Directory with data.
            data_split (str): which split of the data to use (options: `train`, `test`, `val`).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if data_split not in ["train", "test", "val"]:
            raise ValueError(f"data_split must be one of `train`, `test`, or `val`. Input given {data_split}.")
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.x = torch.from_numpy(pd.read_csv(os.path.join(root, data_split, 'x.csv')).to_numpy())
        self.y = torch.from_numpy(pd.read_csv(os.path.join(root, data_split, 'y.csv')).to_numpy())
        self.x = self.x.to(dtype=torch.float)
        self.y = self.y.to(dtype=torch.float)

    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.x)

def standardize_x(x_train, x_val, x_test):
    """Standardize columns of x. 
    Mean zero and variance one the inputs using the training 
    moments. 
    Args:
        x_train (torch.Tensor): N_train matrix of training inputs. 
        x_val (torch.Tensor): N_val matrix of training inputs. 
        x_test (torch.Tensor): N_test matrix of testing inputs. 
    Returns:
        x_train, x_val, x_test, x_mu, x_std
    """
    x_mu = x_train.mean(0,keepdims = True)
    x_std = x_train.std(0,keepdims = True)
    x_std[x_std == 0] = 1.0
    x_train = (x_train - x_mu).div(x_std)
    x_val = (x_val - x_mu).div(x_std)
    x_test = (x_test - x_mu).div(x_std)
    return x_train, x_val, x_test, x_mu, x_std
    

def standardize_y(y_train, y_val, y_test):
    """Standardize y. 
    Mean zero and variance one the targets using the training 
    moments. 
    Args:
        y_train (torch.Tensor): N_train matrix of training targets. 
        y_val (torch.Tensor): N_val matrix of training targets. 
        y_test (torch.Tensor): N_test matrix of testing targets. 
    Returns:
        y_train, y_val, y_test, y_mu, y_std
    """
    y_mu = y_train.mean()
    y_std = y_train.std()

    y_train = (y_train - y_mu).div(y_std)
    y_val = (y_val - y_mu).div(y_std)
    y_test = (y_test - y_mu).div(y_std)

    y_val = y_val.contiguous().squeeze()
    y_train = y_train.contiguous().squeeze()
    y_test = y_test.contiguous().squeeze()


    return y_train, y_val, y_test, y_mu, y_std


def inputs_to_cube(x_train, x_test, default_value = 0.0):
    """cube x onto the unit cube. 
    Put the inputs onto the unit hypercube. If NANs occur
    then replace them with default value. 
    Args:
        x_train (torch.Tensor): N_train x d matrix of training inputs. 
        x_test (torch.Tensor): N_test x d matrix of testing inputs. 
        default_value (float, default = 0.0): Value to replace NANs with.
    Returns:
        x_train, x_test
    """
    x = torch.cat((x_train, x_test), 0)
    x_max = x.max(0)[0].unsqueeze(0)
    x_min = x.min(0)[0].unsqueeze(0)
    max_minus_min = x_train.max(0)[0] - x_train.min(0)[0]
    bad_ind = (max_minus_min == 0.0)
    x_train = torch.div(x_train-x_min, x_max- x_min).contiguous()
    x_test = torch.div(x_test-x_min, x_max- x_min).contiguous()
    x_train[:, bad_ind] = default_value
    x_test[:, bad_ind] = default_value
    return x_train, x_test

def standard_uci_load(data_folder):
    """load a standard uci dataset. 
    Use a standard procedure to load a UCI dataset. 
    Args:
        data_folder (str): folder of the data as expected by UCI loader. 
        tkwargs (dict): the type and device of the data. 
    Returns:
        x_train, x_test, x_val, y_train, y_test, y_val, y_mu, y_std
    """
    # train data.
    data = UCI(data_folder, data_split = "train")
    x_train, y_train = data[:]
    # test data.
    data = UCI(data_folder, data_split = "test")
    x_test, y_test = data[:]
    # validation data.
    data = UCI(data_folder, data_split = "val")
    x_val, y_val = data[:]

    y_train, y_val, y_test, y_mu, y_std = standardize_y(y_train, y_val, y_test)
    x_train, x_val, x_test, x_mu, x_std = standardize_x(x_train, x_val, x_test)

    return x_train, x_test, x_val, y_train, y_test, y_val, y_mu, y_std

def neural_net_uci_load(data_folder, tkwargs = None):
    """load a standard uci dataset for use with neural net. 
    Use a standard procedure to load a UCI dataset,
    but make the inputs mean zero variance one. 
    Args:
        data_folder (str): folder of the data as expected by UCI loader. 
    Returns:
        x_train, x_test, x_val, y_train, y_test, y_val, y_mu, y_std
    """
    # get data from standard load.
    x_train, x_test, x_val, y_train, y_test, y_val, y_mu, y_std = \
    standard_uci_load(data_folder)

    # unsqueeze last dimension of y for nn. 
    y_test = y_test.unsqueeze(-1)
    y_train = y_train.unsqueeze(-1)
    y_val = y_val.unsqueeze(-1)

    if tkwargs is not None:
        x_train = x_train.to(**tkwargs)
        x_test = x_test.to(**tkwargs)
        x_val = x_val.to(**tkwargs)

        y_train = y_train.to(**tkwargs)
        y_test = y_test.to(**tkwargs)
        y_val = y_val.to(**tkwargs)

    return x_train, x_test, x_val, y_train, y_test, y_val, y_mu, y_std

def uci_dataloaders(data_folder, batch_size):
    train_x, test_x, val_x, train_y, test_y, val_y, y_mu, y_std = \
    neural_net_uci_load(data_folder)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    train_data_shape = train_x.shape

    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader, y_mu, y_std

def many_inputs_to_cube(x_list, default_value = 0.0):
    """cube x onto the unit cube. 
    Put the inputs onto the unit hypercube. If NANs occur
    then replace them with default value. 
    Args:
        x_train (torch.Tensor): N_train x d matrix of training inputs. 
        x_test (torch.Tensor): N_test x d matrix of testing inputs. 
        default_value (float, default = 0.0): Value to replace NANs with.
    Returns:
        x_train, x_test
    """
    X = torch.cat(x_list, 0)
    x_max = X.max(0)[0].unsqueeze(0)
    x_min = X.min(0)[0].unsqueeze(0)
    max_minus_min = X.max(0)[0] - X.min(0)[0]
    del X
    bad_ind = (max_minus_min == 0.0)

    new_x = []
    for x in x_list:
        x_ = torch.div(x-x_min, x_max-x_min).contiguous()
        x_[:, bad_ind] = default_value
        new_x.append(x_)

    return new_x