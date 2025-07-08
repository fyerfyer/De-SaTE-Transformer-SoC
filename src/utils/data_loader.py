import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def build_sequences(text, window_size):
    """
    Builds sequences and targets from a time series.
    Adapted from the notebook.
    """
    x, y = [], []
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]
        x.append(sequence)
        y.append(target)
    return np.array(x), np.array(y)

def get_train_test(data_dict, test_battery_name, window_size=16):
    """
    Creates train and test sets using a proper leave-one-out strategy.
    NO data from the test battery is used for training to prevent data leakage.
    
    Args:
        data_dict (dict): The dictionary loaded from the .npy file.
        test_battery_name (str): The name of the battery to be used for testing.
        window_size (int): The sequence window size.
    
    Returns:
        A tuple containing train_x, train_y, test_x, test_y as numpy arrays,
        and the original test sequence for plotting.
    """
    
    test_data_sequence = data_dict[test_battery_name]['capacity']
    
    # Initialize empty training arrays - NO test battery data used for training
    train_x = np.empty((0, window_size))
    train_y = np.empty((0, window_size))

    # Add ALL other batteries to the training set (exclude test battery completely)
    for k, v in data_dict.items():
        if k != test_battery_name:
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
            if len(train_x) == 0:
                train_x, train_y = data_x, data_y
            else:
                train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
    
    # Use the ENTIRE test battery sequence for testing (no data leakage)
    test_x, test_y = build_sequences(text=test_data_sequence, window_size=window_size)

    return train_x, train_y, test_x, test_y, test_data_sequence

def get_data_loaders(processed_data_path, test_battery_name, window_size=16, batch_size=128):
    """
    The main function to get data loaders for training and testing.
    
    Args:
        processed_data_path (str): Path to the processed .npy file.
        test_battery_name (str): The name of the battery to use for the test set.
        window_size (int): The sequence length.
        batch_size (int): The batch size for the data loaders.

    Returns:
        A tuple of (train_loader, test_loader, and the raw test sequence for plotting)
    """
    
    try:
        data_dict = np.load(processed_data_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        return None, None, None
        
    if test_battery_name not in data_dict:
        print(f"Error: Test battery '{test_battery_name}' not found in the dataset.")
        print(f"Available batteries: {list(data_dict.keys())}")
        return None, None, None

    train_x, train_y, test_x, test_y, test_sequence = get_train_test(
        data_dict, test_battery_name, window_size
    )

    if len(test_x) == 0:
        print(f"Warning: Not enough data for test battery '{test_battery_name}' to create a test set with window size {window_size}.")
        print(f"Total cycles available for {test_battery_name}: {len(test_sequence)}. Cycles needed to create at least one test sample: {2 * window_size + 2}.")
        print("Please choose a different test battery or a smaller window size.")
        return None, None, None

    # Reshape and convert to tensors
    train_x = train_x.reshape(-1, window_size, 1)
    train_y = train_y.reshape(-1, window_size, 1)
    test_x = test_x.reshape(-1, window_size, 1)
    test_y = test_y.reshape(-1, window_size, 1)
    
    # The label is the last value of the target sequence
    train_y_label = torch.from_numpy(train_y[:,-1,:]).float()
    test_y_label = torch.from_numpy(test_y[:,-1,:]).float()

    train_dataset = TensorDataset(torch.from_numpy(train_x).float(), train_y_label)
    test_dataset = TensorDataset(torch.from_numpy(test_x).float(), test_y_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, test_sequence
