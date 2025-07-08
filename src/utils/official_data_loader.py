import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging

logger = logging.getLogger(__name__)

def create_feature_sequences(features_array, soh_values, window_size):
    """
    Create sequences from feature arrays for SOH prediction.
    
    Args:
        features_array: Array of shape (n_samples, n_features)
        soh_values: Array of SOH values corresponding to each sample
        window_size: Size of the temporal window
        
    Returns:
        x_sequences: Feature sequences of shape (n_sequences, window_size, n_features)
        y_sequences: Target SOH values of shape (n_sequences,)
    """
    if len(features_array) < window_size:
        return np.array([]), np.array([])
    
    x_sequences = []
    y_sequences = []
    
    for i in range(len(features_array) - window_size + 1):
        # 使用滑动窗口创建特征序列
        x_seq = features_array[i:i + window_size]
        # 目标是窗口末尾的SOH值
        y_val = soh_values[i + window_size - 1]
        
        x_sequences.append(x_seq)
        y_sequences.append(y_val)
    
    return np.array(x_sequences), np.array(y_sequences)

def prepare_official_dataset(processed_data_path, window_size=16, test_split=0.2):
    """
    Prepare the official dataset for training and testing.
    
    Args:
        processed_data_path: Path to the processed .npy file
        window_size: Temporal window size for sequences
        test_split: Fraction of data to use for testing
        
    Returns:
        Tuple of (train_loader, test_loader, feature_info)
    """
    try:
        # 加载处理后的数据
        data_dict = np.load(processed_data_path, allow_pickle=True).item()
        logger.info(f"Loaded dataset with {len(data_dict)} batteries")
    except FileNotFoundError:
        logger.error(f"Processed data file not found at {processed_data_path}")
        return None, None, None
    
    # 分离训练和测试数据
    train_batteries = []
    test_batteries = []
    
    for key, battery_data in data_dict.items():
        if battery_data.get('is_training', False):
            train_batteries.append((key, battery_data))
        else:
            test_batteries.append((key, battery_data))
    
    logger.info(f"Found {len(train_batteries)} training batteries and {len(test_batteries)} testing batteries")
    
    if len(train_batteries) == 0:
        logger.error("No training data found")
        return None, None, None
    
    # 收集特征信息
    sample_battery = list(data_dict.values())[0]
    feature_names = [k for k in sample_battery.keys() 
                    if k not in ['battery_id', 'file_path', 'is_training', 'target_soh']]
    
    logger.info(f"Using {len(feature_names)} features: {feature_names}")
    
    # 准备训练数据 - 使用训练电池的真实SOH值
    train_x_list = []
    train_y_list = []
    
    for battery_id, battery_data in train_batteries:
        # 从特征中提取数值特征
        features = []
        for feature_name in feature_names:
            if feature_name in battery_data:
                features.append(battery_data[feature_name])
            else:
                features.append(0.0)  # 缺失值用0填充
        
        features_array = np.array(features).reshape(1, -1)  # (1, n_features)
        
        # 对于训练数据，我们有真实的SOH值
        if 'target_soh' in battery_data:
            soh_value = battery_data['target_soh']
            # 由于我们只有一个时间点的数据，创建虚拟的时间序列
            # 这里简化处理：复制特征来创建时间序列
            extended_features = np.repeat(features_array, window_size, axis=0)
            soh_values = np.full(window_size, soh_value)
            
            x_seq, y_seq = create_feature_sequences(extended_features, soh_values, window_size)
            if len(x_seq) > 0:
                train_x_list.append(x_seq)
                train_y_list.append(y_seq)
    
    if len(train_x_list) == 0:
        logger.error("No valid training sequences created")
        return None, None, None
    
    # 合并所有训练数据
    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    
    logger.info(f"Created {len(train_x)} training sequences with shape {train_x.shape}")
    
    # 准备测试数据 - 对于测试电池，我们需要预测SOH
    test_x_list = []
    test_battery_ids = []
    
    for battery_id, battery_data in test_batteries:
        features = []
        for feature_name in feature_names:
            if feature_name in battery_data:
                features.append(battery_data[feature_name])
            else:
                features.append(0.0)
        
        features_array = np.array(features).reshape(1, -1)
        extended_features = np.repeat(features_array, window_size, axis=0)
        
        test_x_list.append(extended_features)
        test_battery_ids.append(battery_id)
    
    # 创建数据加载器
    train_x_tensor = torch.FloatTensor(train_x)
    train_y_tensor = torch.FloatTensor(train_y)
    
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    batch_size = min(32, len(train_dataset))  # 确保batch size不大于数据集大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # 测试数据加载器（如果有测试数据）
    test_loader = None
    if test_x_list:
        # 为测试创建虚拟标签（实际预测时不使用）
        test_x = np.stack(test_x_list, axis=0)
        test_y = np.zeros(len(test_x))  # 虚拟标签
        
        test_x_tensor = torch.FloatTensor(test_x)
        test_y_tensor = torch.FloatTensor(test_y)
        
        test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
        test_batch_size = min(32, len(test_dataset))
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'window_size': window_size,
        'test_battery_ids': test_battery_ids
    }
    
    return train_loader, test_loader, feature_info

def get_official_data_loaders(processed_data_path, window_size=16, batch_size=32):
    """
    Get data loaders for the official competition dataset.
    
    Args:
        processed_data_path: Path to processed dataset
        window_size: Temporal window size
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, test_loader, feature_info)
    """
    return prepare_official_dataset(processed_data_path, window_size)