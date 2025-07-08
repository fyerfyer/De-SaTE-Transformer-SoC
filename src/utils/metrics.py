from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np

def get_rul_error(y_test, y_predict, capacity_threshold):
    """
    Calculates the relative error in Remaining Useful Life (RUL) prediction.
    The RUL is the number of cycles before the capacity drops below a threshold.

    Args:
        y_test (np.array): The true capacity values.
        y_predict (np.array): The predicted capacity values.
        capacity_threshold (float): The failure capacity threshold.

    Returns:
        float: The relative error in RUL.
    """
    # Find the true RUL
    true_rul_index = np.argmax(y_test < capacity_threshold) if np.any(y_test < capacity_threshold) else len(y_test) - 1
    
    # Find the predicted RUL
    pred_rul_index = np.argmax(y_predict < capacity_threshold) if np.any(y_predict < capacity_threshold) else len(y_predict) - 1
    
    error = abs(true_rul_index - pred_rul_index)
    
    return error

def get_rmse(y_test, y_predict):
    """
    Calculates the Root Mean Squared Error.
    """
    return sqrt(mean_squared_error(y_test, y_predict))

def get_mae(y_test, y_predict):
    """
    Calculates the Mean Absolute Error.
    """
    return mean_absolute_error(y_test, y_predict)

def get_relative_error(y_test, y_predict):
    """
    Calculates the Mean Relative Error (RE) as used in the reference implementation.
    
    Args:
        y_test (np.array): The true values.
        y_predict (np.array): The predicted values.
    
    Returns:
        float: The mean relative error.
    """
    # Avoid division by zero
    mask = y_test != 0
    if np.sum(mask) == 0:
        return 0.0
    
    relative_errors = np.abs((y_test[mask] - y_predict[mask]) / y_test[mask])
    return np.mean(relative_errors)

def get_e_value(y_test, y_predict):
    """
    Calculate the E value as defined in the project requirements.
    
    E = mean(|measured_capacity - true_capacity| / true_capacity) × 100%
    
    Args:
        y_test (np.array): The true capacity values.
        y_predict (np.array): The predicted capacity values.
    
    Returns:
        float: The E value as a percentage.
    """
    # Avoid division by zero
    mask = y_test != 0
    if np.sum(mask) == 0:
        return 100.0  # Return high error if no valid data
    
    # Calculate relative absolute errors
    relative_errors = np.abs((y_test[mask] - y_predict[mask]) / y_test[mask])
    e_value = np.mean(relative_errors) * 100.0
    
    return e_value

def get_weighted_e_value(y_test, y_predict, weight_strategy='soh_range'):
    """
    Calculate the weighted E value to reduce the impact of low SOH values.
    
    Args:
        y_test (np.array): The true capacity values.
        y_predict (np.array): The predicted capacity values.
        weight_strategy (str): Weighting strategy ('soh_range' or 'inverse_soh')
    
    Returns:
        float: The weighted E value as a percentage.
    """
    # Avoid division by zero
    mask = y_test != 0
    if np.sum(mask) == 0:
        return 100.0  # Return high error if no valid data
    
    y_test_masked = y_test[mask]
    y_predict_masked = y_predict[mask]
    
    # Calculate relative absolute errors
    relative_errors = np.abs((y_test_masked - y_predict_masked) / y_test_masked)
    
    if weight_strategy == 'soh_range':
        # Weight by SOH ranges: High weight for practical range (0.6-0.8), 
        # medium weight for mid range (0.3-0.6), low weight for low range (<0.3)
        weights = np.ones_like(y_test_masked)
        
        # High weight for practical range (0.6-0.8) - most important for battery management
        practical_mask = (y_test_masked >= 0.6) & (y_test_masked <= 0.8)
        weights[practical_mask] = 10.0
        
        # Medium weight for mid range (0.3-0.6) - still operationally relevant
        mid_mask = (y_test_masked >= 0.3) & (y_test_masked < 0.6)
        weights[mid_mask] = 2.0
        
        # Low weight for low range (<0.3) - end of life, less critical for accuracy
        low_mask = y_test_masked < 0.3
        weights[low_mask] = 0.5
        
        # Very low weight for extremely low SOH (<0.1)
        very_low_mask = y_test_masked < 0.1
        weights[very_low_mask] = 0.1
        
    elif weight_strategy == 'inverse_soh':
        # Weight inversely proportional to SOH (higher SOH gets more weight)
        weights = y_test_masked / np.mean(y_test_masked)
        # Clip weights to prevent extreme values
        weights = np.clip(weights, 0.1, 5.0)
    
    else:
        raise ValueError(f"Unknown weight strategy: {weight_strategy}")
    
    # Calculate weighted average
    weighted_e_value = np.average(relative_errors, weights=weights) * 100.0
    
    return weighted_e_value

def get_practical_range_e_value(y_test, y_predict, soh_min=0.6, soh_max=0.8):
    """
    Calculate the E value focusing on the practical SOH range.
    
    Args:
        y_test (np.array): The true capacity values.
        y_predict (np.array): The predicted capacity values.
        soh_min (float): Minimum SOH for practical range (default: 0.6)
        soh_max (float): Maximum SOH for practical range (default: 0.8)
    
    Returns:
        dict: Dictionary containing E value for practical range and data statistics
    """
    # Find practical range mask
    practical_mask = (y_test >= soh_min) & (y_test <= soh_max) & (y_test != 0)
    
    if np.sum(practical_mask) == 0:
        return {
            'e_value': 100.0,
            'data_points': 0,
            'coverage': 0.0,
            'soh_range': f"{soh_min}-{soh_max}",
            'warning': 'No data points in practical range'
        }
    
    # Calculate E value for practical range only
    y_test_practical = y_test[practical_mask]
    y_predict_practical = y_predict[practical_mask]
    
    relative_errors = np.abs((y_test_practical - y_predict_practical) / y_test_practical)
    e_value_practical = np.mean(relative_errors) * 100.0
    
    return {
        'e_value': e_value_practical,
        'data_points': np.sum(practical_mask),
        'coverage': np.sum(practical_mask) / len(y_test) * 100.0,
        'soh_range': f"{soh_min}-{soh_max}",
        'mean_soh': np.mean(y_test_practical),
        'std_soh': np.std(y_test_practical)
    }

def calculate_e_score(e_value):
    """
    Calculate the score based on E value according to project requirements.
    
    Scoring criteria:
    - If E ≤ 10%: Score = (10 - E) × 1.5 (maximum 15 points)
    - If E > 10%: Score = 0 points
    
    Args:
        e_value (float): The E value as a percentage.
    
    Returns:
        float: The score based on E value.
    """
    if e_value <= 10.0:
        return (10.0 - e_value) * 1.5
    else:
        return 0.0

def calculate_all_metrics(y_true, y_pred, include_weighted=True):
    """
    Calculate all standard metrics used in the reference implementation.
    
    Args:
        y_true (np.array): Ground truth values
        y_pred (np.array): Predicted values
        include_weighted (bool): Whether to include weighted metrics
    
    Returns:
        dict: Dictionary containing RE, RMSE, MAE, E_value, E_score, and weighted metrics
    """
    e_value = get_e_value(y_true, y_pred)
    
    metrics = {
        'RE': get_relative_error(y_true, y_pred),
        'RMSE': get_rmse(y_true, y_pred),
        'MAE': get_mae(y_true, y_pred),
        'E_value': e_value,
        'E_score': calculate_e_score(e_value)
    }
    
    if include_weighted:
        # Add weighted E-value metrics
        weighted_e_value = get_weighted_e_value(y_true, y_pred, 'soh_range')
        metrics['E_value_weighted'] = weighted_e_value
        metrics['E_score_weighted'] = calculate_e_score(weighted_e_value)
        
        # Add practical range metrics
        practical_metrics = get_practical_range_e_value(y_true, y_pred)
        metrics['E_value_practical'] = practical_metrics['e_value']
        metrics['E_score_practical'] = calculate_e_score(practical_metrics['e_value'])
        metrics['practical_range_info'] = practical_metrics
    
    return metrics
