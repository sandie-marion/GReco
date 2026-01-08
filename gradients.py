import torch
import torch.nn as nn

def flatten_gradients(gradient:list) -> torch.Tensor:
    """
    Returns:
    A flattened gradient from the gradient list.
    """
    lst_flat_gradients = [param.view(-1) for param in gradient]
    return torch.cat(lst_flat_gradients, dim=0)


def model_parameters_format(row_gradient: torch.Tensor, model: nn.Module) -> list:
    """
    Reshape a flattened gradient into its original format corresponding to model.parameters().

    Returns:
    A list of gradients with the same format and shapes as model.parameters().
    """
    """
    Collect shapes of all parameters in the model. 
    A list of tuples, each containing the shape of a parameter in model.parameters()
    """
    shapes = []

    """
    Collect number of elements for each parameter to determine how to split row_gradient. 
    A list of integers, each representing the number of elements in a parameter
    """
    splits = []
    
    # Collect shapes and compute splits
    for param in model.parameters():
        shape = tuple(param.shape)  # Get the shape of the parameter as a tuple
        
        # Calculate the total number of elements for the parameter
        num_elements = 1
        for dimension in shape:
            num_elements *= dimension
        
        # Store shape and corresponding split size
        shapes.append(shape)
        splits.append(num_elements)
    
    # Initialize the list to store the reshaped gradients
    reshaped_gradient = []

    # Split the flattened gradient and reshape to the original parameter shapes
    offset = 0  # Track the position in row_gradient
    for split, shape in zip(splits, shapes):
        # Extract a segment of row_gradient and reshape it to the original parameter shape
        param = row_gradient[offset:offset + split].view(*shape)
        reshaped_gradient.append(param)
        
        # Update offset for the next segment
        offset += split
        
    return reshaped_gradient



def gradient_mean(row_gradients: list) -> torch.Tensor:
    """
    Computes the mean of the given gradients.
    row_gradients is a list of flatted gradients.
    
    Returns:
    Mean of gradients, a torch.Tensor.
    """
    # Stack gradients along new dimension
    stacked_row_gradients = torch.stack(row_gradients, dim=0)
    
    return stacked_row_gradients.mean(dim=0)


def gradient_std(row_gradients: list) -> torch.Tensor:
    """
    Computes the standard deviation of the given gradients.
    row_gradients is a list of flatted gradients.
    
    Returns:
    Standard deviation of gradients, a torch.Tensor.
    """
    # Stack gradients along new dimension
    stacked_row_gradients = torch.stack(row_gradients, dim=0)
    
    return stacked_row_gradients.std(dim=0)


def gradient_dissimilarity(row_gradients: list) -> torch.Tensor:
    """
    Computes the dissimilarity of gradients using the variance (the equivalence is shown below). 
    row_gradients is a list of flatted gradients.

    Returns:
    A float corresponding to gradient dissimilarity.
    """
    mean_gradient = gradient_mean(row_gradients)
    var = torch.mean(torch.stack([torch.norm(gradient - mean_gradient, p=2)**2 for gradient in row_gradients])).item()

    return var 