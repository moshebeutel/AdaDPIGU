from models.CNN import CIFAR10_CNN_Tanh, CIFAR10_CNN_Relu, MNIST_CNN_Relu, MNIST_CNN_Tanh
import torch

def get_model(algorithm, dataset_name, device):
    """
    Retrieve the appropriate model based on the algorithm and dataset name.
    
    Args:
        algorithm (str): The algorithm type (e.g., DPSGD).
        dataset_name (str): The dataset name (e.g., MNIST, CIFAR-10, FMNIST).
        device (str): The device to load the model onto (e.g., cpu or cuda).

    Returns:
        model (torch.nn.Module): The initialized model.
    """
    if dataset_name in ['MNIST', 'FMNIST']:
        if algorithm == 'DPSGD':
            model = MNIST_CNN_Relu(1)
        else:
            model = MNIST_CNN_Tanh(1)
    elif dataset_name == 'CIFAR10':
        if algorithm == 'DPSGD':
            model = CIFAR10_CNN_Relu(3)
        else:
            model = CIFAR10_CNN_Tanh(3)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Move the model to the specified device
    model.to(device=device)
    return model
