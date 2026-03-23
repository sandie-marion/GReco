import torch
import random
from collections import Counter

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data.Purchase100 import Purchase100Dataset
from utility import flatten

from torch.utils.data import random_split


# Draw local distributions
################################################################################################
################################################################################################

# Draw distribution with a dirichlet
################################################################################################
def draw_distribution(alpha: float, n_workers: int, n_classes: int, n_instances: int) -> torch.Tensor:
    """
    Generates a distribution of class counts across workers using a Dirichlet distribution.

    Returns:
    torch.Tensor: A tensor containing the class counts for each worker, with shape (n_workers, n_classes).
                  Each entry represents the number of instances assigned to a specific class for each worker.
    """
    
    # Create a tensor of shape (n_workers, n_classes) with each value equal to 'alpha'
    heterogeneity_params = alpha * torch.ones(n_workers, n_classes)

    # Sample from the Dirichlet distribution to get the proportions of class instances per worker, each row is a probability vector multiply by n_instances
    class_counts_float = torch.distributions.dirichlet.Dirichlet(heterogeneity_params).sample() * n_instances

    # Floor the floating-point class counts to get integer values, and cast them to long integers
    class_counts = torch.floor(class_counts_float).long()

    return class_counts


def draw_class_wise_distribution (alpha: float, n_workers:int, n_classes:int, n_instances : list) -> torch.Tensor : 
    heterogeneity_params = alpha * torch.ones(n_classes, n_workers)

    dirichlet = torch.distributions.dirichlet.Dirichlet(heterogeneity_params).sample()
    trans_dirichlet = torch.transpose(dirichlet, 0, 1) 

        # Sample from the Dirichlet distribution to get the proportions of class instances per worker, each row is a probability vector multiply by n_instances
    class_counts_float = torch.mul(trans_dirichlet, torch.tensor(n_instances))

    # Floor the floating-point class counts to get integer values, and cast them to long integers
    class_counts = torch.floor(class_counts_float).long() 

    return class_counts


    
# Draw distributions of workers
################################################################################################
def worker_distributions(n_honest_workers: int, n_byzantine_workers: int, alpha: float, n_classes:int, dataset_name:str) -> torch.Tensor:
    """
    Generates data distribution for workers. 
    
    Returns:
    A tensor of the shape n_workers x n_classes with an entry idx_worker, idx_class 
    corresponding to the number of instances of idx_class in the local dataset idx_worker.
    """
    # Local dataset size
    if dataset_name == 'MNIST' or dataset_name == 'Fashion_MNIST':
        gobal_dataset_size = 60000
    elif dataset_name == 'CIFAR10':
        gobal_dataset_size = 50000
    elif dataset_name == 'Purchase100':
        gobal_dataset_size = 157859
    elif dataset_name == 'EMNIST' : 
        gobal_dataset_size = 697932
    elif dataset_name == 'EuroSAT' :
        gobal_dataset_size = 22950
        
    local_dataset_size = gobal_dataset_size // n_honest_workers
    
    # Draw distributions of classes for workers.
    distributions_honest_workers = draw_distribution(alpha, n_honest_workers, n_classes, local_dataset_size)
    
    if n_byzantine_workers>0:
        distributions_byzantine_workers = draw_distribution(alpha, n_byzantine_workers, n_classes, local_dataset_size)
    else:
        distributions_byzantine_workers = None
    
    return distributions_honest_workers, distributions_byzantine_workers


def heterogeneous_distributions (n_honest_workers: int, n_byzantine_workers: int, alpha: float, n_classes:int, dataset_name:str) : 

    train_set, _ = get_dataset(dataset_name) 
    int_targets = []
    for target in train_set.targets : 
        int_targets.append(int(target))
    sample_per_class = Counter(int_targets)

    sample_per_class_row = []
    for i in range (0, len(sample_per_class)) : 
        sample_per_class_row.append(sample_per_class[i])

    distributions_honest_workers = draw_class_wise_distribution(alpha, n_honest_workers, n_classes, sample_per_class_row)
    counts_per_client = torch.sum(distributions_honest_workers, 1).tolist() 
    while 0 in counts_per_client : 
        distributions_honest_workers = draw_class_wise_distribution(alpha, n_honest_workers, n_classes, sample_per_class_row)
        counts_per_client = torch.sum(distributions_honest_workers, 1).tolist() 
        print("pas de chance") 


    if n_byzantine_workers>0:
        distributions_byzantine_workers = draw_class_wise_distribution(alpha, n_byzantine_workers, n_classes, sample_per_class_row)
    else:
        distributions_byzantine_workers = None
    
    return distributions_honest_workers, distributions_byzantine_workers

# Compute dataloader
################################################################################################
################################################################################################

# Return default Pytorch train and test set
################################################################################################
def get_dataset(dataset_name: str) -> tuple:
    """
    Loads the specified dataset.

    Returns:
        A tuple containing the Pytorch training and test datasets.
    """
    if dataset_name == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2023, 0.1994, 0.2010))
        ])
    
        train_set = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
        test_set = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)

    elif dataset_name == "MNIST":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
    
        train_set = datasets.MNIST(root="./data", train=True, transform=transform_train, download=True)
        test_set = datasets.MNIST(root="./data", train=False, transform=transform_test, download=True)
    
    elif dataset_name == "Fashion_MNIST":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])
    
        train_set = datasets.FashionMNIST(root="./data", train=True, transform=transform_train, download=True)
        test_set = datasets.FashionMNIST(root="./data", train=False, transform=transform_test, download=True)
    
    elif dataset_name == "EMNIST" : 
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
    
        train_set = datasets.EMNIST(root="./data", split="byclass", train=True, transform=transform_train, download=True)
        test_set = datasets.EMNIST(root="./data", split="byclass", train=False, transform=transform_test, download=True)
        
    elif dataset_name == "EuroSAT" : 

        global_set = datasets.EuroSAT(root = "./data", download=True, transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2023, 0.1994, 0.2010))
                ]) )

        train_set, test_set = random_split(global_set, [0.85, 0.15])
        
        
    elif dataset_name == "Purchase100":
        train_set = Purchase100Dataset(train_bool=True)
        test_set = Purchase100Dataset(train_bool=False)
        
    else: 
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_set, test_set

# Compute DataLoader for each worker
################################################################################################
def draw_worker_loaders(distributions:torch.Tensor , train_set: torch.utils.data.Dataset, batch_size :int)->list:
    """
    Creates worker-specific data loaders by distributing data according to given class distributions.
    
    Args:
        distributions (torch.Tensor): A tensor of shape (num_workers, n_classes) representing worker-class distributions.
        train_set (Dataset): The original training dataset.
        batch_size (int): The batch size for each worker's DataLoader.
    
    Returns:
        list: A list of DataLoader objects, one per worker.
    """
    
    n_workers, n_classes = distributions.shape

    # tensor of size n_classes; each entry is the global count for that class
    count_of_classes = distributions.sum(0)

    # Get indices of samples belonging to each class
    if len(train_set) == 22950 : 
        print("dataset is EuroSAT") 
        targets = [train_set.dataset.targets[i] for i in train_set.indices]
    else : 
        targets = train_set.targets
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    """
    indices_of_classes is a list of list giving indices for each class, e.g., 
    [
      [0, 5, 7],      # samples where targets == 0
      [1, 3],         # samples where targets == 1
      [2, 4, 6, 8]    # samples where targets == 2
    ]
    """
    indices_of_classes = [(targets == class_idx).nonzero(as_tuple=True)[0].tolist() for class_idx in range(n_classes)] 

    # Create a dictionary to store resampled indices for each class to match count_of_classes
    new_indices = {class_idx:[] for class_idx in range(n_classes)}

    # Resample each class to match count_of_classes
    for class_idx in range(n_classes):
        class_indices = indices_of_classes[class_idx]
        
        original_n_class_idx = len(class_indices)

        # Determine the integer part (full replications) and remainder (fractional part to sample) of the factor
        factor =  count_of_classes[class_idx].item()/original_n_class_idx
        rep = int(factor)
        remainder = factor - rep

        # Fully replicate the class samples based on integer part of the factor
        new_indices[class_idx].extend(class_indices * rep)
        additional_count = int(original_n_class_idx * remainder) + 1
        
        # Sample additional instances based on the fractional part
        if additional_count > 0:
            sampled = random.sample(class_indices, additional_count)
            new_indices[class_idx].extend(sampled)

        # Shuffle in place the indices to ensure randomness
        random.shuffle(new_indices[class_idx])
    """
    new_indices is a dictionary where each key is a class label and each value is a list of sample indices for that class, arranged to match 
    count_of_classes with minimal repetition. 
    """

    # Assign data to each worker according to their class distributions
    worker_loaders=[]
    for worker_id in range(n_workers):
        worker_data = {class_idx:[] for class_idx in range(n_classes)}

        # Allocate class-specific samples to the worker
        for class_idx in range(n_classes):
            class_count = distributions[worker_id, class_idx].item()
            # Transfer the first 'class_count' indices from new_indices[class_idx] to worker_data[class_idx]
            worker_data[class_idx].extend(new_indices[class_idx][:class_count])
            new_indices[class_idx] = new_indices[class_idx][class_count:]
        
        # Flatten the worker's dataset into a single list of indices
        worker_data = flatten([worker_data[class_idx] for class_idx in range(n_classes)])
        
        # Shuffle the data to avoid class clustering within batches
        random.shuffle(worker_data)
        
        # Create a DataLoader for this worker
        local_dataset = Subset(train_set, worker_data) # builds the dataset; duplicates in worker_data repeat samples in local_dataset
        worker_loader = DataLoader(local_dataset, batch_size=batch_size, shuffle=False)
        worker_loaders.append(worker_loader)
        
    return worker_loaders

# Compute test loader
################################################################################################
def draw_test_set_loader(distributions: torch.Tensor, test_set: torch.utils.data.Dataset, batch_size: int) -> DataLoader:
    """
    Creates a test set loader by resampling the dataset based on class distribution factors.
    
    Args:
        distributions (torch.Tensor): A tensor of shape (num_samples, n_classes) representing class distributions.
        test_set (Dataset): The original test dataset.
        batch_size (int): The batch size for the DataLoader.
    
    Returns:
        DataLoader: A DataLoader for the resampled test set.
    """
    distributions = distributions.float()
    # Compute global class distribution
    totals = distributions.sum()
    count_of_classes = distributions.sum(0)
    global_distribution = count_of_classes / totals
    
    # Find the minimal proportion of the least represented class
    min_proportion = torch.min(global_distribution).item()
    
    """
    Compute resampling factors so that the least represented class has a factor of 1,
    and all other classes have factors >1 based on their relative proportion
    """
    factors = (global_distribution / min_proportion).tolist()

    # Get indices of samples belonging to each class
    targets = test_set.targets
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    indices_of_classes = [(targets == class_idx).nonzero(as_tuple=True)[0].tolist() for class_idx in range(distributions.shape[1])]
    """
    indices_of_classes is a list of list giving indices for each class, e.g., 
    [
      [0, 5, 7],      # samples where targets == 0
      [1, 3],         # samples where targets == 1
      [2, 4, 6, 8]    # samples where targets == 2
    ]
    """
    
    # Resample each class based on its computed factor
    new_indices = []
    for class_idx, class_indices in enumerate(indices_of_classes):
        factor = factors[class_idx]
        
        # Determine the integer part (full replications) and remainder (fractional part to sample) of the factor
        rep = int(factor)
        remainder = factor - rep
        
        # Fully replicate the class samples based on integer part of the factor
        new_indices.extend(class_indices * int(rep))
        
        # Sample additional instances based on the fractional part of the factor
        additional_count = int(len(class_indices) * remainder)
        if additional_count > 0:
            new_indices.extend(random.sample(class_indices, additional_count))
    
    # Create a new subset using the sampled indices
    new_test_set = Subset(test_set, new_indices)  # builds the dataset; duplicates in new_indices repeat samples in new_test_set
    
    # Return a DataLoader for the newly created test set
    return DataLoader(new_test_set, batch_size=batch_size, shuffle=False)

# Compute worker and test loaders
################################################################################################        
def load_data(distributions: torch.Tensor, batch_size: int, dataset_name: str, heterogeneous_distrib):
    """
    Get worker loaders and test loader.
    Args:
        distributions (torch.Tensor): A tensor of shape (num_samples, n_classes) representing class distributions or None
    """
    """
    distributions can be None in case distributions = distributions_byzantine_workers with n_byzantine_workers=0
    """
    # There are distribution
    if distributions is not None:
        train_set, test_set = get_dataset(dataset_name)
        worker_loaders = draw_worker_loaders(distributions, train_set, batch_size)
        test_loader = DataLoader(test_set) 
       
    # If distribution is None
    else:
        worker_loaders = []
        test_loader = None

    return worker_loaders, test_loader


