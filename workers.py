import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gradients import flatten_gradients
from losses import Loss
from copy import deepcopy

# Class for a worker
################################################################################################
class Worker:
    def __init__(self, 
                 worker_id: int, 
                 honest: bool, 
                 criterion_name: str, 
                 criterion_parameters:dict,
                 train_loader: DataLoader,
                 model: nn.Module) -> None:
        """
        Initialize a Worker instance.
        """
        # Worker characteristics
        self.id = worker_id
        self.honest = honest
        self.byzantine = not honest
        
        # Gradient and momentum initialization
        self.gradient = [torch.zeros_like(param) for param in model.parameters()]
        self.momentum = [torch.zeros_like(param) for param in model.parameters()]
        
        # Criterion setup
        self.criterion = Loss(criterion_name, worker_id, **criterion_parameters)
                
        # Local dataset
        self.train_loader = train_loader

        # self.gradient_per_class = {l : [torch.zeros_like(param) for param in model.parameters()] for l in self.batch_provider.labels}
        # self.momentum_per_class = {l : [torch.zeros_like(param) for param in model.parameters()] for l in self.batch_provider.labels}
        # self.used_per_class = {l : 0 for l in self.batch_provider.labels}

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.
        """
        return self.criterion(outputs, labels)

    def compute_momentum(self, model: nn.Module, beta: float) -> None:
        """
        Compute momentum with beta the momentum factor.
        """
        # Update gradients and compute momentum
        with torch.no_grad():
            for param_idx, param in enumerate(model.parameters()):
                if param.grad is None:
                    continue

                # Get gradient (of param) from local dataset
                grad = param.grad.clone().detach() 
                
                # Store gradient
                self.gradient[param_idx] = grad
                
                # Update momentum: m = beta * m + (1 - beta) * g
                self.gradient[param_idx] = grad
                self.momentum[param_idx] = beta * self.momentum[param_idx] + (1 - beta) * grad
                    
        # Clear gradients for the next worker
        model.zero_grad()
    

    def save_class_momentum (self, c: int) -> None:
        self.gradient_per_class[c] = torch.tensor(self.gradient)
        self.momentum_per_class[c] = deepcopy(self.momentum)


    def flatten_momentum(self) -> torch.Tensor:
        """
        Flatten the momentum list into a Pytorch vector.
        """
        return flatten_gradients(self.momentum)


    def flatten_gradient(self) -> torch.Tensor:
        """
        Flatten the momentum list into a Pytorch vector.
        """
        return flatten_gradients(self.gradient)
        
# Class to manage workers
################################################################################################
class Workers:
    """
    Manages a collection of workers including both honest and byzantine workers.
    """
    def __init__(self, 
                 n_honest_workers: int, 
                 n_byzantine_workers: int, 
                 worker_loaders: list, 
                 criterion_name: str, 
                 criterion_parameters: dict, 
                 model: nn.Module) -> None:
        # Total number of workers
        self.n_workers = n_honest_workers + n_byzantine_workers
        
        # Number of honest and byzantine workers
        self.n_honest_workers = n_honest_workers
        self.n_byzantine_workers = n_byzantine_workers

        # Initialize all workers and store them in self.workers
        self.workers = []
        for worker_id in range(self.n_workers):    
            # Extra worker-specific variables
            is_honest = self.is_honest(worker_id) # True is honest, False otherwise
            local_loader = worker_loaders[worker_id]
            
            # Create a Worker instance and add it to the list
            worker = Worker(worker_id, is_honest, criterion_name, criterion_parameters, local_loader, model)
            self.workers.append(worker)
        
    def is_honest(self, worker_id: int) -> bool:
        """
        Attribute if a worker is honest or not based on its index.
        Honest workers are indexed from 0 to (n_honest_workers - 1).
        """
        return worker_id < self.n_honest_workers
    
    def get_momentums(self, only_honest: bool = False, row: bool = False) -> list:
        """
        Get the momentum tensors from the workers, just the honest ones or everyone, flattened or not.        
        """
        # Filter workers based on the 'only_honest' flag
        workers_to_consider = self.workers if not only_honest else [worker for worker in self.workers if worker.honest]
        
        # Return flattened row tensors or original momentum tensors
        if row:
            return [worker.flatten_momentum() for worker in workers_to_consider]
        else:
            return [worker.momentum for worker in workers_to_consider]
    
    def get_selected_momentums(self, worker_idxs, row = False) -> list:
        """
        Get the momentum tensors from the workers, just the honest ones or everyone, flattened or not.        
        """
        # Filter workers based on the 'only_honest' flag
        workers_to_consider = [self.workers[idx] for idx in worker_idxs]
        
        # Return flattened row tensors or original momentum tensors
        if row:
            return [worker.flatten_momentum() for worker in workers_to_consider]
        else:
            return [worker.momentum for worker in workers_to_consider]
                
    def loaders(self, seed: int = 42) -> list:
        """
        Returns the DataLoaders for all workers.
        """
        return [worker.train_loader for worker in self.workers]

    def class_batch_providers (self) -> list: 

        return [worker.batch_provider for worker in self.workers]
        
    def __getitem__(self, idx: int) -> 'Worker':
        """
        Access to idx worker.
        """
        return self.workers[idx]