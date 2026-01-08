import torch
import torch.nn as nn
from torch import Tensor
from defenses import rfa

# Class Loss
################################################################################################
################################################################################################
class Loss:
    def __init__(self, loss_name: str, worker_id: int, **kwargs):
        """
        Wrapper for different loss functions.
        """
        self.name = loss_name

        if self.name == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss()

        elif self.name =='WoLA' or self.name =='WoLA_under_attack':
            self.loss = WoLA(worker_id=worker_id, **kwargs)

        elif self.name =='FedLC':
            self.loss = FedLC(worker_id=worker_id,**kwargs)

        elif self.name =='DMFL':
            self.loss = DMFL(worker_id=worker_id,**kwargs)

        else:
            print(self.name)
            raise ValueError("Unknown loss")

    def __call__(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return self.loss(outputs, labels)

# WoLA
################################################################################################
class WoLA:
    def __init__(
        self,
        device: torch.device,
        local_distributions: Tensor,
        Byzantine_local_distribution: Tensor,
        worker_id: int,
        epsilon: float = 1e-3, 
        under_attack: bool = False,
    ):
        """
        Worker Label Alignment loss (WoLA).
        Computes class weights based on local vs global data imbalance.
        """
        self.device = torch.device(device)
        
        if under_attack:
            self.weight = self.get_weight_under_attack(local_distributions, Byzantine_local_distribution, worker_id, epsilon)
        else:
            self.weight = self.get_weight(local_distributions, Byzantine_local_distribution, worker_id, epsilon)
               

    def get_weight(self, local_distributions: Tensor, Byzantine_local_distribution: Tensor, worker_id: int, epsilon: float) -> Tensor:
        """
        Compute per-class weights based on global and local distributions.
        """
        # Compute global distribution
        global_count = local_distributions.sum(dim=0)
        global_total = global_count.sum()
        global_distribution = global_count / global_total

        # Compute local distribution
        honest_and_Byzantine_distributions = torch.cat((local_distributions, Byzantine_local_distribution), dim=0)
        local_count = honest_and_Byzantine_distributions[worker_id]
        local_total = local_count.sum()
        local_distribution = local_count / local_total

        # Compute weight
        weight = global_distribution / (local_distribution + epsilon)
        return weight.float().to(self.device)

    
    def get_weight_under_attack(self, local_distributions: Tensor, Byzantine_local_distribution: Tensor, worker_id: int, epsilon: float) -> Tensor:
        """
        Compute per-class weights based on global and local distributions.
        """
        # Compute local distribution (even for Byzantine workers, although this is not used in the following, we do it for code consistency)
        honest_and_Byzantine_distributions = torch.cat((local_distributions, Byzantine_local_distribution), dim=0)
        local_count = honest_and_Byzantine_distributions[worker_id]
        local_total = local_count.sum()
        local_distribution = local_count / local_total
    
        # Compute global distribution (impacted by Byzantine attack)
        
        ## Least common class
        global_honest_count = local_distributions.sum(dim=0)
        least_common_class = torch.argmin(global_honest_count)

        ## Compute worst Byzantine distribution (i.e., all mass on the least common class)
        total_Byzantine_count = Byzantine_local_distribution.sum(dim=1)
        worst_Byzantine_local_distribution = torch.zeros_like(Byzantine_local_distribution)    
        worst_Byzantine_local_distribution[:, least_common_class] = total_Byzantine_count

        ## Compute robust distribution
        honest_and_worst_Byzantine_distributions = torch.cat((local_distributions, worst_Byzantine_local_distribution), dim=0)
        global_count = rfa([d.float() for d in honest_and_worst_Byzantine_distributions])
        global_total = global_count.sum()
        global_distribution = global_count / global_total
        
        # Compute weight
        weight = global_distribution / (local_distribution + epsilon)
        return weight.float().to(self.device)

        
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute weighted cross-entropy.
        """
        logits = logits.to(self.device)
        targets = targets.to(self.device)
        
        log_probs = torch.log_softmax(logits, dim=1)
        loss_per_sample = -log_probs[torch.arange(len(targets)), targets]
        loss_per_sample = loss_per_sample * self.weight[targets]

        return loss_per_sample.mean()
        
# FedLC
################################################################################################
class FedLC:
    def __init__(
        self,
        device: torch.device,
        local_distributions: Tensor,
        Byzantine_local_distribution: Tensor,
        worker_id: int,
        tau: float = 1e-1
    ):
        honest_and_Byzantine_distributions = torch.cat((local_distributions, Byzantine_local_distribution), dim=0)
        self.local_count = honest_and_Byzantine_distributions[worker_id].to(device)
        self.tau = tau

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Logits shape
        batch_size, C = logits.shape
    
        # Compute calibrated logits
        calibration_term = self.tau * torch.pow(self.local_count, -1 / 4)
        calibration_term = calibration_term.unsqueeze(0).expand(batch_size, C)
        calibrated_logits = logits - calibration_term
        
        # Predictions
        label_predictions = calibrated_logits[torch.arange(batch_size), targets]
    
        # Loss
        log_denominator = torch.logsumexp(calibrated_logits, dim=1)
        loss = - (label_predictions - log_denominator)
     
        return loss.mean()

# DMFL
################################################################################################
class DMFL:
    def __init__(
        self,
        device: torch.device,
        local_distributions: Tensor,
        Byzantine_local_distribution: Tensor,
        worker_id: int,
        T: float = 1e-1
    ):
        honest_and_Byzantine_distributions = torch.cat((local_distributions, Byzantine_local_distribution), dim=0)
        self.local_count = honest_and_Byzantine_distributions[worker_id].to(device)
        self.T = T

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Logits shape
        batch_size, C = logits.shape
    
        # Compute calibrated logits
        calibration_term = self.T * torch.pow(self.local_count, -1 / 4)
        calibration_term = calibration_term.unsqueeze(0).expand(batch_size, C)
        calibrated_logits = logits.clone()
        calibrated_logits[torch.arange(batch_size), targets] = logits[torch.arange(batch_size), targets] - calibration_term[torch.arange(batch_size), targets]
        
        # Predictions
        label_predictions = calibrated_logits[torch.arange(batch_size), targets]
    
        # Loss
        log_denominator = torch.logsumexp(calibrated_logits, dim=1)
        loss = - (label_predictions - log_denominator)
     
        return loss.mean()