import torch
from utility import euclidean_distance
from Defenses.GReco import GReco
from agg_funs import agg_bulyan
from math import floor 

class Aggregator:
    def __init__(self, 
                 aggregator_name: str, 
                 aggregator_args: dict, 
                 pre_aggregator_name: str, 
                 pre_aggregator_args: dict) -> None:
        
        self.aggregator_name = aggregator_name
        self.aggregator_args = aggregator_args
        self.pre_aggregator_name = pre_aggregator_name
        self.pre_aggregator_args = pre_aggregator_args

        # Initialize pre-aggregator
        self.pre_aggregator = self._init_pre_aggregator()

        # Initialize aggregator
        self.aggregator = self._init_aggregator()

    def _init_pre_aggregator(self):
        """ 
        Initializes the pre-aggregation method.
        """
        if self.pre_aggregator_name == 'NNM':
            return lambda inputs: NNM(inputs, **self.pre_aggregator_args)
        
        elif self.pre_aggregator_name == 'BKT':
            return lambda inputs: Bucketing(inputs, **self.pre_aggregator_args)
        
        elif self.pre_aggregator_name == 'FoundFL':
            return lambda inputs: FoundationFL(inputs, **self.pre_aggregator_args)
        
        elif self.pre_aggregator_name == 'ARC' :
            return lambda inputs: NNM(AdaptiveRobustClipping(inputs, **self.pre_aggregator_args), **self.pre_aggregator_args) 
        
        elif self.pre_aggregator_name == 'None':
            return lambda inputs: inputs
        
        else:
            raise("Unknown pre-aggregator")
            
    def _init_aggregator(self):
        """ 
        Initializes the main aggregation method. 
        """
        if self.aggregator_name == 'Mean':
            return mean
        
        elif self.aggregator_name == 'CWMed':
            return coordinate_wise_median
            
        elif self.aggregator_name == 'CwTM':
            return lambda inputs: coordinate_wise_trimmed_mean(inputs, **self.aggregator_args)
        
        elif self.aggregator_name == 'RFA':
            return lambda inputs: rfa(inputs, **self.aggregator_args)
        
        elif self.aggregator_name == 'Krum':
            return lambda inputs: multi_krum(inputs, **self.aggregator_args)
        
        elif self.aggregator_name == 'GReco':
            f_rfa =  lambda inputs: rfa(inputs)
            greco = GReco(agg=f_rfa, **self.aggregator_args)
            return lambda inputs, pi: greco(inputs, pi)

        elif self.aggregator_name == 'GAS':
            return lambda inputs : gas_aggregate(inputs, **self.aggregator_args)

        else:
            raise ValueError("Unknown aggregator")

    def __call__(self, inputs: list, pi = None, selected_byz = None) -> torch.Tensor:
        if pi is not None:
            return self.aggregator(pi, inputs)
        elif selected_byz is not None : 
            return self.aggregator(inputs, selected_byz)
            
        else:
            return self.aggregator(self.pre_aggregator(inputs))
        
# Pre-aggregator
################################################################################################
################################################################################################

# NNM
################################################################################################
def NNM(inputs: list, f: int) -> torch.Tensor:
    """
    Perform Neighborhood Noise Mixing (NNM) on the input vectors using PyTorch.

    Args:
        inputs (torch.Tensor): Tensor of shape (n, d), where n = number of workers, d = dimension.
        f (int): Fault parameter.

    Returns:
        torch.Tensor: Tensor of shape (n, d), the averaged row vectors.
    """
    inputs = torch.stack(inputs, dim=0)
    
    n, d = inputs.shape

    # Compute squared pairwise distances (n x n)
    # dist[i,j] = ||inputs[i] - inputs[j]||^2
    dist = torch.cdist(inputs, inputs, p=2)

    # For each i, get indices of n-f-1 closest neighbors (excluding itself)
    # Set diagonal to large value to ignore self-distance
    dist.fill_diagonal_(float('inf'))
    _, idx = torch.topk(dist, k=n-f-1, dim=1, largest=False)

    # Gather neighbors for each row
    neighbors = inputs[idx]  # shape: (n, n-f-1, d)

    # Include the original vector itself
    combined = torch.cat([inputs.unsqueeze(1), neighbors], dim=1)  # (n, n-f, d)

    # Average
    z_outputs = combined.mean(dim=1)  # (n, d)

    return [z_output for z_output in z_outputs]

# Bucketing
################################################################################################
def Bucketing(inputs: list[torch.Tensor], s: int) -> torch.Tensor:
    """
    Bucketing aggregation (PyTorch version).

    This function shuffles the input vectors, splits them into buckets of size `s`,
    and computes the mean vector for each bucket.

    Args:
        inputs (list[torch.Tensor]): A list of 1D tensors, each representing a vector of shape (d,).
        s (int): Bucket size (number of vectors per bucket).

    Returns:
        torch.Tensor: A tensor of shape (num_buckets, d), where each row is the mean
                      of one bucket. If n is not divisible by s, the last bucket will
                      contain fewer than s vectors.
    """
    # Stack list of vectors into a matrix of shape (n, d)
    inputs = torch.stack(inputs, dim=0)   # (n, d)

    # Number of input vectors and dimension
    n, d = inputs.shape

    # Generate a random permutation of indices [0, 1, ..., n-1]
    perm = torch.randperm(n)

    # Reorder the inputs according to the random permutation
    shuffled = inputs[perm]

    # Split into chunks (buckets) of size s along the first dimension
    # Note: if n is not divisible by s, the last chunk will be smaller
    chunks = torch.split(shuffled, s, dim=0)

    # Compute the mean vector for each bucket
    means = torch.stack([c.mean(dim=0) for c in chunks], dim=0)  # (num_buckets, d)

    return [z_output for z_output in means] 

# FoundationFL
################################################################################################
def compute_s_score(g: torch.Tensor, g_max: torch.Tensor, g_min: torch.Tensor) -> float:
    s_min = euclidean_distance(g, g_min)
    s_max = euclidean_distance(g, g_max)
    return min(s_min, s_max)
    
def FoundationFL(inputs: list, m: int) -> list:
    stacked_inputs = torch.stack(inputs, dim=0)    
    g_max, _ = stacked_inputs.max(dim=0)
    g_min, _ = stacked_inputs.min(dim=0)
    s_scores = [compute_s_score(g, g_max, g_min) for g in inputs]
    i_star = torch.argmax(torch.tensor(s_scores)).item()
    selected = inputs[i_star]
    for _ in range(m):
        inputs.append(selected.clone())
    return inputs




# Adaptive Robust Clipping 
##########################

def AdaptiveRobustClipping (inputs: list, f :int) -> list :
    n = len(inputs)
    k = floor(2*(f/n)*(n-f))

    #calculate norms of gradients
    norms = []
    for gradient in inputs : 
        norms.append(torch.norm(gradient)) 
    
    norms_tensor = torch.stack(norms) 

    #sort args 
    sorted_idx = torch.argsort(norms_tensor).tolist() 

    #clip the (k+1) smallest gradients (in norm)
    clip_threshold = norms[sorted_idx[k+1]] 

    for i in range (k, n) : 
        clip_factor = clip_threshold/norms_tensor[sorted_idx[i]]
        inputs[sorted_idx[i]] = clip_factor*inputs[sorted_idx[i]] 

    return inputs 
    
# Aggregator
################################################################################################
################################################################################################

# Mean
################################################################################################
def mean(inputs:list) -> torch.Tensor:
    """
    Compute the mean of a list of row gradients.

    Returns:
    The mean of the row gradients.
    """
    return torch.stack(inputs, dim=0).mean(dim=0)

# Coordinate-wise Median
################################################################################################
def coordinate_wise_median(inputs: list, f=None)-> torch.Tensor:
    """
    Compute the coordonate-wise median of a list of row gradients.

    Returns:
    The coordonatewise median of the row gradients.
    """
    stacked = torch.stack(inputs, dim=0)
    median_values, _ = stacked.median(dim=0)
    return median_values

# Trimmed Mean
################################################################################################
def coordinate_wise_trimmed_mean(inputs: list[torch.Tensor], q: int) -> torch.Tensor:
    """
    Compute the coordinate-wise trimmed mean of row gradients by removing 
    the top-q largest and bottom-q smallest values along each coordinate.

    Args:
        inputs (list[Tensor]): List of tensors of shape (d,)
        q (int): Number of smallest and largest values to trim along each coordinate

    Returns:
        Tensor: Coordinate-wise trimmed mean of shape (d,)
    """
    stacked = torch.stack(inputs, dim=0)  # shape (n, d)
    n = stacked.shape[0]

    # Sort along worker dimension
    sorted_vals, _ = torch.sort(stacked, dim=0)

    # Drop q smallest and q largest along each coordinate
    trimmed = sorted_vals[q:n-q, :]  # shape (n-2q, d)

    # Mean of remaining
    return trimmed.mean(dim=0)

# Multi-Krum
################################################################################################
@torch.no_grad()
def multi_krum(inputs: list[torch.Tensor] | torch.Tensor, f: int) -> torch.Tensor:
    """
    Multi-Krum aggregation (PyTorch version).

    This function implements the Multi-Krum rule for robust aggregation:
    1. Compute pairwise Euclidean distances between all input vectors.
    2. For each vector, compute its Krum score as the sum of distances
       to its q1 = n - f - 2 closest neighbors.
    3. Select q2 = n - f vectors with the smallest Krum scores.
    4. Return the average of the selected vectors.

    Args:
        inputs (list[torch.Tensor] | torch.Tensor):
            Either a list of 1D tensors of shape (d,) or a single tensor
            of shape (n, d), where n is the number of workers.
        f (int): Number of Byzantine (potentially faulty) workers to tolerate.

    Returns:
        torch.Tensor: A tensor of shape (d,) representing the aggregated vector.
    """
    inputs = torch.stack(inputs, dim=0)

    n, d = inputs.shape

    # Define q1 (#neighbors to consider) and q2 (#vectors to keep)
    q1 = n - f - 2
    q2 = n - f

    # Pairwise Euclidean distances (n, n)
    dist = torch.cdist(inputs, inputs, p=2)
    dist.fill_diagonal_(float('inf'))  # exclude self-distance

    # Krum score for each vector = sum of its q1 closest distances
    topk_vals, _ = torch.topk(dist, k=q1, largest=False, dim=1)  # smallest q1
    scores = topk_vals.sum(dim=1)  # (n,)

    # Select q2 vectors with the lowest scores
    _, idx = torch.topk(scores, k=q2, largest=False)
    selected = inputs.index_select(0, idx)  # (q2, d)

    # Aggregate = average of selected vectors
    return selected.mean(dim=0)

# Geometric Median
################################################################################################
@torch.no_grad()
def smoothed_weiszfeld_torch(
    inputs: torch.Tensor,
    alphas: torch.Tensor | None = None,
    z0: torch.Tensor | None = None,
    nu: float = 0.1,
    T: int = 3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Vectorized smoothed Weiszfeld iteration in PyTorch.
    inputs: (m, d) tensor of worker vectors
    alphas: (m,) tensor of nonnegative weights (defaults to uniform)
    z0: (d,) initial point (defaults to zeros_like inputs[0])
    nu: smoothing parameter
    T: number of iterations
    """
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    inputs = inputs.contiguous()
    m, d = inputs.shape
    device = inputs.device
    dtype = inputs.dtype

    if alphas is None:
        alphas = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    else:
        alphas = torch.as_tensor(alphas, device=device, dtype=dtype)

    z = torch.zeros(d, device=device, dtype=dtype) if z0 is None else z0.to(device=device, dtype=dtype)

    for _ in range(T):
        # distances to current z
        dist = torch.norm(inputs - z, dim=1)                         # (m,)
        betas = alphas / torch.clamp(dist, min=nu)                   # (m,)
        num = (betas[:, None] * inputs).sum(dim=0)                   # (d,)
        den = betas.sum() + eps
        z = num / den

    return z

@torch.no_grad()
def rfa(inputs: list, T: int = 3, nu: float = 0.1) -> torch.Tensor:
    """
    Robust Federated Aggregation using smoothed Weiszfeld.
    inputs: (m, d) tensor
    """
    inputs = torch.stack(inputs, dim=0)
    return smoothed_weiszfeld_torch(inputs, alphas=None, z0=torch.zeros_like(inputs[0]), nu=nu, T=T)




#################################################

# Byzantine GAS with multi krum base aggregator 

#################################################




@torch.no_grad()
def gas_aggregate(inputs : list, f : int, gas_p : int, base_agg : str):
    flat_models = torch.stack(inputs)
    groups = split(flat_models, gas_p)
    # identification
    n_cl = len(inputs)
    n_sel_byz = f
    identification_scores = torch.zeros(n_cl)

    if base_agg == 'krum' :
        base_agg = multi_krum
    elif base_agg == 'bulyan' :
        base_agg = agg_bulyan
    elif base_agg == 'median' :
        base_agg = coordinate_wise_median

    for group in groups:
        group_agg = base_agg(list(group), f)
        group_scores = (group - group_agg).square().sum(dim=-1).sqrt().cpu()
        identification_scores += group_scores
    _, cand_idxs = identification_scores.topk(k=n_cl - n_sel_byz, largest=False)
    # aggregation
    flat_agg_model = flat_models[cand_idxs].mean(dim=0)
    
    return flat_agg_model


@torch.no_grad()
def split(flat_models, gas_p):
    d = flat_models.shape[1]
    shuffled_dims = torch.randperm(d).to(flat_models.device)
    p = gas_p
    partition = torch.chunk(shuffled_dims, chunks=p)
    groups = [flat_models[:, partition_i] for partition_i in partition]
    return groups