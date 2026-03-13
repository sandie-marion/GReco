import time
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
from itertools import combinations
import random


class HullGuard:
    def __init__(
        self, 
        agg, C: int, 
        h: int, 
        f: int
        ):
        
        self.agg = agg  # Robust aggregation rule used to combine worker updates
    
        self.C = C      # Number of classes
        self.h = h      # Number of honest workers
        self.f = f      # Number of Byzantine workers
        self.N = h + f  # Total number of workers
    
        self.num_subsets = 1000     # Number of candidate worker subsets sampled
        self.num_sub_subsets = 5   # Number of sub-subsets used to estimate the variance of each subset
        
        self.sub_subset_size = 2 * self.C  # Size of each sub-subset
        
        self.window_size = 1  # Number of recent rounds used to score workers
        self.window = []      # Stores worker scores over the last `window_size` rounds

    
    def class_gradient_estimator(
        self, 
        pi: torch.Tensor, 
        grads: torch.Tensor, 
        subset: torch.Tensor
        ) -> torch.Tensor:
    
        # Select the workers belonging to the subset
        pi_subset = pi[subset]          # (k, C) class distributions of the selected workers
        grads_subset = grads[subset]    # (k, D) gradients reported by the selected workers
    
        # Estimate the class-wise gradients by solving:
        #     grads_subset â‰ˆ pi_subset @ G
        # using the Mooreâ€“Penrose pseudo-inverse
        pinv = torch.linalg.pinv(pi_subset)
        G = pinv @ grads_subset
            
        return G  # (C, D) estimated gradient for each class

        
    def gradient_checker(
        self, 
        grads: torch.Tensor
        ):
    
        # Ensure gradients are provided
        if grads is None:
            raise ValueError("grads must not be None.")
    
        # Check that all gradient values are finite
        # (detects NaN or Â±Inf which would break aggregation)
        if not torch.isfinite(grads).all():
            raise ValueError("grads contains NaN or Inf values.")

    def pi_checker_normalizer(
        self, 
        pi: torch.Tensor
        ) -> torch.Tensor:
        """
        Row-normalize the assignment matrix 'pi'.

        Each row is scaled so that its entries sum to 1, producing a
        valid class distribution for each worker.
        """
        
        # Validate input
        if pi is None:
            raise ValueError("pi must not be None.")
    
        if not torch.isfinite(pi).all():
            raise ValueError("pi contains NaN or Inf values.")
        
        if torch.any(pi < 0):
            raise ValueError("pi must contain non-negative values.")
    
        # Compute row sums
        row_sum = pi.sum(dim=1, keepdim=True)
    
        # Ensure rows have strictly positive marginals
        if torch.any(row_sum <= 0):
            raise ValueError("Each row of pi must have a strictly positive sum.")
    
        # Normalize rows so that they sum to 1
        pi = pi / row_sum
    
        return pi

    
    def get_subsets(self) -> Tuple[torch.Tensor, torch.Tensor]:
         
        # Boolean matrix indicating worker membership in each subset
        # candidate_matrix[i, j] = True if worker j belongs to subset i
        candidate_matrix = torch.zeros(
            self.num_subsets, self.N,
            device=self.device,
            dtype=torch.bool
        )

        # Uniform sampling weights over workers
        weights = torch.ones(self.N, device=self.device)

        subsets = []
        for k in range(self.num_subsets):

            # Sample a subset of h workers without replacement
            subset = torch.multinomial(weights, self.h, replacement=False)

            # Mark selected workers in the candidate matrix
            candidate_matrix[k, subset] = True

            subsets.append(subset)

        # Tensor containing the worker indices for each subset
        subsets = torch.stack(subsets, dim=0)  # (num_subsets, h)
    
        return subsets, candidate_matrix

    
    def get_sub_subsets(
        self, 
        subset: torch.Tensor
        ) -> torch.Tensor:
        """
        Sample sub-subsets from a given subset of workers.

        From a subset of size `h`, the method samples `num_sub_subsets`
        sub-subsets of size `sub_subset_size`. Sampling is performed
        without replacement within each sub-subset.

        Args:
            subset: Tensor of worker indices, shape (h,).

        Returns:
            Tensor of sampled worker indices with shape
            (num_sub_subsets, sub_subset_size).
        """
        subset = subset.to(self.device)

        sub_subsets = []
        for _ in range(self.num_sub_subsets):

            # Randomly select indices within the subset
            selected = torch.randperm(self.h, device=self.device)[: self.sub_subset_size]

            # Map them to the corresponding worker indices
            sub_subsets.append(subset[selected])

        # Stack sampled sub-subsets
        return torch.stack(sub_subsets, dim=0)  # (num_sub_subsets, sub_subset_size)


    def variance_estimator(
        self, 
        lst_grad: list[torch.Tensor]
        ) -> float:
        """
        Estimate the dispersion of class-wise gradient estimates.
    
        Computes the mean squared Frobenius distance between each
        gradient estimate and their mean:
    
            (1/K) Î£ ||G_k - Î¼||_F^2
    
        where G_k âˆˆ R^{CÃ—D}.
        """
    
        # Stack estimates: (K, C, D)
        G = torch.stack(lst_grad, dim=0)
    
        # Mean estimate
        mu_hat = G.mean(dim=0, keepdim=True)
    
        # Squared Frobenius deviations
        diffs = G - mu_hat
        variance_estimator = torch.norm(diffs, p='fro', dim=(1, 2)).pow(2).mean()
    
        return float(variance_estimator)


    def window_append(
        self, 
        new_element: torch.Tensor
        ) -> torch.Tensor:
        """
        Append a new element to the sliding window and return the window mean.
    
        The window stores the most recent `window_size` elements. If the window
        exceeds this size, the oldest element is removed.
    
        Args:
            new_element: Tensor to append to the window.
    
        Returns:
            Tensor equal to the mean of the elements currently in the window.
        """
    
        # Add new element
        self.window.append(new_element)
    
        # Maintain fixed window size
        if len(self.window) > self.window_size:
            self.window.pop(0)  # remove oldest element
    
        # Compute window mean
        return torch.stack(self.window, dim=0).mean(dim=0)
    
        
    def get_candidates(
        self
        ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Sample candidate subsets and, for each subset, sample sub-subsets.
    
        Returns:
            candidates: list of length `num_subsets`. Each element is a tuple
                (subset, sub_subsets) where:
                    subset:      Tensor of worker indices, shape (h,)
                    sub_subsets: Tensor of sampled sub-subsets, shape
                                 (num_sub_subsets, sub_subset_size)
    
            candidate_matrix: Boolean tensor of shape (num_subsets, N) indicating
                worker membership in each candidate subset.
        """

        # Sample candidate subsets
        subsets, candidate_matrix = self.get_subsets()  # (num_subsets, h)
    
        candidates = []
        for subset in subsets:
    
            # Sample sub-subsets from the current subset
            sub_subsets = self.get_sub_subsets(subset)  # (num_sub_subsets, sub_subset_size)
    
            candidates.append((subset, sub_subsets))
    
        return candidates, candidate_matrix

        
    def compute_scores(
            self,
            candidates,
            variances,
            candidate_matrix: torch.Tensor
            ) -> torch.Tensor:
        """
        Compute a score for each worker based on subset variances.
    
        For worker n, compare the variances of candidate subsets that contain n
        with those of candidate subsets that do not contain n. The score is the
        proportion of pairwise comparisons in which a subset containing n has a
        lower variance than a subset not containing n.
    
        Args:
            candidates: Candidate subsets and associated sub-subsets.
                Included for interface consistency, but not used here.
            variances: Variance value for each candidate subset, shape (num_subsets,).
            candidate_matrix: Boolean matrix of shape (num_subsets, N), where
                candidate_matrix[i, n] is True if worker n belongs to subset i.
    
        Returns:
            Tensor of shape (N,) containing one score per worker.
        """
        variances = torch.as_tensor(variances, device=self.device, dtype=torch.float32)
    
        # Score of each worker
        scores = torch.zeros(self.N, device=self.device)
    
        for n in range(self.N):
            # Candidate subsets that contain / do not contain worker n
            present_idx = candidate_matrix[:, n]
            absent_idx = ~present_idx
    
            present_vars = variances[present_idx]
            absent_vars = variances[absent_idx]
    
            p = present_vars.numel()
            a = absent_vars.numel()
    
            # Skip workers that cannot be compared
            if p == 0 or a == 0:
                continue
    
            # Pairwise comparison between variances of subsets containing n
            # and subsets not containing n
            cmp = present_vars[:, None] - absent_vars[None, :]
    
            # Count how often subsets containing n have lower variance
            score = (cmp < 0).float().sum() 
    
            # Normalize by the total number of pairwise comparisons
            scores[n] = score / (p * a)
    
        return scores

    
    def compute_variances(
        self, 
        candidates, 
        pi: torch.Tensor, 
        grads: torch.Tensor
        ) -> list[float]:
        """
        Compute a variance score for each candidate subset.
    
        For each candidate subset, several sub-subsets are sampled. For each
        sub-subset, class-wise gradients are estimated, and the variance of
        these estimates (measured with the Frobenius norm) is used as a score
        for the candidate subset.
    
        Args:
            candidates: list of tuples (subset, sub_subsets) where
                subset:      Tensor of worker indices, shape (h,)
                sub_subsets: Tensor of shape (num_sub_subsets, sub_subset_size)
            pi: Assignment matrix, shape (N, C)
            grads: Worker gradients, shape (N, D)
    
        Returns:
            List of variance scores, one per candidate subset.
        """
            
        variances = []
        self.estimates = []
    
        for _, sub_subsets in candidates:
    
            # Collect gradient estimates from each sub-subset
            grad_estimates = []
            for sub_subset in sub_subsets:
                G = self.class_gradient_estimator(pi, grads, sub_subset)  # (C, D)
                grad_estimates.append(G)
    
            # Compute Frobenius variance across the estimates
            var = self.variance_estimator(grad_estimates)
            variances.append(var)
    
        return variances

    
    def __call__(
        self, 
        pi: list[torch.Tensor], 
        inputs: list[torch.Tensor]
        ) :
        """
        Select a subset of workers based on variance-derived scores and
        aggregate their gradients with a robust aggregator.
        
        Args:
            pi: List of worker assignment vectors, one per worker.
            inputs: List of worker gradients, one per worker.
        
        Returns:
            Aggregated robust gradient.
        """
        
        start = time.time()
        
        # Stack worker gradients to form the working tensor
        grads = torch.stack(inputs, dim=0)  # (N, D)
        self.gradient_checker(grads)
        
        # Stack and normalize the distribution matrix pi
        pi = torch.stack(pi, dim=0)         # (N, C)
        pi = self.pi_checker_normalizer(pi)
        
        # Use the gradient device for all subsequent computations
        self.device = grads.device
        
        # Sample candidate subsets and their associated sub-subsets
        candidates, candidate_matrix = self.get_candidates()
        
        # Compute one variance score per candidate subset
        variances = self.compute_variances(candidates, pi, grads)
        
        # Derive one score per worker from candidate variances
        scores = self.compute_scores(candidates, variances, candidate_matrix)
        
        # Smooth worker scores over the last few rounds
        scores = self.window_append(scores)
        
        # Select the h highest-scoring workers
        best_subset = torch.topk(scores, self.h)[1]
        
        # Aggregate gradients from the selected workers
        robust_grad = robust_aggregator(grads[best_subset])
        
        end = time.time()

        filter_score = (best_subset < self.h).sum().item()
        
        #print('num of honest:', filter_score,  '/', self.h)
        #print("running time:", round(end - start, 2))
        
        return robust_grad, filter_score


def robust_aggregator(grads: torch.Tensor) -> torch.Tensor:
    """
    Robust aggregation using bucketing followed by a coordinate-wise median.

    The worker gradients are first randomly shuffled and partitioned into
    buckets of fixed size. The mean gradient of each bucket is computed,
    and the final aggregate is obtained by taking the coordinate-wise
    median across bucket means.

    Args:
        grads: Tensor of worker gradients of shape (n, d),
                where n is the number of workers and d the gradient dimension.

    Returns:
        Tensor of shape (d,) containing the robust aggregated gradient.
    """

    # Bucket size used for bucketing aggregation
    bucket_size = 5

    n, d = grads.shape

    # Shuffle workers to randomize bucket composition
    perm = torch.randperm(n, device=grads.device)
    shuffled = grads[perm]

    # Split shuffled gradients into buckets
    chunks = torch.split(shuffled, bucket_size, dim=0)

    # Compute the mean gradient of each bucket
    bucket_means = torch.stack(
        [chunk.mean(dim=0) for chunk in chunks],
        dim=0
    )

    # Coordinate-wise median across bucket means
    robust_grad = bucket_means.median(dim=0).values

    return robust_grad


class DistributionAttack:
    """
    Choose (or estimate) a class-mixture vector alpha on the simplex.

    Modes:
      - "dirac":      fixed one-hot vector (chosen once)
      - "uniform":    fixed uniform vector
      - "random":     fixed Dirichlet(1) sample (chosen once)
      - "projection": compute alpha by least-squares + simplex constraint (varies per call)

    Note: device/dtype are only known from cw_grads, so for fixed modes we
    store alpha on CPU and cast on each call (values stay constant).
    """

    def __init__(self, name: str, C: int, h:int):
        self.name = name.lower()
        self.C = C
        self.h = h

        if self.name not in {"dirac", "uniform", "random", "projection"}:
            raise ValueError(
                f"Unknown name={name!r}. Expected one of: "
                f"'dirac', 'uniform', 'random', 'projection'."
            )

        if self.name == "projection":
            alpha = None

        elif self.name == "dirac":
            idx = torch.randint(low=0, high=self.C, size=(1,)).item()
            alpha = torch.zeros(self.C, dtype=torch.float32, device="cpu")
            alpha[idx] = 1.0

        elif self.name == "uniform":
            alpha = torch.full((self.C,), 1.0 / self.C, dtype=torch.float32, device="cpu")

        elif self.name == "random":
            dist = torch.distributions.Dirichlet(torch.ones(self.C, dtype=torch.float32, device="cpu"))
            alpha = dist.sample()

        self._alpha_cpu = alpha

    def __call__(self, inputs: list[torch.Tensor], pi: torch.Tensor) -> torch.Tensor:
        device = inputs[0].device
        dtype = inputs[0].dtype
        
        if self.name == "projection":
            honest_grads = torch.stack(inputs[:self.h], dim=0)          # (N, D)
            honest_pi = torch.stack(pi[:self.h], dim=0)
            byz_grad = inputs[-1]
            cw_grads = self.class_gradient_estimator(honest_pi, honest_grads)
 
            return self.active_set_simplex_projection(cw_grads, byz_grad)
        
        else:
            return self._alpha_cpu.to(device=device, dtype=dtype)

    
    @staticmethod
    def class_gradient_estimator(
        honest_pi: torch.Tensor,
        honest_grads: torch.Tensor
        ) -> torch.Tensor:

        # Solve: pi_subset @ G â‰ˆ grads_subset => G = pinv(pi_subset) @ grads_subset
        G = torch.linalg.pinv(honest_pi) @ honest_grads
    
        return G
        
    @staticmethod
    def active_set_simplex_projection(
        cw_grads: torch.Tensor,
        byz_grad: torch.Tensor,
        eps: float = 1e-10,
        ridge: float = 1e-12,
        max_iter: int = 10_000,
    ) -> torch.Tensor:
        """
        Solve:
            min_a ||byz_grad - cw_grads^T a||^2
            s.t.  a >= 0, sum(a)=1

        cw_grads: (C, D)  rows are class-wise gradients in R^D
        byz_grad: (D,) or (D,1)
        returns:  (C,)
        """
        if byz_grad.ndim == 2:
            if byz_grad.shape[1] != 1:
                raise ValueError(f"byz_grad must be (D,) or (D,1), got {byz_grad.shape}")
            byz_grad = byz_grad[:, 0]

        if cw_grads.ndim != 2 or byz_grad.ndim != 1:
            raise ValueError(
                f"Expected cw_grads (C,D) and byz_grad (D,), got "
                f"{cw_grads.shape=} and {byz_grad.shape=}"
            )

        C, D = cw_grads.shape
        if byz_grad.shape[0] != D:
            raise ValueError(f"Shape mismatch: cw_grads is {cw_grads.shape} but byz_grad is {byz_grad.shape}")

        device, dtype = cw_grads.device, cw_grads.dtype

        G = cw_grads @ cw_grads.T                         # (C, C)
        c = cw_grads @ byz_grad                           # (C,)

        G = G + ridge * torch.eye(C, device=device, dtype=dtype)

        active = torch.ones(C, dtype=torch.bool, device=device)
        alpha = torch.zeros(C, device=device, dtype=dtype)

        for _ in range(max_iter):
            idx = torch.where(active)[0]
            k = idx.numel()

            if k == 0:
                j = torch.argmax(c)
                alpha.zero_()
                alpha[j] = 1.0
                return alpha

            Gs = G.index_select(0, idx).index_select(1, idx)  # (k, k)
            cs = c.index_select(0, idx)                       # (k,)
            ones = torch.ones(k, device=device, dtype=dtype)

            try:
                x = torch.linalg.solve(Gs, cs)
                y = torch.linalg.solve(Gs, ones)
            except RuntimeError:
                Gs = Gs + (10.0 * ridge + 1e-10) * torch.eye(k, device=device, dtype=dtype)
                x = torch.linalg.solve(Gs, cs)
                y = torch.linalg.solve(Gs, ones)

            denom = ones @ y
            if torch.abs(denom) < eps:
                j = torch.argmax(c)
                alpha.zero_()
                alpha[j] = 1.0
                return alpha

            lam = (ones @ x - 1.0) / denom
            alpha_s = x - lam * y

            alpha.zero_()
            alpha[idx] = alpha_s

            neg = alpha_s < -eps
            if not torch.any(neg):
                alpha.clamp_(min=0.0)
                s = alpha.sum()
                if s <= eps:
                    j = torch.argmax(c)
                    alpha.zero_()
                    alpha[j] = 1.0
                else:
                    alpha /= s
                return alpha

            neg_idx = idx[neg]
            worst = neg_idx[torch.argmin(alpha[neg_idx])]
            active[worst] = False

        raise RuntimeError("active-set did not converge (max_iter reached)")