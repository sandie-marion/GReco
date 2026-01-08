import torch
import torch.nn.functional as F

def get_byzantine_pi(stacked, pi, h, f):
    """
    stacked: (n, D) worker grads
    pi:      (n, C) worker class weights
    h: number of honest workers
    f: number of byzantine workers (assumed to be the last f workers here)
    """
    device = pi.device
    dtype  = pi.dtype
    C = pi.shape[1]

    honest_idx = torch.arange(h, device=device)
    byzantine_idx = torch.arange(h, h + f, device=device)
    
    one_matrix = torch.ones((C, h), device=device, dtype=dtype)
    diag_len = min(C, h)
    one_matrix[torch.arange(diag_len, device=device), torch.arange(diag_len, device=device)] = 0

    selected_grad = stacked[honest_idx].to(dtype)   # (h, D)
    selected_pi   = pi[honest_idx].to(dtype)        # (h, C)

    M = one_matrix @ selected_pi                                   # (C, K)(K, C) -> (C, C)
    B = one_matrix.to(selected_grad.dtype) @ selected_grad         # (C, K)(K, D) -> (C, D)
    global_grads = torch.linalg.pinv(M) @ B                        # (C, D)

    byzantine_grad = stacked[byzantine_idx]             # (f, D)
    GGt = global_grads @ global_grads.T                 # (C, C)

    best_byzantine_pi = byzantine_grad @ global_grads.T @ torch.linalg.pinv(GGt)  # (f, C)

    return best_byzantine_pi

import torch
import torch.nn.functional as F

def select_representative_vectors(X, n, center=True, eps=1e-12):
    if center:
        X = X - X.mean(dim=0, keepdim=True)

    N, d = X.shape
    selected = []
    Q = []

    Xc = X.clone()
    selected_mask = torch.zeros(N, dtype=torch.bool, device=X.device)

    for _ in range(n):
        if len(Q) == 0:
            scores = (Xc**2).sum(dim=1)
        else:
            Qmat = torch.stack(Q, dim=1)      # (d, k)
            proj = Xc @ Qmat @ Qmat.T
            R = Xc - proj
            scores = (R**2).sum(dim=1)

        scores[selected_mask] = -float("inf")
        i = torch.argmax(scores).item()

        r = Xc[i] if len(Q) == 0 else R[i]
        q = r / (r.norm() + eps)

        selected.append(i)
        Q.append(q)
        selected_mask[i] = True

    return torch.tensor(selected)


class GReco:
    """
    Greedy Reconstruction scoring:
    - sample many subsets of size K
    - reconstruct class-wise gradients via pseudo-inverse
    - score workers by reconstruction similarity
    - keep top-h workers from the best subset
    """

    def __init__(self, agg, C: int, h: int, f: int, K: int | None = None):
        self.agg = agg

        self.C = C
        self.h = h
        self.f = f
        self.N = h + f
        self.K = 4
        
        self.T = self._compute_num_trials(alpha=2)

        self.diag_len = min(self.C, self.K)

        self.step = 0

    def _compute_num_trials(self, alpha: float = 1.0) -> int:
        """
        Heuristic number of unique draws.
        p ~ probability a random K-subset is fully honest (hypergeometric-like).
        """
        n = self.N
        p = 1.0
        for k in range(self.K):
            p *= (self.h - k) / (n - k)
        base = int(1.0 / p) + 1
        return int(alpha * base)

    def _draw_unique_subsets(self) -> torch.Tensor:
        """Return (T, K) tensor of unique worker index subsets."""
        draws = []
        seen = set()

        while len(draws) < self.T:
            x = torch.randperm(self.N, device=self.device )[: self.K]
            key = tuple(torch.sort(x).values.tolist())
            if key not in seen:
                seen.add(key)
                draws.append(x)

        return torch.stack(draws, dim=0)

    
    def _draw_unique_subsets2(self) -> torch.Tensor:
        
        x1 = honest_idx = torch.arange(self.f, device=self.device)
        x2 = byzantine_idx = torch.arange(self.h, self.h + self.f, device=self.device)
        x3 = byzantine_idx = torch.arange(self.h-int(self.f/2), self.h + int(self.f/2), device=self.device)
        
        draws = [x1, x2, x3]
        
        return draws

    
    def _solve_global_grads(
        self,
        sel_pi: torch.Tensor,      # (N, C)
        sel_grads: torch.Tensor,   # (N, D)
    ) -> torch.Tensor:
        return torch.linalg.pinv(sel_pi) @ sel_grads   # (C, D)

        
    def _score_subset(self, subset: torch.Tensor, grads: torch.Tensor, pi: torch.Tensor):
        """
        subset: (K,) long indices
        grads:  (N, D)
        pi:     (N, C)
        """
        subset = subset.long()

        sel_grads = grads[subset]
        sel_pi = pi[subset]

        global_grads = self._solve_global_grads(sel_pi, sel_grads)   # (C, D) typically
        recon = (pi @ global_grads).to(grads.dtype)                  # (N, D)

        recon_n = torch.nn.functional.normalize(recon, dim=1)
        grads_n = torch.nn.functional.normalize(grads, dim=1)
        
        # Pairwise distance matrices (N, N)
        dist_rec = torch.cdist(recon_n, recon_n, p=2)
        dist_ori = torch.cdist(grads_n, grads_n, p=2)
    
        # Elementwise mismatch (N, N)
        dist_diff = (dist_rec - dist_ori)**2

        # Global score: median of per-sample scores (scalar)
        score = dist_diff.sum().item()
    
        # Pick best h (smallest mismatch)
        per_sample = ((recon_n-grads_n)**2).sum(dim=1) # dist_diff.sum(dim=1)
        top = torch.argsort(per_sample)[:self.h]
    
        return score, top
        
    def __call__(self, pi: torch.Tensor, inputs: list[torch.Tensor]):
        grads = torch.stack(inputs, dim=0)  # (N, D)
        self.device = grads.device
        self.idx = torch.arange(self.diag_len, device=self.device)
        
        #subsets = self._draw_unique_subsets()

        subsets = [select_representative_vectors(grads, self.K, center=True, eps=1e-12)]

        best_score = float("inf")
        best_idx = None

        for subset in subsets:
            score, keep_idx = self._score_subset(subset, grads, pi)

            """stat0 = score
            stat1 = int(100*sum(subset<self.h).item()/len(subset))
            stat2 = int(100*sum(keep_idx<self.h).item()/len(keep_idx))
            print('Score', stat0, ' Workers', stat1, 'Filter', stat2)
            """
            if score < best_score:
                best_score = score
                best_idx = keep_idx
                best_workers = subset

        stat0 = best_score
        stat1 = int(100*sum(best_workers<self.h).item()/len(best_workers))
        stat2 = int(100*sum(best_idx<self.h).item()/len(best_idx))
        print('-------------------------')
        print('Score', stat0, ' Workers', stat1, 'Filter', stat2)
        print('-------------------------')
        #print('-------------------------')
        #filtered = [inputs[i] for i in best_idx.tolist()]
        grad = 0
        for i in best_idx:
            grad = grad + inputs[i]
        grad = grad / len(best_idx)
        return grad #self.agg(filtered)
   
        