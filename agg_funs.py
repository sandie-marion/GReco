import torch


@torch.no_grad()
def agg_median(tensor: torch.Tensor, f):
    return tensor.median(dim=0)[0]

@torch.no_grad()
def agg_rbrm(tensor: torch.Tensor, f):
    n_cl = len(tensor)
    n_byz = f
    n_byz = min(n_byz, (n_cl - 1) // 2)
    if n_byz == 0:
        agg_tensor = tensor.mean(dim=0)
    else:
        sorted_tensor, _ = tensor.sort(dim=0)
        agg_tensor = sorted_tensor[n_byz:-n_byz].mean(dim=0)
    return agg_tensor


@torch.no_grad()
def agg_krum(tensor: torch.Tensor, f):
    n_cl = len(tensor)
    n_byz = min(f, (n_cl - 3) // 2)

    squared_dists = torch.cdist(tensor, tensor).square()
    topk_dists, _ = squared_dists.topk(k=n_cl - n_byz -1, dim=-1, largest=False, sorted=False)    
    scores = topk_dists.sum(dim=-1)
    _, candidate_idxs = scores.topk(k=n_cl - n_byz, dim=-1, largest=False)

    agg_tensor = tensor[candidate_idxs].mean(dim=0)

    return agg_tensor


@torch.no_grad()
def agg_bulyan(tensor: torch.Tensor, f):
    n_cl = len(tensor)
    n_byz = min(f, (n_cl - 3) // 4)

    squared_dists = torch.cdist(tensor, tensor).square()
    topk_dists, _ = squared_dists.topk(k=n_cl - n_byz -1, dim=-1, largest=False, sorted=False)    
    scores = topk_dists.sum(dim=-1)
    _, krum_candidate_idxs = scores.topk(k=n_cl - 2 * n_byz, dim=-1, largest=False)

    krum_tensor = tensor[krum_candidate_idxs]
    med, _ = krum_tensor.median(dim=0)
    dist = (krum_tensor - med).abs()
    tr_updates, _ = dist.topk(k=n_cl - 4 * n_byz, dim=0, largest=False)
    agg_tensor = tr_updates.mean(dim=0)
    
    return agg_tensor