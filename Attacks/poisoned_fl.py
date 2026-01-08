import numpy as np
import torch
from scipy.special import betainc


def _binom_cdf(k, n, p):
    # regularized incomplete beta
    return betainc(n - k, k + 1, 1 - p)


def binomial_quantile(n, q):
    p = 0.5
    low, high = 0, n
    while low < high:
        mid = (low + high) // 2
        if _binom_cdf(mid, n, p) < q:
            low = mid + 1
        else:
            high = mid
    return low


def rand_pm1(n, device=None):
    """
    Generate a Tensor of size n with values +1 or -1,
    each with probability 0.5.
    """
    r = torch.randint(low=0, high=2, size=(n,), device=device, dtype=torch.int8)
    return r * 2 - 1


class PoisonedFL:
    def __init__(self, net):
        self.model_size = sum(p.numel() for p in net.parameters())
        self.device = next(net.parameters()).device
        self.s = torch.randint(0, 2, (self.model_size,), device=self.device).float().mul_(2).sub_(1)
        self.k_99 = binomial_quantile(self.model_size, 0.99)
        
        self.k_t_minus_1_times_s = None
        self.c_t = 10.0
        self.previous_50_model = [p.data.clone() for p in net.parameters()]
        
    def __call__(self, t, net, row_honest_gradients):
        if t > 1:
            current_model = [p.detach().clone() for p in net.parameters()]
            
            # g^{t-1}
            current_model_vector = torch.cat([x.view(-1) for x in current_model], dim=0)
            previous_model_vector = torch.cat([x.view(-1) for x in self.previous_model], dim=0)
            self.previous_model = current_model

            g_t_minus_1 = current_model_vector - previous_model_vector
            g_t_minus_1_norm = torch.norm(g_t_minus_1)
        
            # k^{t-1}·s (keep it 1D)
            k_t_minus_1_times_s = (
                torch.zeros_like(self.s) if self.k_t_minus_1_times_s is None
                else self.k_t_minus_1_times_s
            )
            k_t_minus_1_times_s_norm = torch.norm(k_t_minus_1_times_s)

            # v^t
            norm_ratio = g_t_minus_1_norm / (k_t_minus_1_times_s_norm + 1e-9)
            v = g_t_minus_1 - norm_ratio * k_t_minus_1_times_s
            v_t = torch.abs(v)/torch.norm(v)

            previous_50_model = self.previous_50_model
            if t % 50 == 0:
                previous_50_model_vector = torch.cat([x.view(-1) for x in previous_50_model], dim=0)
                total_update = current_model_vector - previous_50_model_vector
                total_update = torch.where(total_update == 0, current_model_vector, total_update)
                
                aligned_dim_cnt = (torch.sign(total_update) == torch.sign(self.s)).sum().item()
                
                if aligned_dim_cnt < self.k_99 and self.c_t * 0.7 >= 0.5:
                    self.c_t *= 0.7
                    
                self.previous_50_model = current_model
    
            lamda_t = self.c_t * g_t_minus_1_norm
           
            k_t = lamda_t * v_t  
            
            poison_point = k_t * self.s 
            
            self.k_t_minus_1_times_s = poison_point

            if not torch.isfinite(poison_point).all():
                raise ValueError("NaN or Inf value detected in poison point.")
            return poison_point
        
        else:
            self.previous_model = [p.data.clone() for p in net.parameters()]
            row_honest_gradients = torch.stack(row_honest_gradients)
            mu = torch.mean(row_honest_gradients, dim=0)
            return mu