import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import time
import matplotlib.pyplot as plt

def get_max_distance(point, c_in, r_in, c_out):
    dtype_in = point.dtype
    point = point.to(torch.float64)
    c_in  = c_in.to(torch.float64)
    c_out = c_out.to(torch.float64)
    r_in  = r_in.to(torch.float64)
    
    point_c_in = point - c_in

    c_out_point = c_out - point

    dot = torch.dot(point_c_in, c_out_point)         
    norm_c_out_point = torch.norm(c_out_point) 
    norm_point_c_in = torch.norm(point_c_in) 

    inside = dot**2 - norm_c_out_point**2 * (norm_point_c_in**2 - r_in**2)
  
    sqrt = torch.sqrt(inside)

    return (-dot + sqrt) / norm_c_out_point


def get_min_in_max_distances(point, c_ins, r_ins, c_out):
    min_max_distance = float("inf")
    for c_in, r_in in zip(c_ins, r_ins):
        d = get_max_distance(point, c_in, r_in, c_out).item()
        if d < min_max_distance:
            min_max_distance = d
    return min_max_distance


def in_or_not(point, c_ins, r_ins):
    for center, radius in zip(c_ins, r_ins):
        if torch.norm(point - center) >= radius:
            return False
    return True

    
def final_step(point, c_ins, r_ins, direction, mu):
    projection_size = torch.dot(point,  direction)/direction.norm()
    projection = projection_size * direction/direction.norm()

    is_in = in_or_not(projection, c_ins, r_ins)

    if is_in:
        target = direction*10**9 # same as projection + direction*10**9
        distance = max(get_min_in_max_distances(projection, c_ins, r_ins, target) - 1e-3, 0)
        final_point = projection + direction/direction.norm() * distance
    else:
        target = point + direction*10**9
        distance = max(get_min_in_max_distances(point, c_ins, r_ins, target) - 1e-3, 0)
        point_bis = point + direction/direction.norm() * distance

        projection_size = torch.dot(point_bis, direction)/direction.norm()
        projection = projection_size * direction/direction.norm()

        distance = max(get_min_in_max_distances(point, c_ins, r_ins, projection) - 1e-3, 0)
        new_direction = projection - point
        final_point = point + new_direction/new_direction.norm() * distance

    return final_point

def one_step(point, c_ins, r_ins, c_out, r_out):
    # get dOutMin dOutMax
    d_out_min, d_out_max = get_out_distances(point, c_out, r_out)
    
    # if already in B(c_out, r_out), return point
    if d_out_min <= 0:
        return point
        
    d_in_max = get_min_in_max_distances(point, c_ins, r_ins, c_out)

    # if it is not possible to reach the new ball return None
    if d_in_max < d_out_min:
        return None
  
    dist = (d_out_min + min(d_out_max, d_in_max)) / 2
    
    d = get_direction(point, c_out)
       
    return point + d * dist


def get_out_distances(point, c_out, r_out):
    d = torch.norm(point - c_out)
    d_out_min = d - r_out
    d_out_max = d + r_out
    return d_out_min, d_out_max


def get_direction(from_point, to_point):
    v = to_point - from_point
    norm_v = torch.norm(v)
    return v / norm_v


def fall_in_intersection(centers, radii, direction, mu):
    point = centers[0] + .5 * get_direction(centers[0], centers[1]) * radii[0]
    
    c_ins = [centers[0]]
    r_ins = [radii[0]]
    
    s = len(centers)
    
    for i in range(1, s):
        c_ins_tensor = torch.stack(c_ins)
        r_ins_tensor = torch.stack(r_ins)
        
        point = one_step(point, c_ins_tensor, r_ins_tensor, centers[i], radii[i])

        if point is None:
            return None
        
        c_ins.append(centers[i])
        r_ins.append(radii[i])
    
    c_ins_tensor = torch.stack(c_ins)
    r_ins_tensor = torch.stack(r_ins)

    
    final_point = final_step(point, c_ins_tensor, r_ins_tensor, direction, mu)

    for i in range(s):
        if torch.norm(final_point - centers[i]) >= radii[i]:
            raise ValueError("The final point is outside the intersection.")

    return final_point

    
def get_score(X, ref_vector, f, contaminability_matrix, dist_matrix):
    ref_vector = ref_vector.unsqueeze(0)
    
    cos = F.cosine_similarity(X, ref_vector, dim=1)
    
    norm = torch.norm(X, dim = 1)

    return  cos * norm

class NearestNeighborPoisoning:
    def __init__(self, f, n, robust_aggregator, net):
        self.n = n
        self.f = f
        self.h = n - f
        self.s = int(n / 2 + 1) - f
        self.model_size = sum(p.numel() for p in net.parameters())
        self.device = next(net.parameters()).device
        self.sign = torch.randint(0, 2, (self.model_size,), device=self.device).float().mul_(2).sub_(1)
        self.robust_outputs = 0
        self.robust_aggregator = robust_aggregator
        self.outputs = 0
        self.old_output = 0
        self.max_norm = 0
        self.direction = None
        self.old_mu = None
        self.sign = None
        self.path = 0
        self.first = True
        
    def compute_distance_rank_contaminability(self, points, direction):
        dist_matrix = torch.cdist(points, points)
        rank_matrix = dist_matrix.argsort(dim=1).argsort(dim=1)
        contaminability_matrix = self.h - rank_matrix
        
        score = get_score(points, direction, self.f, contaminability_matrix, dist_matrix)
        
        return dist_matrix, contaminability_matrix, score

    def get_poison_point(self, points, indexes, contaminability_matrix, dist_matrix, direction, mu):
        f_idx = (contaminability_matrix[indexes] == self.f).nonzero(as_tuple=False)[:,1]
        
        radii = dist_matrix[indexes, f_idx]
        
        selected_points = points[indexes]

        poison_point = fall_in_intersection(selected_points, radii, direction, mu)
    
        return poison_point

    
    def __call__(self, points_, best_best_poison_point=None):
        with torch.no_grad():
            points = torch.stack(points_)

            mu = torch.mean(points, dim=0)
            
            if self.first:
                poison_point = mu
                
                self.first = False
            
            else:                
                centered_points = points - mu
                
                direction = -(self.path + mu)
                
                dist_matrix, contaminability_matrix, score = self.compute_distance_rank_contaminability(centered_points, direction)
                
                _, indexes = torch.topk(score, k=self.s)
    
                poison_point = self.get_poison_point(centered_points, indexes, contaminability_matrix, dist_matrix, direction, mu)
                
                poison_point += mu
                    
            if not torch.isfinite(poison_point).all():
                    raise ValueError("NaN or Inf value detected in poison point.")

            inputs = points_ + [poison_point for _ in range(self.f)] 
            
            output = self.robust_aggregator(inputs)

            self.path += output
            
            return poison_point