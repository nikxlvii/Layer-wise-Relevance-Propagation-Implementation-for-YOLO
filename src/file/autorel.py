import torch
import time 

def lrp_v1(
        layer: torch.nn.Linear, a: torch.tensor, r: torch.tensor, eps: float = 1e-5
) -> torch.tensor: 
    z = layer.forward(a) + eps 
    s = (r / z).data
    (z * s).sum().backward()
    c = a.grad
    r = (a * c).data
    return r


def lrp_v2(
        layer: torch.nn.Linear, a: torch.tensor, r: torch.tensor, eps: float = 1e-5
) -> torch.tensor:
    w = layer.weight
    b = layer.bias
    z = torch.mm(a,w.T) + b + eps
    s = r / z
    c = torch.mm(s , w)
    r = (a * c).data
    return r

def lrp_v3(
        layer: torch.nn.Linear, a: torch.tensor, r: torch.tensor, eps: float = 1e-5
) -> torch.tensor:
    z = layer.forward(a) + eps
    s = r / z 
    c = torch.mm(s, layer.weight)
    r = (a * c).data
    return r


def main():
    torch.manual_seed(69)

    batch_size = 16
    # still incomplete