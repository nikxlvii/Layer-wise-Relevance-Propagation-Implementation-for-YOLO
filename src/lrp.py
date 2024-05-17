from copy import deepcopy

import torch
from torch import nn

from src.utils import layers_lookup

class LRPModel(nn.Module):

    def __init__(self,model:torch.nn.Module,top_k:float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.top_k = top_k

        self.model.eval()

        self.layers = self._get_layer_operations()

        self.lrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:

        layers = deepcopy(self.layers)
        lookup_table = layers_lookup()


        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = lookup_table[layer.__class__](layer=layer,top_k = self.top_k)
            except KeyError:
                message = (
                    f"Layer-wise Relevance Propagation not implemented for "
                    f"{layer.__class__.__name__} layer."
                )
                raise NotImplementedError(message)
        
        return layers 
    
    def _get_layer_operations(self) -> torch.nn.ModuleList:

        layers = torch.nn.ModuleList()

        for layer in self.model.features:
            layers.append(layer)

        layers.append(self.model.avgpool)
        layers.append(torch.nn.Flatten(start_dim=1))


        for layer in self.model.classifier:
            layers.append(layer)
        
        return layers 
    
    def forward(self,x:torch.tensor) -> torch.tensor:
        activations = list()

        with torch.no_grad():
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)

        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        relevance = torch.softmax(activations.pop(0), dim=-1)

        for i, layer in enumerate(self.lrp_layers):
            relevance = layer.forward(activations.pop(0),relevance)
        
        return relevance.permute(0,2,3,1).sum(dim=-1).squeeze().detach().cpu()
