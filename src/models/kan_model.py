"""
Kolmogorov-Arnold Network (KAN) Implementation for PyTorch.
Implements learnable 1D B-spline activation functions on network edges,
inspired by the Kolmogorov-Arnold representation theorem and Efficient-KAN.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BSplineBasis(nn.Module):
    """
    Computes B-Spline basis functions of given order k over a uniform grid.
    """
    def __init__(self, grid_min=-1.0, grid_max=1.0, num_grids=5, spline_order=3):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.spline_order = spline_order
        
        # Extended knot vector
        h = (grid_max - grid_min) / num_grids
        grid = torch.linspace(
            grid_min - spline_order * h,
            grid_max + spline_order * h,
            num_grids + 2 * spline_order + 1,
            dtype=torch.float32
        )
        self.register_buffer("grid", grid)

    def forward(self, x):
        """
        x: (batch_size, in_features)
        Returns: (batch_size, in_features, num_bases)
        """
        # x shape: (B, In, 1)
        x = x.unsqueeze(-1)
        grid = self.grid # (G + 2k + 1)
        
        # 0th-order basis (step functions)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        
        # Cox-de Boor recursive formulation
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[: -(k + 1)])
                / (grid[k:-1] - grid[: -(k + 1)] + 1e-7)
                * bases[:, :, :-1]
            ) + (
                (grid[k + 1 :] - x)
                / (grid[k + 1 :] - grid[1:-k] + 1e-7)
                * bases[:, :, 1:]
            )
        return bases # (B, In, num_grids + spline_order)


class KANLinear(nn.Module):
    """
    A Kolmogorov-Arnold Network Layer where activations live on the edges (weights),
    parameterized by a base linear function + linear combination of B-splines.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_bases = grid_size + spline_order
        
        self.base_activation = base_activation()
        self.b_splines = BSplineBasis(
            grid_min=grid_range[0],
            grid_max=grid_range[1],
            num_grids=grid_size,
            spline_order=spline_order
        )
        
        # Base linear weights: W_base * b(x)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Spline coefficients: (out_features, in_features, num_bases)
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, self.num_bases))
        
        # Learnable scaling parameters
        self.scale_base = nn.Parameter(torch.full((out_features, in_features), scale_base))
        self.scale_spline = nn.Parameter(torch.full((out_features, in_features), scale_spline))
        
        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise=0.1):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_noise)
        with torch.no_grad():
            noise = (torch.rand(self.out_features, self.in_features, self.num_bases) - 0.5) * scale_noise / self.grid_size
            self.spline_weight.copy_(noise)

    def forward(self, x):
        """
        x: (batch_size, in_features)
        Returns: (batch_size, out_features)
        """
        # 1. Base transformation: (B, Out)
        base_output = F.linear(self.base_activation(x), self.base_weight * self.scale_base)
        
        # 2. Spline transformation
        # B-spline basis: (B, In, num_bases)
        spline_basis = self.b_splines(x)
        
        # Spline output: sum over (In, num_bases) -> (B, Out)
        # spline_basis: (B, In, Bases), spline_weight: (Out, In, Bases)
        spline_output = torch.einsum("bik,oik->bo", spline_basis, self.spline_weight * self.scale_spline.unsqueeze(-1))
        
        return base_output + spline_output


class KolmogorovArnoldNetwork(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network for Fishery Abundance / CPUE modeling.
    """
    def __init__(
        self,
        layers_hidden=[22, 16, 8, 1],
        grid_size=5,
        spline_order=3,
        base_activation=nn.SiLU,
        grid_range=(-2.0, 2.0)
    ):
        super().__init__()
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features=in_dim,
                    out_features=out_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_feature_importance(self):
        """
        Computes the L1 norm of incoming spline weights for the first layer.
        Represents relative contribution of each input variable.
        """
        first_layer = self.layers[0]
        # (Out, In, Bases)
        spline_energy = torch.norm(first_layer.spline_weight, p=1, dim=-1) # (Out, In)
        base_energy = torch.abs(first_layer.base_weight) # (Out, In)
        total_energy = (spline_energy + base_energy).sum(dim=0).detach().cpu().numpy() # (In,)
        
        # Normalize to 100%
        importance = (total_energy / (total_energy.sum() + 1e-8)) * 100.0
        return importance

    def evaluate_univariate_splines(self, feature_idx, x_range=(-2.0, 2.0), n_pts=100):
        """
        Evaluates the learned 1D univariate function for a specific input feature in layer 0.
        Used for explainability plots.
        """
        layer = self.layers[0]
        # Determine device from layer parameters
        device = next(layer.parameters()).device
        
        x_vals = torch.linspace(x_range[0], x_range[1], n_pts).unsqueeze(-1).to(device)  # (N, 1)
        
        # Compute base part — all on same device now
        base_out = layer.base_activation(x_vals) * layer.base_weight[:, feature_idx] * layer.scale_base[:, feature_idx]  # (N, Out)
        
        # Compute spline part
        spline_basis = layer.b_splines(x_vals)  # (N, 1, Bases)
        w = layer.spline_weight[:, feature_idx, :] * layer.scale_spline[:, feature_idx].unsqueeze(-1)  # (Out, Bases)
        spline_out = torch.einsum("nbk,ok->no", spline_basis, w)  # (N, Out)
        
        total_out = (base_out + spline_out).mean(dim=1).detach().cpu().numpy()
        x_out = x_vals.squeeze().detach().cpu().numpy()
        return x_out, total_out

if __name__ == "__main__":
    kan = KolmogorovArnoldNetwork(layers_hidden=[22, 16, 8, 1])
    sample_input = torch.randn(32, 22)
    out = kan(sample_input)
    print(f"[SUCCESS] KAN Output shape: {out.shape}")
    imp = kan.get_feature_importance()
    print(f"[SUCCESS] Feature Importance sum: {imp.sum():.2f}%")
