import math

import torch
from torch import nn
import torch.nn.functional as F


class ProductGraphResolvent(nn.Module):
    """Observation-conditioned completion on the time-modality graph.

    H has shape [batch, time, modality, hidden]. Only observed features
    contribute to the right-hand side for real utterances.
    """

    def __init__(self, dim):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.log_gamma_m = nn.Parameter(torch.zeros(()))
        self.log_gamma_t = nn.Parameter(torch.zeros(()))
        self.log_beta = nn.Parameter(torch.tensor(math.log(5.0)))

        # Inverse softplus of 1, for unit modality affinity and precision.
        unit = math.log(math.expm1(1.0))
        self.pair = nn.Parameter(torch.full((3, 3), unit))
        self.kappa_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.kappa_head.weight)
        nn.init.constant_(self.kappa_head.bias, unit)

    def modality_affinity(self, H, obs):
        normalized = F.normalize(H, dim=-1)
        cosine = normalized @ normalized.transpose(-1, -2)
        both_observed = obs.unsqueeze(-1) * obs.unsqueeze(-2)
        similarity = torch.exp(self.log_beta.exp() * (cosine - 1.0) * both_observed)
        base = F.softplus((self.pair + self.pair.t()) / 2)
        diagonal = torch.eye(H.size(2), device=H.device, dtype=H.dtype)
        return base * similarity * (1 - diagonal)

    def temporal_affinity(self, H, obs, valid):
        length = H.size(1)

        positions = torch.arange(length, device=H.device, dtype=H.dtype)

        distance = (positions.view(-1, 1) - positions.view(1, -1)).abs()

        position_weight = torch.exp(-distance / self.log_tau.exp()).unsqueeze(0)

        valid_pairs = valid.unsqueeze(-1) * valid.unsqueeze(-2)

        diagonal = torch.eye(length, device=H.device, dtype=H.dtype)

        position_weight = position_weight * valid_pairs * (1 - diagonal)

        normalized = F.normalize(H, dim=-1).transpose(1, 2)

        cosine = (normalized @ normalized.transpose(-1, -2)).permute(0, 2, 3, 1)

        both_observed = obs.unsqueeze(2) * obs.unsqueeze(1)

        similarity = torch.exp(self.log_beta.exp() * (cosine - 1.0) * both_observed)

        return position_weight.unsqueeze(-1) * similarity

    def system(self, H, obs, valid):
        """Build A and source weights r for A X = diag(r) H."""
        batch, length, modalities, _ = H.shape
        nodes = length * modalities
        obs = obs.float() * valid.float().unsqueeze(-1)
        valid = valid.float()
        node_valid = valid.unsqueeze(-1).expand(batch, length, modalities)

        modality = self.modality_affinity(H, obs) * valid.view(batch, length, 1, 1)
        eye_time = torch.eye(length, device=H.device, dtype=H.dtype)
        weights = self.log_gamma_m.exp() * torch.einsum('btuw,ts->btusw', modality, eye_time)
        temporal = self.temporal_affinity(H, obs, valid)
        eye_modality = torch.eye(modalities, device=H.device, dtype=H.dtype)
        weights = weights + self.log_gamma_t.exp() * torch.einsum('btsu,uw->btusw', temporal, eye_modality)
        weights = weights.reshape(batch, nodes, nodes)
        laplacian = torch.diag_embed(weights.sum(-1)) - weights

        kappa = F.softplus(self.kappa_head(H)).squeeze(-1)
        precision = (kappa * obs).reshape(batch, nodes)
        padding = (1 - node_valid).reshape(batch, nodes)
        system = laplacian + torch.diag_embed(precision + padding + 1e-5)
        return system, precision + padding

    def forward(self, H, obs, valid):
        batch, length, modalities, hidden = H.shape
        system, sources = self.system(H, obs, valid)
        rhs = sources.unsqueeze(-1) * H.reshape(batch, length * modalities, hidden)

        chol, info = torch.linalg.cholesky_ex(system)
        if bool((info > 0).any()):
            completed = torch.linalg.solve(system, rhs)
        else:
            completed = torch.cholesky_solve(rhs, chol)
        return completed.reshape(batch, length, modalities, hidden)
