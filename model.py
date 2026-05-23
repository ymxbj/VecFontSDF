import torch
import torch.nn as nn
from torchvision.models import resnet18


class SDFDecoder(nn.Module):
    """Two Linear layers that map an image feature to (N_p * N_a, 6) parabolic
    curve parameters (k, p, q, d, e, f)."""

    def __init__(self, feat_dim: int, hidden_dim: int, v_dim: int, p_dim: int):
        super().__init__()
        self.v_dim = v_dim
        self.p_dim = p_dim
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, v_dim * p_dim * 6)
        for m in (self.fc1, self.fc2):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        y = self.fc1(feat)
        y = self.fc2(y)
        return y.view(-1, self.v_dim * self.p_dim, 6)


class VecFontSDF(nn.Module):
    """End-to-end reconstruction model: RGB image -> ResNet-18 -> SDF decoder ->
    parabolic curve parameters."""

    def __init__(self, feat_dim: int, hidden_dim: int, v_dim: int, p_dim: int):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, feat_dim)
        self.encoder = backbone
        self.decoder = SDFDecoder(feat_dim, hidden_dim, v_dim, p_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))
