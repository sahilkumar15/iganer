"""Frozen artifact-only probe (HF residual classifier). Verifies concealment;
never receives game gradients."""
import torch, torch.nn as nn
import kornia.filters as KF
class ArtifactProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,2,1), nn.GroupNorm(8,32), nn.GELU(),
            nn.Conv2d(32,64,3,2,1), nn.GroupNorm(8,64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64,1))
    def _resid(self,x): return (x-KF.gaussian_blur2d(x,(7,7),(3.,3.))).clamp(-1,1)
    def logits(self,x): return self.net(self._resid(x)).squeeze(-1)
    def fake_prob(self,x): return torch.sigmoid(self.logits(x))
    def freeze(self):
        self.eval()
        for p in self.parameters(): p.requires_grad_(False)
        return self
