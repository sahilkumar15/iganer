"""Runnable CIFT stand-in so the game + ablation execute. Replace for real runs.
identity_gap() is a non-negative pseudo-Delta; real CIFT returns ||g_s - g_t||."""
import torch, torch.nn as nn, torch.nn.functional as F
from .interface import BaseDetector

def _focal(logit,y,g=2.0):
    p=torch.sigmoid(logit); ce=F.binary_cross_entropy_with_logits(logit,y,reduction="none")
    pt=torch.where(y>.5,p,1-p); return ((1-pt)**g*ce).mean()

class StubDetector(BaseDetector):
    def __init__(self, cfg):
        super().__init__(); self._state_dim=256
        self.backbone=nn.Sequential(
            nn.Conv2d(3,32,3,2,1), nn.GroupNorm(8,32), nn.GELU(),
            nn.Conv2d(32,64,3,2,1), nn.GroupNorm(8,64), nn.GELU(),
            nn.Conv2d(64,128,3,2,1), nn.GroupNorm(8,128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128,256), nn.GELU())
        self.cls=nn.Linear(256,1); self.gap=nn.Linear(256,1)
        self.gamma=cfg.losses.focal_gamma
    def extract_state(self,x): return self.backbone(x)
    def logits(self,x): return self.cls(self.backbone(x)).squeeze(-1)
    def identity_gap(self,x): return F.softplus(self.gap(self.backbone(x)).squeeze(-1))
    def compute_loss(self,x,y):
        z=self.backbone(x); logit=self.cls(z).squeeze(-1); yf=y.float()
        gap=F.softplus(self.gap(z).squeeze(-1))
        igs=(yf*F.relu(1-gap)+(1-yf)*gap).mean()
        return _focal(logit,yf,self.gamma)+0.1*igs
