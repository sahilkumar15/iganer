"""Detector contract used by the game and ablation. CIFT plugs in here."""
from abc import ABC, abstractmethod
import torch, torch.nn as nn

class BaseDetector(nn.Module, ABC):
    @abstractmethod
    def logits(self, x): ...
    @abstractmethod
    def extract_state(self, x): ...
    @abstractmethod
    def identity_gap(self, x): ...
    @abstractmethod
    def compute_loss(self, x, y): ...
    def fake_prob(self, x): return torch.sigmoid(self.logits(x))
    @property
    def state_dim(self): return self._state_dim

def build_detector(cfg, device):
    if cfg.detector.type=="stub":
        from .stub import StubDetector
        return StubDetector(cfg).to(device)
    if cfg.detector.type=="cift":
        # === CIFT INTEGRATION ===========================================
        # from cift.models import CIFTModel
        # from .cift_adapter import CIFTAdapter
        # m = CIFTAdapter(CIFTModel(...), cfg)
        # if cfg.detector.ckpt: m.load_cift(cfg.detector.ckpt)
        # return m.to(device)
        # ================================================================
        raise NotImplementedError("Wire CIFTAdapter, then set detector.type=cift.")
    raise ValueError(cfg.detector.type)
