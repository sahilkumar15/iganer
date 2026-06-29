"""CIFTAdapter — wraps the real CIFT model (cldm.diffusionfake.DiffusionFake,
ControlNet/Lightning) behind IGANER's BaseDetector interface.

This is the ONLY file that knows CIFT internals. It maps:
  - logits(x)        -> CIFT GuideNet classification head (control_model.fc)
  - extract_state(x) -> pooled 1792-d feature filter features (policy state)
  - identity_gap(x)  -> Δ readout from DualIdentityMambaFusion (XID/IGS core)
  - compute_loss(x,y)-> CIFT focal+BCE (+ gap term), via the model's own _focal_bce

It is designed to live INSIDE the CIFT repo (sahilkumar15/ImageDifussionFake),
where `cldm`, `share`, `models` are importable. Build it from a model created
by `cldm.model.create_model(...)` exactly as train.py does.

CIFT image convention is [-1,1] BCHW; IGANER passes [0,1] BCHW, so we convert.

NOTE ON SOURCE-FREE INFERENCE: at test time CIFT is donor-free — the GuideNet
runs the EfficientNet feature filter on the target image and the classifier
head produces the logit, with the diffusion branch unused. We replicate that
path directly (no SD sampling), which is what makes IGANER's RL training-only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .interface import BaseDetector

# CIFT forgery-type tokens (cldm/mamba_modules.py)
FTYPE_SWAP = "swap"
FTYPE_REENACT = "reenact"
FTYPE_GENUINE = "genuine"


def _to_cift_space(x01):
    """IGANER [0,1] BCHW -> CIFT [-1,1] BCHW."""
    return (x01.clamp(0, 1) * 2.0) - 1.0


class CIFTAdapter(BaseDetector):
    def __init__(self, cift_model, cfg):
        """cift_model: a DiffusionFake LightningModule already built via
        create_model(...) with control_model.define_feature_filter(backbone)
        called (as in train.py). cfg: the IGANER OmegaConf config."""
        super().__init__()
        self.m = cift_model               # DiffusionFake (LatentDiffusion)
        self.gn = cift_model.control_model  # GuideNet (holds feature filter + mamba head)
        self._state_dim = 1792            # pooled GuideNet feature width (self.gn.fc in_features)
        self.gap_weight = float(getattr(cfg.reward, "delta_protect", 0.2))

    # ---- core GuideNet feature path (source-free, diffusion unused) ----
    def _features(self, x01):
        """Pooled 1792-d features from CIFT's EfficientNet feature filter on the
        TARGET image. Mirrors GuideNet._preprocess_img + encoder + global_pool."""
        x = _to_cift_space(x01)
        img = self.gn._preprocess_img(x)                 # ImageNet-norm, resize_hw
        feat = self.gn.encoder(img)
        feat = feat[-1] if isinstance(feat, (list, tuple)) else feat
        feat = self.gn.encoder_proj(feat)                # -> 1792 channels
        pooled = self.gn.global_pool(feat).flatten(1)    # [B, 1792]
        return pooled

    def extract_state(self, x01):
        return self._features(x01)

    def logits(self, x01):
        # CIFT classifier head on pooled target features
        return self.gn.fc(self._features(x01)).squeeze(-1)

    def identity_gap(self, x01):
        """Source-free Δ readout. CIFT's DualIdentityMambaFusion projects the
        target features into the relational space; in donor-free mode the
        privileged stream is unavailable, so Δ is read as the magnitude of the
        target identity-gap embedding (proxy for ||g_s - g_t|| the head was
        trained to separate). Real CIFT exposes this on the mamba head.
        === CONFIRM: if your mamba_head has a dedicated `gap_readout(feat)` or
        returns `delta`, call it here instead of this norm proxy. ==="""
        feat = self._features(x01)
        head = getattr(self.gn, "mamba_head", None)
        if head is not None and hasattr(head, "dual_mamba"):
            dm = head.dual_mamba                          # DualIdentityMambaFusion
            g_t = dm.norm_t(dm.proj_t(feat))              # target identity embedding
            return g_t.norm(dim=-1)                       # ||g_t|| as donor-free Δ proxy
        # fallback: a small linear gap head if present
        return F.softplus(self.gn.fc(feat).squeeze(-1)).abs()

    def compute_loss(self, x01, y, forgery_type=None):
        """CIFT focal+BCE classification loss (+ light gap term). Reuses the
        model's own _focal_bce so weighting matches the CIFT paper."""
        logit = self.logits(x01)
        yf = y.float().view(-1)
        logit = torch.clamp(logit.view(-1), -20.0, 20.0)
        if hasattr(self.m, "_focal_bce"):
            bce_main, focal_main = self.m._focal_bce(logit, yf)
            cls = 0.6 * bce_main + 0.4 * focal_main
        else:
            cls = F.binary_cross_entropy_with_logits(logit, yf)
        # identity-gap regularizer (keep Δ high for fakes, low for reals)
        gap = self.identity_gap(x01)
        gap_term = (yf * F.relu(1.0 - gap) + (1 - yf) * gap).mean()
        return cls + self.gap_weight * gap_term

    def load_cift(self, ckpt_path, map_location="cpu"):
        sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = sd.get("state_dict", sd)
        self.m.load_state_dict(sd, strict=False)
        return self
