"""EMA detector = Concealer's reward target (slows the moving target)."""
import copy, torch
class EMADetector:
    def __init__(self, detector, decay=0.99):
        self.ema=copy.deepcopy(detector).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay=decay
    @torch.no_grad()
    def update(self, detector):
        for e,p in zip(self.ema.parameters(), detector.parameters()):
            e.mul_(self.decay).add_(p, alpha=1-self.decay)
        for e,p in zip(self.ema.buffers(), detector.buffers()): e.copy_(p)
    def __getattr__(self, name): return getattr(self.__dict__["ema"], name)
