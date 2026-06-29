"""FIFO replay of concealed fakes so the detector doesn't forget old attacks."""
import random, torch
class ConcealReplay:
    def __init__(self, cap=2048): self.cap=cap; self.buf=[]
    def add(self, x_til):
        for img in x_til.detach().cpu():
            self.buf.append(img)
            if len(self.buf)>self.cap: self.buf.pop(0)
    def sample(self, n, device):
        if len(self.buf)<n: return None
        idx=random.sample(range(len(self.buf)),n)
        return torch.stack([self.buf[i] for i in idx]).to(device)
