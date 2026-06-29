import math
class Curriculum:
    def __init__(self, cfg): self.total=cfg.game.total_epochs
    def strength_cap(self, e):
        t=min(max(e/max(self.total,1),0),1); return 0.2+0.8*0.5*(1-math.cos(math.pi*t))
    def explore_coef(self, e):
        t=e/max(self.total,1); return math.exp(-((t-0.4)**2)/0.08)
