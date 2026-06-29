import torch.nn.functional as F
def focal_bce(logit,y,gamma=2.0):
    p=logit.sigmoid(); ce=F.binary_cross_entropy_with_logits(logit,y,reduction="none")
    pt=y*p+(1-y)*(1-p); return ((1-pt)**gamma*ce).mean()
