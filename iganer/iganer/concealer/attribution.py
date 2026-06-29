"""Saliency = where the detector currently looks (factor B targeting signal)."""
import torch
def saliency_mask(detector, x, thresh=0.5):
    x=x.clone().requires_grad_(True)
    g,=torch.autograd.grad(detector.fake_prob(x).sum(), x)
    m=g.abs().mean(1,keepdim=True)
    m=(m-m.amin(dim=(2,3),keepdim=True))/(m.amax(dim=(2,3),keepdim=True)-m.amin(dim=(2,3),keepdim=True)+1e-6)
    import kornia.filters as KF
    m=KF.gaussian_blur2d(m,(9,9),(3.,3.))
    return (m>thresh).float()*m
