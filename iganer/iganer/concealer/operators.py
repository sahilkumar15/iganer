"""Evidence-suppression operators (the action space). Optionally masked to the
detector's saliency region (factor B)."""
import torch, torch.nn.functional as F
import kornia.filters as KF
def _blur(x,s):
    k=int(2*round(1+3*s))+1; return KF.gaussian_blur2d(x,(k,k),(0.3+2.5*s,)*2)
def _fft_lowpass(x,s):
    B,C,H,W=x.shape
    X=torch.fft.fftshift(torch.fft.fft2(x),dim=(-2,-1))
    yy,xx=torch.meshgrid(torch.arange(H,device=x.device),torch.arange(W,device=x.device),indexing="ij")
    r=torch.sqrt(((yy-H/2)**2+(xx-W/2)**2)); keep=(0.5-0.4*s)*min(H,W)/2
    m=(r<=keep).float()[None,None]
    return torch.fft.ifft2(torch.fft.ifftshift(X*m,dim=(-2,-1))).real.clamp(0,1)
def _jpeg(x,s):
    H,W=x.shape[-2:]; f=1-0.5*s; h2,w2=max(8,int(H*f)),max(8,int(W*f))
    d=F.interpolate(x,(h2,w2),mode="bilinear",align_corners=False)
    return F.interpolate(d,(H,W),mode="bilinear",align_corners=False)
def _bilateral(x,s): return KF.bilateral_blur(x,(5,5),0.1+0.4*s,(1.5,1.5))
OPERATORS={"blur":_blur,"fft_lowpass":_fft_lowpass,"jpeg":_jpeg,"bilateral":_bilateral}

class ConcealBank:
    def __init__(self, names, levels=4):
        self.ops=[OPERATORS[n] for n in names]; self.names=list(names)
        self.L=levels; self.n_actions=len(self.ops)*levels
    def decode(self,a):
        oi,lvl=divmod(int(a),self.L); return oi,(lvl+1)/self.L
    def apply(self, x1, a, mask=None, cap=1.0):
        oi,s=self.decode(a); s=min(s,cap)
        edited=self.ops[oi](x1.unsqueeze(0),s).squeeze(0).clamp(0,1)
        if mask is not None: edited=mask*edited+(1-mask)*x1     # factor B: targeted
        return edited
    def apply_uniform(self, x, oi, s):                          # for the curve eval
        return self.ops[oi](x,s).clamp(0,1)
