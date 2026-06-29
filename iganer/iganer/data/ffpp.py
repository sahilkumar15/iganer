"""FF++ frame dataset with synthetic fallback.
Layout: <root>/{original,Deepfakes,Face2Face,FaceSwap,NeuralTextures}/<vid>/<frame>.png
FF++ (EULA): https://github.com/ondyari/FaceForensics
Frames:      https://github.com/SCLBD/DeepfakeBench
data.root=null -> SyntheticFFPP (smoke test, no data needed)."""
import glob, os, random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def _tf(sz, train):
    a=[T.Resize((sz,sz))]
    if train: a+=[T.RandomHorizontalFlip()]
    return T.Compose(a+[T.ToTensor()])

class FFPPFrames(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg, self.root = cfg, cfg.data.root
        self.tf=_tf(cfg.data.image_size, split=="train")
        self.samples=self._index()
    def _index(self):
        items, manips = [], ["original"]+list(self.cfg.data.manipulations)
        for m in manips:
            label=0 if m=="original" else 1
            for vd in sorted(glob.glob(os.path.join(self.root,m,"*"))):
                for fp in sorted(glob.glob(os.path.join(vd,"*.png")))[:32]:
                    items.append((fp,label))
        random.shuffle(items); return items
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        fp,label=self.samples[i]
        return self.tf(Image.open(fp).convert("RGB")), label

class SyntheticFFPP(Dataset):
    """Random tensors with a faint injected 'artifact' on fakes so the probe and
    suppression curve have real signal to act on."""
    def __init__(self, cfg, split="train"):
        self.sz=cfg.data.image_size
        self.n=cfg.data.get("synth_n",512) if split=="train" else 160
    def __len__(self): return self.n
    def __getitem__(self,i):
        label=i%2
        x=torch.rand(3,self.sz,self.sz)
        if label==1:                              # high-freq stripe = artifact shortcut
            x[:, ::3, :]=(x[:, ::3, :]+0.06).clamp(0,1)
        return x, label

def build_dataset(cfg, split):
    if not cfg.data.get("root",None): return SyntheticFFPP(cfg, split)
    return FFPPFrames(cfg, split)
