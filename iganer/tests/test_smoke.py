"""End-to-end smoke: probe -> ablation (2 cells) -> Table A, on synthetic data."""
import os, sys, subprocess
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _run(c):
    r=subprocess.run(c,cwd=ROOT,env=dict(os.environ,PYTHONPATH=ROOT),capture_output=True,text=True)
    assert r.returncode==0, r.stdout+r.stderr; return r.stdout
def test_smoke():
    opts=["--config","configs/default.yaml","--experiment","configs/experiment/iganer_ffpp.yaml"]
    ov=["data.root=null","wandb.enabled=false","reward.use_lpips=false","device=cpu",
        "data.num_workers=0","train.batch_size=8","data.synth_n=64"]
    _run([sys.executable,"scripts/train_probe.py",*opts,*ov,"probe.pretrain_epochs=1"])
    out=_run([sys.executable,"scripts/run_ablation_tableA.py",*opts,"--only","1,8",
              *ov,"train.epochs=1","game.total_epochs=1","eval.curve_points=3"])
    assert "TABLE A" in out and "v8_iganer" in out
