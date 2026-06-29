"""Minimal PPO for the Concealer (horizon-1 contextual game)."""
import torch, torch.nn.functional as F
class PPO:
    def __init__(self, policy, cfg, device):
        self.p=policy; self.opt=torch.optim.Adam(policy.parameters(),lr=cfg.concealer.lr)
        self.clip=cfg.concealer.clip; self.ent=cfg.concealer.entropy_coef
        self.vc=cfg.concealer.value_coef; self.epochs=cfg.concealer.ppo_epochs
    def update(self, state, actions, old_logp, rewards):
        state,actions=state.detach(),actions.detach()
        old_logp,rewards=old_logp.detach(),rewards.detach()
        if rewards.numel()>1: rewards=(rewards-rewards.mean())/(rewards.std()+1e-6)
        st={}
        for _ in range(self.epochs):
            logp,ent,v=self.p.evaluate(state,actions); adv=(rewards-v).detach()
            ratio=torch.exp(logp-old_logp)
            s1=ratio*adv; s2=torch.clamp(ratio,1-self.clip,1+self.clip)*adv
            pi=-torch.min(s1,s2).mean(); vl=F.mse_loss(v,rewards); e=ent.mean()
            loss=pi+self.vc*vl-self.ent*e
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.p.parameters(),0.5); self.opt.step()
            st=dict(pi_loss=pi.item(), v_loss=vl.item(), entropy=e.item())
        return st
