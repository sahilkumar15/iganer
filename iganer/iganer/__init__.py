"""IGANER: Identity-Gap Adversarial Nash Equilibrium via Reinforcement learning
for rendering-invariant deepfake detection.

Three factors, independently toggleable for the factorial ablation:
  A = Nash game     (concealer is a learned PPO adversary vs random policy)
  B = targeting     (suppression masked to detector saliency vs uniform)
  C = protection    (identity-gap preservation on concealer reward + detector)
"""
__version__ = "0.1.0"
