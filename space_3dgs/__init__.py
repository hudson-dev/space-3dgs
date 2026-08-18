"""space-3dgs: Gaussian splatting for multi-sequence captures of station interiors.

nerfstudio components:
  - SequenceNerfstudio      dataparser that tags every camera with its capture-sequence ID
  - SequenceAppearanceModel splatfacto + one achromatic exposure code per sequence
  - CullAfterRefineStrategy gsplat strategy that keeps pruning after densification stops
  - method_config.space_3dgs the registered `space-3dgs` training recipe
"""
