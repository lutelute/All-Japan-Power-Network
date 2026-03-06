"""Network reconstruction pipeline for the All-Japan Grid model.

Provides configurable strategies for handling isolated network elements:

- **Simplification mode**: Remove isolated substations, transmission lines,
  and generators to produce a clean, connected network suitable for power
  flow analysis.

- **Reconnection mode**: Generate synthetic connection points for isolated
  elements and regenerate the Ybus admittance matrix.

Usage::

    from src.reconstruction.config import ReconstructionConfig
    from src.reconstruction.pipeline import ReconstructionPipeline

    cfg = ReconstructionConfig.from_yaml("config/reconstruction.yaml")
    pipeline = ReconstructionPipeline(cfg)
    result = pipeline.run(net, region="shikoku")
"""
