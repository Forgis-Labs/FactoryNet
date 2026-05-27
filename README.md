# FactoryNet

A large-scale, openly available industrial time-series dataset for training foundation models on manufacturing processes. FactoryNet spans 23,000+ episodes across 6 robotic embodiments, covering 28 anomaly types and 3 industrial tasks.

**Dataset**: [huggingface.co/datasets/Forgis/FactoryNet](https://huggingface.co/datasets/Forgis/FactoryNet)  
**Paper**: [arXiv:2605.09081](https://arxiv.org/abs/2605.09081)
**License**: MIT (lab + synthetic data), original licenses for open-source subsets

## Dataset Overview

| Category | Episodes | Source |
|----------|----------|--------|
| Real fault-injected lab recordings | ~8,340 | UR3e (KUKA KR10 forthcoming) |
| Unified open-source episodes | ~4,500 | voraus-AD (Yu-Cobot), AURSAD (UR3e), UMich CNC |
| Synthetic (Isaac Sim) | ~10,000 | Simulated UR5 |
| **Total** | **23,000+** | **6 embodiments** |

**Tasks**: Pick and Place, Screwdriving, Peg-in-Hole  
**Anomaly types**: 28 fault categories across all tasks

## S-E-F-C Schema

FactoryNet organizes all signals under a machine-agnostic, control-theoretic decomposition:

- **Setpoint (S)** - Commanded references (target positions, velocities)
- **Effort (E)** - Applied forces and torques
- **Feedback (F)** - Measured sensor readings (actual positions, velocities)
- **Context (C)** - Environmental and metadata signals (timestamps, labels, task phase)

This schema is grounded in a hierarchical backbone following the IEC 81346 industrial standard, enabling consistent cross-machine representation regardless of manufacturer or protocol.

## Supported Tasks

FactoryNet is designed for research in:

- Anomaly detection and fault diagnosis
- Counterfactual reasoning over process trajectories
- Cross-machine transfer learning
- Process optimization and baseline modeling
- Foundation model pretraining on industrial time-series

## Repository Structure

```
FactoryNet/
├── core/                    # Data pipeline utilities, adapters, feature extraction
├── scripts/                 # Dataset processing and download scripts
└── taxonomy/                # S-E-F-C signal taxonomy definitions (YAML)
```

## Quick Start

### Installation

```bash
git clone https://github.com/Forgis-Labs/factorynet.git
cd factorynet

uv venv
uv pip install -r requirements.txt
```

### Load the Dataset

```python
from datasets import load_dataset

ds = load_dataset("forgis/factorynet")
```

## Citation

If you use FactoryNet in your research, please cite:

```bibtex
@article{othman2026factorynet,
  title={FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models},
  author={Karim Othman and Jonas Petersen and Matei Ignuta-Ciuncanu and Camilla Mazzoleni and Federico Martelli and Alessandro Lombardi and Riccardo Maggioni and Philipp Petersen},
  journal={arXiv preprint arXiv:2605.09081},
  year={2026}
}
```

## License

MIT license for lab recordings and synthetic data. Open-source subsets retain their original licenses. See individual dataset cards for details.
