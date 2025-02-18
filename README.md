<p align="center">
    <a href= "https://pypi.org/project/rewarped/">
    <img src="https://img.shields.io/pypi/v/rewarped" /></a>
    <a href= "https://arxiv.org/abs/2412.12089">
    <img src="https://img.shields.io/badge/arxiv-2412.12089-b31b1b" /></a>
</p>

# Rewarped

Rewarped 🌀 is a platform for reinforcement learning in parallel differentiable multiphysics simulation, built on [`NVIDIA/warp`](https://github.com/NVIDIA/warp). Rewarped supports:

- **parallel environments**: to train RL agents at scale.
- **differentiable simulation**: to compute batched analytic gradients for optimization.
- **multiphysics**: (CRBA, FEM, MPM, XPBD) physics solvers and coupling to support interaction between rigid bodies, articulations, and various deformables.

We use Rewarped to *re-implement* a variety of RL tasks from prior works, and demonstrate that first-order model-based RL algorithms (which use differentiable simulation to compute analytic policy gradients) can be scaled to a range of challenging manipulation and locomotion tasks that involve interaction between rigid bodies, articulations, and deformables.

> For control and reinforcement learning algorithms, see [`etaoxing/mineral`](https://github.com/etaoxing/mineral).

# Contributing

Contributions are welcome! Please refer to [`CONTRIBUTING.md`](CONTRIBUTING.md).

# Citing

```bibtex
@inproceedings{xing2025stabilizing,
  title={Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation},
  author={Xing, Eliot and Luk, Vernon and Oh, Jean},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=DRiLWb8bJg}
}
```

# Acknowledgements

Differentiable Simulation
- [`NVIDIA/warp`](https://github.com/NVIDIA/warp)
- [`NVlabs/DiffRL`](https://github.com/NVlabs/DiffRL)
- MPM
  - [`PingchuanMa/SGA`](https://github.com/PingchuanMa/SGA), [`PingchuanMa/NCLaw`](https://github.com/PingchuanMa/NCLaw)
  - [`sizhe-li/DexDeform`](https://github.com/sizhe-li/DexDeform)
  - [`hzaskywalker/PlasticineLab`](https://github.com/hzaskywalker/PlasticineLab)

RL Tasks (alphabetical)
- [`sizhe-li/DexDeform`](https://github.com/sizhe-li/DexDeform)
- [`NVlabs/DiffRL`](https://github.com/NVlabs/DiffRL)
- [`hzaskywalker/PlasticineLab`](https://github.com/hzaskywalker/PlasticineLab)
- [`gradsim/gradsim`](https://github.com/gradsim/gradsim)
- [`isaac-sim/IsaacGymEnvs`](https://github.com/isaac-sim/IsaacGymEnvs)
- [`leggedrobotics/legged_gym`](https://github.com/leggedrobotics/legged_gym)
- [`Xingyu-Lin/softgym`](https://github.com/Xingyu-Lin/softgym)
- ...
