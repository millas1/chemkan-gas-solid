# Modeling Gas-Solid Phase Chemical Kinetics with ChemKANs

**Sara Milla** — TU Wien, Interdisciplinary Project in Data Science
Supervisor: Ass. Prof. DI Dr. Markus Bösenhofer

This project adapts ChemKAN to gas-solid reactions (carbon combustion
with Thiele Based Models, iron ore reduction with Layer Models).
It builds on Berke Bankoglu's PyTorch reproduction of ChemKAN,
which this repository is forked from.

- My work: [`gas_solid/`](gas_solid/) and [`docs/gas_solid/`](docs/gas_solid/)
- Project log: [`docs/gas_solid/project_log.md`](docs/gas_solid/project_log.md)
- Original reproduction documentation: see below.

---
# ChemKAN — Reproduction and Extension

A PyTorch reproduction and extension of **ChemKAN** (*ChemKANs for combustion chemistry modeling and acceleration*, Koenig, Kim & Deng, *Physical Chemistry Chemical Physics*, 2025). ChemKAN represents chemical reaction dynamics with a physics-informed Kolmogorov–Arnold network (KAN) inside an ordinary differential equation (ODE) solver.

This repository studies **biodiesel model inference**, **hydrogen–air combustion**, and a **methane–air combustion extension**. It also includes a DeepONet baseline, forward sensitivity analysis (FSA), recorded experiments, and notebooks that explain **how to build and train the model from scratch**.

| Project information | Details |
|---|---|
| Institution | TU Wien — MSc Data Science |
| Course | [194.147 Interdisciplinary Project in Data Science](https://tiss.tuwien.ac.at/course/courseDetails.xhtml?semester=2026S&courseNr=194147) |
| Semester | Summer semester 2026 |
| Student | Hasan Berke Bankoglu |
| Main supervisor | Ass. Prof. DI Dr. Markus Bösenhofer |
| Co-supervisor | Univ.-Prof. Dr. Thomas Gärtner |
| Original paper | [Koenig, Kim & Deng (2025)](https://doi.org/10.1039/D5CP02009C) · [PDF](docs/paper/d5cp02009c.pdf) |

> **Scope.** This is a reproduction study, not an official implementation by the paper's authors. Some qualitative behaviors reproduce well, but several quantitative claims differ. The [reproduction summary](docs/reproduction_summary.md) records both agreement and discrepancies.

## Contents

- [What is included](#what-is-included)
- [Where to start](#where-to-start)
- [Installation](#installation)
- [Learn ChemKAN from scratch](#learn-chemkan-from-scratch)
- [Quick start](#quick-start)
- [Training and experiment settings](#training-and-experiment-settings)
- [Reproduce the recorded experiments](#reproduce-the-recorded-experiments)
- [Evaluation, figures and interpretability](#evaluation-figures-and-interpretability)
- [Main results](#main-results)
- [Use your own data](#use-your-own-data)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Repository layout](#repository-layout)
- [Documentation and references](#documentation-and-references)

## What is included

| Component | System or purpose | Status |
|---|---|---|
| **Biodiesel** — paper Section 3.1 | Six species; kinetic-only ChemKAN, robustness to noise (Figs. 3, 5, 6), neural scaling (Fig. 4) | 13 recorded ChemKAN figure runs; results partially reproduce paper claims |
| **Hydrogen** — paper Section 3.2 | Nine species + temperature; two-stage combustion training; Figs. 7–8 | 344-parameter model; Stage 2 trained to 200,000 epochs |
| **Methane** — project extension | GRI-Mech 3.0; 52 species + temperature | Stage 2 trained to 200,000 epochs; **180,000-epoch checkpoint selected**, model not converged |
| **DeepONet** | Independent PyTorch biodiesel baseline for the paper's comparison | 14 recorded figure runs |
| **Forward sensitivity analysis (FSA)** | Forward state-parameter sensitivities for neural-ODE gradients | Implemented, tested and explained; used by the recorded biodiesel ChemKAN runs |
| **Inference benchmarks** | ChemKAN PyTorch/Julia vs. Cantera and Arrhenius.jl | Recorded for biodiesel, hydrogen and methane |
| **Learning material** | From-scratch KAN/ChemKAN notebooks, source-code walkthrough, FSA and data generation | Runnable notebooks and a companion learning guide |

**Training backend matters:** the recorded **biodiesel** ChemKAN figure runs use **FSA**, whereas the recorded **hydrogen and methane** runs use **direct autograd**. The availability of FSA in the code does not mean every recorded experiment used it.

## Where to start

Choose the route that matches your goal:

| Goal | Start here | What you will learn or do |
|---|---|---|
| **Understand and build ChemKAN step by step** | [`extra/03_chemkan_from_scratch_paper_aligned.ipynb`](chemkan/notebooks/extra/03_chemkan_from_scratch_paper_aligned.ipynb) | Construct the RBF edge functions, AddKAN/LeanKAN layers and biodiesel kinetic core, then train it as a neural ODE |
| **Practice implementing a KAN yourself** | [`extra/01_implement_kan.ipynb`](chemkan/notebooks/extra/01_implement_kan.ipynb) + [answer/learning guide](docs/implementing_chemkan_from_scratch_guide.md) | Guided exercises, milestones and reference explanations |
| **Understand the actual codebase** | [`extra/04_chemkan_actual_solution_walkthrough.ipynb`](chemkan/notebooks/extra/04_chemkan_actual_solution_walkthrough.ipynb) | Trace state preparation → normalization → KAN → ODE solve → loss → optimizer |
| **Understand the edge-function math** | [`extra/02_einsum_edge_functions_paper_aligned.ipynb`](chemkan/notebooks/extra/02_einsum_edge_functions_paper_aligned.ipynb) | Connect the paper's indexed functions with `torch.einsum` and equivalent implementations |
| **Read a complete, small training example** | [`chemkan/scripts/train.py`](chemkan/scripts/train.py) | Kinetic-only training from data loading to saved checkpoint, without a large training framework |
| **Understand FSA** | [`forward_sensitivity_analysis.ipynb`](chemkan/notebooks/forward_sensitivity_analysis.ipynb) | Sensitivity equations, analytic tests, gradient comparisons, solver tolerances and implementation |
| **Inspect recorded experiments** | [Reproduction summary](docs/reproduction_summary.md) · [Results](results/PATHS.md) | Results, checkpoints, differences from the paper and reproduction commands |

## Installation

Use **Python 3.11**. The pinned experiment environment includes PyTorch 2.13.0, Cantera 3.0.0, NumPy 1.26.4, SciPy 1.17.1, and `torchdiffeq` at commit `657943a` (including Tsit5, unavailable in the standard PyPI release used for this project).

Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r chemkan/requirements.txt
pip install -e chemkan --no-deps
```

`chemkan/requirements.txt` is the recorded environment, including notebook and testing tools. `chemkan/pyproject.toml` declares the Python package and its runtime dependencies. An offline solver-source fallback is provided in [`chemkan/third_party/torchdiffeq/`](chemkan/third_party/torchdiffeq/README.md).

The Julia inference benchmark additionally requires **Julia 1.12**; see [`results/inference_benchmark/julia/`](results/inference_benchmark/julia/).

## Learn ChemKAN from scratch

The notebook sequence is intended to explain not just *what* each module does, but **how its mathematical expressions turn into code**:

1. **Implement the basics:** [`01_implement_kan.ipynb`](chemkan/notebooks/extra/01_implement_kan.ipynb) introduces a learnable Gaussian RBF, AddKAN, LeanKAN and a small KAN-ODE through guided exercises. Use the [companion guide](docs/implementing_chemkan_from_scratch_guide.md) for explanations and reference solutions.
2. **Understand every learned edge:** [`02_einsum_edge_functions_paper_aligned.ipynb`](chemkan/notebooks/extra/02_einsum_edge_functions_paper_aligned.ipynb) derives the edge-function tensor operations from the paper's notation and checks them against loops and other vectorized forms.
3. **Build and train biodiesel ChemKAN:** [`03_chemkan_from_scratch_paper_aligned.ipynb`](chemkan/notebooks/extra/03_chemkan_from_scratch_paper_aligned.ipynb) builds the model inline: **Gaussian RBF → edge functions → AddKAN → LeanKAN → 156-parameter kinetic core → normalization → ODE right-hand side → numerical integration → loss → gradient update → trajectory predictions**.
4. **Connect the notebook to the repository:** [`04_chemkan_actual_solution_walkthrough.ipynb`](chemkan/notebooks/extra/04_chemkan_actual_solution_walkthrough.ipynb) follows the library's implementation and checks shapes, numerical integration, loss and optimization step by step.
5. **Continue to combustion and gradients:** Study the [two-stage trainer](chemkan/scripts/combustion/train_combustion.py), the [hydrogen reproduction notebook](chemkan/notebooks/hydrogen_reproduction.ipynb), and the [FSA validation notebook](chemkan/notebooks/forward_sensitivity_analysis.ipynb).

**Important distinction:** the from-scratch `03` notebook implements and trains the **biodiesel kinetic-only ChemKAN**. It explains the thermodynamic extension but **does not rebuild the complete hydrogen/methane Stage-1/Stage-2 trainer inline**. Notebook `04` illustrates the library's educational kinetic path; the final reproduction scripts use the author-matched configurations described below.

Open the notebooks using Jupyter Lab after installation:

```bash
jupyter lab chemkan/notebooks/extra/03_chemkan_from_scratch_paper_aligned.ipynb
```

For a non-notebook overview, open the [interactive ChemKAN architecture visualization](docs/visualization/chemkan_architecture.html) in a browser. Its trajectories are **schematic**, not predictions from a trained checkpoint. The [data-generation walkthrough](chemkan/scripts/data_gen/data_generation_walkthrough.ipynb) explains how the committed biodiesel and hydrogen arrays were generated.

## Quick start

The commands below use the installed environment and save outputs to Git-ignored directories: `train_example/` for the simple kinetic-only example and `runs/` for the combustion smoke tests. They do not overwrite the recorded experimental results.

```bash
# Run the tests
cd chemkan && python -m pytest tests -q && cd ..

# Train the small kinetic-only biodiesel example (1,000 epochs)
python chemkan/scripts/train.py

# Check the full two-stage hydrogen pipeline with two epochs per stage
python chemkan/scripts/train_hydrogen.py stage1 --run-dir runs/check/S1 --epochs 2
python chemkan/scripts/train_hydrogen.py stage2 --run-dir runs/check/S2 --epochs 2 \
  --stage1-archive runs/check/S1/archive/stage1_S1_epoch2.pt
```

The short combustion runs are **smoke tests**, not meaningful reproduced models. Neither these commands nor the from-scratch notebook overwrite committed experiment records by default; **figure/evaluation scripts may write to `results/`**, as described below.

## Training and experiment settings

### Choose the right entry point

| Entry point | Purpose | Model |
|---|---|---|
| [`scripts/train.py`](chemkan/scripts/train.py) | Short, editable example; accepts a simple custom `.npz` | Kinetic core only, constant temperature |
| [`scripts/train_biodiesel.py`](chemkan/scripts/train_biodiesel.py) | Full biodiesel figure configurations, FSA, noise levels, checkpointing and resume | Kinetic core only |
| [`scripts/train_hydrogen.py`](chemkan/scripts/train_hydrogen.py) | Hydrogen Stage 1 or Stage 2 | Full combustion ChemKAN |
| [`scripts/train_methane.py`](chemkan/scripts/train_methane.py) | Methane Stage 1 or Stage 2 | Full combustion ChemKAN |
| [`deeponet/train_deeponet.py`](deeponet/train_deeponet.py) | Biodiesel DeepONet baseline | Direct operator prediction |

The hydrogen and methane files are system-specific **entry points** to the shared [`combustion/train_combustion.py`](chemkan/scripts/combustion/train_combustion.py). The shared trainer implements loading, the stages, solving, gradients, losses, logging and checkpoints; the selected system determines the chemical mechanism and configuration.

### Kinetic-only training: biodiesel

Temperature is a constant model input for each trajectory; it is **not predicted as a changing state**. For the recorded biodiesel runs, the kinetic right-hand side is divided by a **time-scale parameter of 50 seconds**. This RHS divisor is **not the loss normalization**.

Example: a 156-parameter model trained on the 5% noise dataset for Figure 5:

```bash
python chemkan/scripts/train_biodiesel.py --data canonical \
  --noise-percent 5 --hidden 4 --solve-mode batched \
  --sensitivity fsa --rhs-divisor 50 --epochs 10000 \
  --run-dir runs/biodiesel/noise05
```

| Option | Default | Recorded figure-run choice | Purpose |
|---|---|---|---|
| `--data` | `author_repo` | `canonical` | Authors' released example arrays or the canonical `.npz` dataset |
| `--data-file` | Depends on data mode | `chemkan/data/generated/biodiesel_v2.npz` | Dataset path |
| `--noise-percent` | `0` | 0, 1, 2, 3, 5, 7, 10, 15 | Stored synthetic noise level (canonical mode) |
| `--hidden` | `4` | 2, 3, 4, 9, 11 | Hidden width; the main model has 156 parameters |
| `--sensitivity` | `direct_autograd` | `fsa` | Differentiate through the solver or integrate forward sensitivities |
| `--solve-mode` | `per_trajectory` | `batched` | Independent solve per trajectory or one batched solve |
| `--rhs-divisor` | `50` | `50` | Divisor applied to the ODE right-hand side, in seconds |
| `--epochs` | `10000` | 5,000 (Fig. 4); 10,000 (Fig. 5) | Total training epochs |
| `--seed` | `0` | `0` | Random seed |

For less frequently used options (initializing from a checkpoint, checkpoint frequency, threads and resuming), use `python chemkan/scripts/train_biodiesel.py --help`.

### Two-stage combustion: hydrogen and methane

**Stage 1 — kinetic core.** Integrate the species while providing the observed temperature trajectory as an input. The training temperature provider uses dense Cantera temperature histories. **Stage 2 — kinetics + thermodynamics.** Initialize from a Stage-1 archive, add the thermodynamic linear branch and KAN correction, then optimize species and temperature together. **The kinetic core is not frozen in Stage 2.** The full loss includes an element-conservation penalty of weight `1e-4`.

```bash
# Example: hydrogen, full recorded Stage-1 training budget
python chemkan/scripts/train_hydrogen.py stage1 \
  --run-dir runs/hydrogen/S1_seed0 --epochs 100000

# Stage 2 begins from the saved Stage-1 archive
python chemkan/scripts/train_hydrogen.py stage2 \
  --run-dir runs/hydrogen/S2_seed0 --epochs 100000 \
  --stage1-archive runs/hydrogen/S1_seed0/archive/stage1_S1_seed0_epoch100000.pt
```

Use `train_methane.py` with the corresponding methane archive and run directory. The two systems share the training procedure but use different species, datasets and reference times.

| Option | Hydrogen default | Methane default | Meaning |
|---|---|---|---|
| `--epochs` | `100000` | `100000` | Total number of epochs in the selected stage |
| `--t-ref` | `6e-4` s | `5e-3` s | Right-hand-side time divisor (the system's observation window) |
| `--lr` | `2e-3` | `2e-3` | Adam learning rate |
| `--rtol`, `--atol` | `1e-6`, `1e-8` | Same | Tsit5 solver tolerances |
| `--dtype` | `float32` | Same | Arithmetic precision |
| `--solve-mode` | `batched` | Same | Batched or per-trajectory ODE solves |
| `--sensitivity` | `direct_autograd` | Same | Gradient backend; the recorded runs used direct autograd |
| `--checkpoint-every` | `500` | Same | Working checkpoint interval |
| `--archive-every` | `5000` | Same | Permanent snapshot interval |
| `--stage1-archive` (Stage 2) | Required | Required | Initial Stage-1 checkpoint |
| `--warmup-epochs` (Stage 2) | `0` | Same | Optional thermodynamic-linear warm-up |

To resume an interrupted run, use `--resume` with the same run directory. Set a larger `--epochs` value to extend a completed stage. For the complete multi-stage experiment commands, use the launchers in the next section.

## Reproduce the recorded experiments

Run these from the repository root. **Figure-run launchers may skip completed checkpoints** already provided in `results/`; use separate output directories for fresh training.

| Experiment | Launcher | Existing records |
|---|---|---|
| Biodiesel ChemKAN (13 runs; Figs. 3–6) | `python chemkan/scripts/biodiesel/run_figure_runs.py` | [`results/biodiesel/`](results/biodiesel/README.md) |
| Biodiesel DeepONet (14 runs) | `bash deeponet/run_figure_runs.sh` | [`results/biodiesel/`](results/biodiesel/README.md) |
| Hydrogen Stage 1 → Stage 2 | `bash chemkan/scripts/combustion/run_hydrogen.sh` | [`results/hydrogen/`](results/hydrogen/README.md) |
| Methane Stage 1 → Stage 2 | `bash chemkan/scripts/combustion/run_methane.sh` | [`results/methane/`](results/methane/README.md) |
| Inference benchmark | `bash chemkan/scripts/benchmark/run_inference_benchmark.sh` | [`results/inference_benchmark/`](results/inference_benchmark/README.md) |

For an independent biodiesel/DeepONet rerun without overwriting committed model runs:

```bash
export BIODIESEL_RESULTS=runs/biodiesel_rerun
python chemkan/scripts/biodiesel/run_figure_runs.py
bash deeponet/run_figure_runs.sh
python chemkan/scripts/figures/plot_all.py --only fig03 fig04 fig05a fig05b fig06
```

The combustion launchers use `runs/<system>/` by default; set `OUT=<directory>` to redirect them. The inference benchmark writes timing outputs to its designated directory: check its README before rerunning it. Historic launch scripts under `results/` are retained as records and may use obsolete development paths; use the **current launchers** above.

## Evaluation, figures and interpretability

Figures can be regenerated from saved checkpoints and evaluation records without retraining. **Some plotting/evaluation commands write into `results/`**, potentially changing timestamps or machine-specific paths. Use alternate outputs where supported if you want a clean working tree.

| Output | Script or command | Notes |
|---|---|---|
| Original-paper Figs. 3–8 | `python chemkan/scripts/figures/plot_all.py` | Reads biodiesel and hydrogen recorded runs; see [figure guide](chemkan/scripts/figures/README.md) |
| Hydrogen checkpoint metrics | `python chemkan/scripts/combustion/evaluate_checkpoint.py --checkpoint <pt> --out <json>` | Losses; Stage-2 temperature and ignition metrics |
| Hydrogen 441-condition evaluation | `python chemkan/scripts/combustion/evaluate_hydrogen_grid.py --run-dir results/hydrogen/S2_seed0_200k --out runs/eval --label main_S2_seed0_200k` | Paper Fig. 8A analogue |
| Hydrogen ignition delays | `python chemkan/scripts/combustion/evaluate_hydrogen_ignition.py` | See `--help` for its table/output arguments |
| Methane generalization | `python chemkan/scripts/combustion/evaluate_methane_grid.py` | 441-condition extension grid |
| Individual trajectory plots | `python chemkan/scripts/combustion/plot_trajectory.py --checkpoint <pt> --out-dir runs/plots` | Inspect saved Stage-2 model |
| **Methane key-species figure** | `python chemkan/scripts/figures/methane_species.py` | CH₄, O₂, CO₂, H₂O and CO, comparing prediction and reference at 1550 K / φ 1.3 |
| Interpretability | `python chemkan/scripts/interpretability/biodiesel.py`; `python chemkan/scripts/interpretability/combustion.py --system hydrogen` (or `methane`) | See recorded per-system interpretability READMEs |

The methane figure is a **project extension, not a figure from the ChemKAN paper**. Its selected model is `results/methane/S2_seed0/archive/stage2_S2_seed0_epoch180000.pt`, and the output is [`results/methane/figures/`](results/methane/figures/).

## Main results

The tables below are a concise overview, **not claims of exact reproduction**. Unless stated otherwise, reported MSE is averaged over time and states in training-set min–max-normalized coordinates. For figure-by-figure evidence, evaluation definitions and discrepancies, see [`docs/reproduction_summary.md`](docs/reproduction_summary.md).

### Biodiesel — model inference and noise

The DeepONet comparison uses project-specific tanh activation and Adam at a learning rate of `1e-2`; the paper does not specify these settings. See [DeepONet details](deeponet/README.md).

| Comparison | Published paper | This reproduction |
|---|---|---|
| ChemKAN noise-free test MSE ratio, 15% vs. 0% training noise | About 2× | 1.84× |
| DeepONet noise-free test MSE ratio, 15% vs. 0% | About 5× | 6.90× |
| DeepONet / ChemKAN noise-free test MSE at 15% noise | 4.4× | 2.73× (same ordering, smaller difference) |
| ChemKAN noise-free neural scaling: training / test slopes | 1.0 / 0.6 | 1.49 / 0.37 (**does not match numerically**) |
| Figure 5A comparison across noise levels | ChemKAN has lower test MSE throughout | **Not reproduced at every noise level**; see the detailed comparison |

### Hydrogen — two-stage combustion

| Comparison | Published paper | This reproduction |
|---|---|---|
| Model size | 344 parameters | 344 parameters |
| Figure 7 train/held-out profiles | Accurate reconstructions | MSE about `4.5e-4` / `3.5e-4` |
| Figure 8A, 441-condition grid | Errors mainly near `1e-4`, with weaker cold-region performance | Median MSE about `4.9e-4`; substantially higher in the coldest region |
| Figure 8B ignition delays | Good agreement | Median absolute relative error about **1.4%** across 30 igniting conditions |
| Speed-up vs. detailed chemistry | **2×** vs. Arrhenius.jl | Julia ChemKAN port: **6.3×** vs. Arrhenius.jl at the training tolerance, and **2.5×** vs. Cantera at its dataset tolerance; PyTorch version slower |

### Methane — new project extension

The paper does not contain a methane experiment. We used GRI-Mech 3.0 with Ar excluded from the modeled state, leaving 52 species. We selected the 180,000-epoch Stage-2 snapshot because the 200,000-epoch model lies on a training-loss spike and training had not converged.

| Metric | Recorded result |
|---|---|
| Training-condition median MSE | About `9.7e-4` |
| Held-out 1550 K / φ 1.3 MSE | About `1.16e-3` |
| Median MSE across 406 unseen grid conditions | About `9.2e-4` |
| Median ignition-delay relative error | About **1.4%** |
| Julia ChemKAN inference speed | **4.0×** vs. Cantera; **13.5×** vs. Arrhenius.jl (under the benchmark settings) |

Links: [biodiesel notebook](chemkan/notebooks/biodiesel_reproduction.ipynb) · [hydrogen notebook](chemkan/notebooks/hydrogen_reproduction.ipynb) · [methane notebook](chemkan/notebooks/methane_extension.ipynb) · [inference benchmark](results/inference_benchmark/README.md).

## Use your own data

The simple kinetic-only [`train.py`](chemkan/scripts/train.py) accepts one NumPy `.npz` file:

| Key | Shape | Description |
|---|---|---|
| `t` | `(n_times,)` | Shared, increasing observation times |
| `train_states` | `(n_train, n_times, m)` | Training species trajectories |
| `test_states` | `(n_test, n_times, m)` | Held-out species trajectories |
| `train_T` | `(n_train,)` | Constant temperature of each training trajectory |
| `test_T` | `(n_test,)` | Constant temperature of each held-out trajectory |

```bash
python chemkan/scripts/train.py --data my_system.npz
```

The example derives its normalization from the training data and constructs a kinetic core for `m` species. It assumes **one common time grid and constant temperature per trajectory**. Its `rhs_divisor` must be chosen for the new system's timescale; the biodiesel value **50 s** is not a universal constant.

For systems with evolving temperature, study [`train_hydrogen.py`](chemkan/scripts/train_hydrogen.py), [`train_methane.py`](chemkan/scripts/train_methane.py) and [`combustion/train_combustion.py`](chemkan/scripts/combustion/train_combustion.py). The existing combustion scripts are **system-specific**, not a plug-and-play trainer for any mechanism. See the [data-generation documentation](chemkan/scripts/data_gen/README.md) for the supplied datasets.

## Reproducibility and limitations

- **Fixed configurations, single seeds.** The recorded main ChemKAN experiments use seed 0; cross-seed variability was not systematically evaluated. The unimplemented ChemNODE baseline from the original project proposal is not included.
- **Reproduction checks.** Short five-epoch reruns of the 31 principal ChemKAN/DeepONet configurations agreed with the recorded histories exactly or within cross-platform floating-point differences. This does **not** prove full 200,000-epoch runs would be bit-identical on another machine.
- **Platform differences.** Parts of biodiesel and methane Stage 1 were run on a Linux x86 JupyterHub, while other runs used an ARM Mac. Matching random seeds does not guarantee identical last-bit numerical results across those environments, particularly with an adaptive ODE solver.
- **Saved runs.** Historical experiments were interrupted/resumed; the repository includes selected checkpoints, histories, figures, metrics and configuration files. [Results path mappings](results/PATHS.md) describe moved records. The selected 33 histories retain all epoch rows but only analysis-essential columns; the full diagnostic histories remain in the development archive.
- **Scientific discrepancies.** Biodiesel neural-scaling slopes and several DeepONet/ChemKAN comparisons differ from the paper; hydrogen generalization deteriorates around 975–987.5 K. The methane training is not converged and can predict slightly negative CH₄/O₂ after ignition at the designated held-out condition. See the [reproduction summary](docs/reproduction_summary.md).
- **Performance scope.** Reported Julia speed-ups depend on solver tolerances and the measured environment. They must not be attributed to the PyTorch implementation.
- **FSA validation.** FSA gradient checks are documented in the [validation notebook](chemkan/notebooks/forward_sensitivity_analysis.ipynb) and tested in [`test_fsa.py`](chemkan/tests/test_fsa.py). The hydrogen/methane **recorded runs do not use FSA**.
- **Provenance.** Some data-generation and implementation conventions followed direct clarification from the original ChemKAN authors. The public code, dataset metadata and [reproduction summary](docs/reproduction_summary.md) document the applied conventions; the private correspondence is not included.

## Repository layout

```text
chemkan/
├── src/chemkan/              reusable KAN layers, ChemKAN model, dynamics, FSA and losses
├── scripts/
│   ├── train.py              simple kinetic-only training example
│   ├── train_biodiesel.py    final biodiesel reproduction trainer
│   ├── train_hydrogen.py     hydrogen two-stage training entry point
│   ├── train_methane.py      methane two-stage training entry point
│   ├── combustion/           shared combustion trainer, evaluation and launchers
│   ├── data_gen/             SciPy/Cantera datasets and generation walkthrough
│   ├── figures/              paper figures and methane extension figure
│   ├── interpretability/     learned-function analysis
│   ├── benchmark/            inference performance experiments
│   └── biodiesel/            biodiesel reproduction launcher and checks
├── notebooks/
│   ├── extra/                **from-scratch and source-code explanation notebooks**
│   ├── forward_sensitivity_analysis.ipynb
│   └── ...                   biodiesel, hydrogen and methane reproduction notebooks
├── data/generated/           included, regenerable datasets
├── tests/                    automated tests
└── third_party/torchdiffeq/  pinned solver source and license
deeponet/                     PyTorch DeepONet baseline and upstream_reference/
docs/                        original paper, sources, interactive visualization, learning guide,
                             detailed reproduction summary
results/                     selected experimental artifacts; PATHS.md documents provenance
```

## Documentation and references

| Resource | Description |
|---|---|
| [Reproduction summary](docs/reproduction_summary.md) | Figure-by-figure paper-versus-ours findings, numerical evidence and limitations |
| [Learning guide](docs/implementing_chemkan_from_scratch_guide.md) | Detailed reference explanations and solutions for the guided KAN workbook |
| [Notebook index](chemkan/notebooks/README.md) | What each reproduction, FSA and from-scratch notebook contains |
| [Interactive architecture visualization](docs/visualization/chemkan_architecture.html) | Browser-based schematic of the ChemKAN architecture (illustrative trajectories) |
| [Published ChemKAN paper](docs/paper/d5cp02009c.pdf) | The original article; [paper attribution/license](docs/paper/README.md) |
| [Sources and further reading](docs/sources.md) | Related literature, tools and code references |
| [DeepONet details](deeponet/README.md) | Baseline implementation, reproduction choices and upstream relationship |
| [Result mappings](results/PATHS.md) | Historical experiment paths, retained artifacts and diagnostics |
| [Project poster](docs/report_poster_and_proposal_project/poster/Bankoglu_ChemKAN_Poster.pdf) | A0 poster; editable source and verification scripts in [`Bankoglu_ChemKAN_Poster_Source.zip`](docs/report_poster_and_proposal_project/poster/Bankoglu_ChemKAN_Poster_Source.zip) |

**Primary reference**

B. C. Koenig, S. Kim and S. Deng, *ChemKANs for combustion chemistry modeling and acceleration*, **Physical Chemistry Chemical Physics**, 27, 17313–17330 (2025). [DOI: 10.1039/D5CP02009C](https://doi.org/10.1039/D5CP02009C) · [arXiv](https://arxiv.org/abs/2504.12580).

```bibtex
@article{koenig2025chemkans,
  title   = {ChemKANs for combustion chemistry modeling and acceleration},
  author  = {Koenig, Benjamin C. and Kim, Suyong and Deng, Sili},
  journal = {Physical Chemistry Chemical Physics},
  volume  = {27},
  pages   = {17313--17330},
  year    = {2025},
  doi     = {10.1039/D5CP02009C}
}
```

The code in this repository is released under the [MIT License](LICENSE). The original paper PDF, vendored `torchdiffeq`, and upstream DeepONet source have **their own respective license terms**; see the license files and attribution within those directories.
