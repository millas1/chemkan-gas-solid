"""Plot ChemKAN predictions against the TBM data for the test particles."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# make Berke's training script and its helpers importable
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "chemkan" / "scripts"))
from train import load_npz, predict                               
from reference_components import NormalizedDynamics, build_core     
from chemkan.solver import SolverConfig                              

DATA = REPO / "gas_solid/data/generated/tbm_oxidation_v1.npz"
RUN = REPO / "gas_solid/runs/tbm_baseline"
NAMES = ["x_O2", "x_CO2", "X_C (carbon conversion)"]

torch.set_default_dtype(torch.float64)        # same precision as in training

# load the trained model and its settings
ckpt = torch.load(RUN / "chemkan_kinetic.pt", weights_only=False)
s = ckpt["settings"]

# load the data, normalized exactly as in training
data = load_npz(DATA)
t, m = data["t"], data["m"]
T_test = np.load(DATA)["test_T"]

# rebuild the model and put the trained knob settings back in
core = build_core(s["hidden"], species_dim=m)
core.load_state_dict(ckpt["model_state"])
dynamics = NormalizedDynamics(core, s["rhs_divisor"])
solver = SolverConfig(method="tsit5", rtol=s["rtol"], atol=s["atol"],
                      sensitivity="direct_autograd")

# predict all test particles from their starting states
with torch.no_grad():
    pred = predict(dynamics, data["u0_test"], t, solver, m)   # (particles, states, times)
true = data["target_test"]

# error per test particle, sorted from best to worst
mse = ((pred - true) ** 2).mean(dim=(1, 2)).numpy()
order = np.argsort(mse)
print("test particles, best to worst:")
for i in order:
    print(f"  particle {i}:  T = {T_test[i]:6.0f} K   MSE = {mse[i]:.2e}")

# back to physical units for plotting
n = data["normalization"]
y_min, y_span = n["y_min"][None, :, None], (n["y_max"] - n["y_min"])[None, :, None]
pred_phys = (pred * y_span + y_min).numpy()
true_phys = (true * y_span + y_min).numpy()

# plot the best, a middle and the worst particle
picks = {"best": order[0], "middle": order[len(order) // 2], "worst": order[-1]}
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
for row, (label, i) in enumerate(picks.items()):
    for col in range(3):
        ax = axes[row, col]
        ax.plot(t, true_phys[i, col], "o", ms=3, color="0.4", label="TBM (data)")
        ax.plot(t, pred_phys[i, col], "-", color="C0", label="ChemKAN")
        if row == 0:
            ax.set_title(NAMES[col])
        if col == 0:
            ax.set_ylabel(f"{label}: {T_test[i]:.0f} K")
        if row == 2:
            ax.set_xlabel("dimensionless time tau")
axes[0, 0].legend()
fig.suptitle("Baseline ChemKAN vs TBM, test particles")
fig.tight_layout()

out = RUN / "predictions.png"
fig.savefig(out, dpi=150)
print("saved", out)
plt.show()