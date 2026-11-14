   # Project log

   ## 2026-10-28 — Setup
   - Forked bnkglu/chemkan-tuwien, set up Python 3.11 venv on Windows.
   - Tests: 258/260 pass; 2 failures are Windows-specific
     (bit-exact data comparison, deleting an open log file).
   - Ran scripts/train.py (biodiesel, 156 params, 1000 epochs):
     train MSE 5.4e-4, test MSE 1.6e-3, still decreasing.

  ## 2026-11-04
   **Done:** Worked through Berke's notebooks 01–04 and train.py.
   **Learned:** ChemKAN learns the rate rule; the TBM will produce the training curves.

  ## 2026-11-14
  **Done:** Built the TBM data generator (gas_solid/data_gen/tbm_oxidation.py):
  gas properties (Cantera), boundary layer (Re, Sc, Sh, beta), pores (D_eff, Thiele modulus, eta),
  oxidation rate, gas/carbon amounts, rate-of-change function, simulation over time.

  **Learned / decided:**
  - Coke oxidation only (C + O2 -> CO2); gasification later.
  - Parameter ranges: T 300-3000 K, p up to 10 bar, 5-30 % O2 (rest N2), U up to 150 m/s, alpha 0-0.7.
  - Formulas for eta, Thiele modulus, Sherwood number, D_eff and k_eff, and the oxidation kinetics: k_i = 4.8e9 * exp(-16731/T), n = 0.59.
  - Particle keeps its size, O2 is consumed, isothermal (for now).
  - Particle diameter fixed at d_p = 10 mm. Assumption.
  - Pore diameter 1 um (for Knudsen diffusion). Assumption
  - Coke density 1000 kg/m3. Assumption
  - Thiele modulus divided by D_eff (as thesis eq. 2.56; missing in the email).
  - Boundary-layer term in k_eff uses c^(1-n) (thesis eq. 2.66); the email has c^(n-1)
  - Gas volume per particle from the void fraction: V_gas = V_p * alpha / (1 - alpha). Assumption.
  - Reaction stops when x_O2 < 1e-10.
  - Gas properties (density, viscosity, O2 diffusion coefficient) from Cantera's gri30 model.
  - Dimensionless time tau = t / t_ref, so all trajectories share one time grid