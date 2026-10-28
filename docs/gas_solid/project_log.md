   # Project log

   ## 2026-10-28 — Setup
   - Forked bnkglu/chemkan-tuwien, set up Python 3.11 venv on Windows.
   - Tests: 258/260 pass; 2 failures are Windows-specific
     (bit-exact data comparison, deleting an open log file).
   - Ran scripts/train.py (biodiesel, 156 params, 1000 epochs):
     train MSE 5.4e-4, test MSE 1.6e-3, still decreasing.