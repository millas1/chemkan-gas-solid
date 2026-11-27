import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

# constants
R = 8.314462618        # J/(mol K)
M_O2 = 0.032           # kg/mol
M_C = 0.012            # kg/mol

settings = {
    # parameter ranges
    "T_range" : (300, 3000),          #K
    "P_range" : (0.1e5, 10e5),            #Pa (0.1-10) bar
    "U_range" : (0.1, 150),           #m/s
    "x_o2_range" : (0.05, 0.30),    #mole fraction o2, rest n2
    "alpha_range" : (0.5, 0.5),       #alpha

    # fixed values
    "d_p": 0.01,                        # m, particle diameter. ASSUMPTION
    "d_pore": 1e-6,                     # m, pore diameter for Knudsen diffusion. ASSUMPTION
    "rho_coke": 1000.0,                 # kg/m3, coke density. ASSUMPTION
 
    # oxidation kinetics C + O2 -> CO2
    "A_ox": 4.8e9,                      # A 
    "TA_ox": 16731.0,                   # K, activation temperature
    "n_ox": 0.59,                       # reaction order in O2
 
}

# gas properties
gas = ct.Solution("gri30.yaml")
def gas_properties(T, P, x_O2, x_CO2):
    
    gas.TPX = T, P, {"O2": x_O2, "CO2": x_CO2, "N2": 1 - x_O2 - x_CO2}

    rho = gas.density
    mu = gas.viscosity
    D_O2 = gas.mix_diff_coeffs[gas.species_index("O2")]
    
    return rho, mu, D_O2

# beta calculation
def boundary_layer(rho, mu, D_O2, U, d_p, alpha):

    Re = (rho*U*d_p)/mu
    Sc = mu / (rho*D_O2)
    Sh = 0.357/(1-alpha)*Re**0.641*Sc**(1/3)
    beta = (Sh*D_O2)/d_p

    return beta

# oxygen diffusing into the pores
def pores(T, P, x_O2, D_O2, settings):

    # Knudsen diffusion
    D_K = settings["d_pore"] / 3 * np.sqrt((8*R*T) / (np.pi*M_O2))

    # Effective diffusion
    D_eff = 1 / ((1/D_K) + (1/D_O2))

    # Rate constant (per m3 of particle). 
    # convert to mol units, so it fits c in mol/m3 everywhere else: factor 1000^(1-n)
    n = settings["n_ox"]
    k_i = settings["A_ox"] * np.exp(-settings["TA_ox"]/T) * 1000**(1 - n)

    # O2 concentration
    c = (x_O2 * P) / (R*T)

    # Thiele Modulus
    phi = (settings["d_p"]/2) * np.sqrt((n+1)/2 * k_i * c**(n-1) / D_eff)

    # Effectiveness factor (uses 1 - phi^2/15 for phi < 1e-3 (full formula is numerically unstable there))
    if phi < 1e-3:
        eta = 1 - phi**2 / 15
    else:
        eta = (3/ phi)*(1/np.tanh(phi) - 1/phi)

    return eta, k_i, c

# Oxidation rate
def oxidation_rate(T, P, x_O2, x_CO2, U, alpha, settings):
    n = settings["n_ox"]
    d_p = settings["d_p"]
    rho, mu, D_O2 = gas_properties(T, P, x_O2, x_CO2)
    beta = boundary_layer(rho, mu, D_O2, U, d_p, alpha)
    eta, k_i, c = pores(T, P, x_O2, D_O2, settings)

    resistence_bl = 1/(beta*np.pi*d_p**2*c**(1-n))
    V_p = np.pi/6 * d_p**3                    # particle volume, m3
    resistence_chem = 1/(eta*k_i*V_p)         # k_i is per m3 of particle
    k_eff = 1/(resistence_bl + resistence_chem)
    rate = k_eff*c**(n)

    return rate

# amount of gas
def amounts(T, P, alpha, settings):

    # particle volume
    V_p = np.pi/6 * settings["d_p"]**3

    # gas volume
    V_gas = V_p*(1-alpha) / alpha

    # moles of gas
    n_gas = P*V_gas / (R*T)

    # moles carbom
    n_C = settings["rho_coke"] * V_p / M_C

    return n_gas, n_C

# rate of change
def rhs(t, y, T, P, U, alpha, n_gas, n_C0, settings):

    # current state values
    x_O2, x_CO2, x_C = y

    # if oxygen is used up, nothing changes anymore
    if x_O2 < 1e-10:
        return [0.0,0.0,0.0]

    # consumption per second right now
    rate = oxidation_rate(T, P, x_O2, x_CO2, U, alpha, settings)

    # how fast each state changes
    dx_O2 = -rate / n_gas       # O2 is used up therefore -rate
    dx_CO2 = rate / n_gas
    dX_C = rate / n_C0

    return [dx_O2, dx_CO2, dX_C]

# one particle over time
def simulate(T, P, x_O2_0, U, alpha, tau, settings):

    # how much gas and carbon there is
    n_gas, n_C0 = amounts(T, P, alpha, settings)

    # this particle's time unit
    rate0 = oxidation_rate(T,P,x_O2_0,0.0,U,alpha,settings)
    t_ref = x_O2_0*n_gas / rate0
    # turn into real time
    t = tau *t_ref

    # the starting state
    y0 = [x_O2_0, 0.0, 0.0]

    # step forward
    sol = solve_ivp(
        rhs,                                            # the ROC function
        (t[0], t[-1]),                                  # start and end time
        y0,                                             # starting state
        t_eval=t,                                       # the times for results
        args=(T, P, U, alpha, n_gas, n_C0, settings),   # the fixed conditions, passed on to rhs
        method="LSODA",                                 # a solver that handles fast and slow changes
        rtol=1e-8, atol=1e-12,                          # tol
    )

    return sol.y.T, t_ref

# random sets of conditions from the ranges in settings
def sample_conditions(n, rng, settings):
    T = rng.uniform(*settings["T_range"], n)
    P = rng.uniform(*settings["P_range"], n)
    x_O2 = rng.uniform(*settings["x_o2_range"], n)
    U = rng.uniform(*settings["U_range"], n)
    alpha = rng.uniform(*settings["alpha_range"], n)

    #one row per trajectory, one col per condition
    return np.column_stack([T, P, x_O2, U, alpha])

# simulate n particles with random conditions
def generate(n, rng, tau, settings, label):

    conditions = sample_conditions(n, rng, settings)

    # one trajectory per particle
    states = np.empty((n, len(tau), 3))     # (particles, time points, 3 states)
    t_refs = np.empty(n)                    # each particle's time unit
    ratios = np.empty(n)                    # NEW: moles of gas / moles of carbon

    for i in range(n):
        T, P, x_O2_0, U, alpha = conditions[i]
        states[i], t_refs[i] = simulate(T, P, x_O2_0, U, alpha, tau, settings)
        n_gas, n_C0 = amounts(T, P, alpha, settings)
        ratios[i] = n_gas / n_C0
        print(f"{label} {i + 1}/{n}  T = {T:.0f} K  t_ref = {t_refs[i]:.2e} s")

    return conditions, states, t_refs, ratios              # NEW: also return ratios

# --- generate the dataset ---
rng = np.random.default_rng(0)               # fixed seed: same data every run
tau = np.linspace(0, 5, 50)                  # dimensionless time grid

cond_train, states_train, tref_train, ratio_train = generate(40, rng, tau, settings, "train")
cond_test, states_test, tref_test, ratio_test = generate(10, rng, tau, settings, "test")

# --- save in the format train.py reads ---
out = Path("gas_solid/data/generated/tbm_oxidation_v1.npz")
out.parent.mkdir(parents=True, exist_ok=True) 

np.savez(
    out,
    # the five arrays train.py needs
    t=tau,
    train_states=states_train,
    test_states=states_test,
    train_T=cond_train[:, 0],                    # column 0 = temperature
    test_T=cond_test[:, 0],
    # extra information, ignored by train.py
    train_conditions=cond_train,
    test_conditions=cond_test,
    train_t_ref=tref_train,
    test_t_ref=tref_test,
    train_gas_to_carbon=ratio_train,
    test_gas_to_carbon=ratio_test,
)
print("saved", out)