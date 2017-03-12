"""  
This script implements the thermal model for Electromigration (EM) in nanowires (NW) as described in Xiang, APL,
2014. Various modifications to the model, such as junction geometry, are also implemented along with the numerical
machinery needed to solve the modified model. This script also houses a large number of plotting recipes for visualizing
the model 
"""

__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.interactive(True)
import snips as snp
snp.prettyplot(matplotlib)
import pickle
from scipy.optimize import least_squares
from scipy.integrate import ode
from scipy.interpolate import UnivariateSpline as spline
import os

AREA = 15e-9*50e-9  # Nanowire cross-sectional area (50nm width, 15nm height)

os.chdir("C:/Users/Sonya/Documents/My Box Files/molT Project/170311_SDS20_ThermalModel")

def compute_pow(tcrit, tenv):
    """
    Calculates power per x-section area dissipated in NW given environmental temp & critical junction temp. This is the
    simple model outlined in Xiang2014 - it is the solution to the quasi-1D steady state for a rectangular geometry.
    
    :param tcrit: The critical temperature of the junction for EM initiation (in Kelvin) 
    :param tenv: The environmental temperature that the left side is pinned at (in Kelvin)
    :return: critical power dissipated in the junction needed to drive compute_tcrit given tenv (in W/m^2)
    """
    x = 3e-9*tenv + 1.1e-6  # (in meters) NW length, depends on temp
    k = -0.069*(tcrit + tenv)/2 + 338  # (in W/m*K) Thermal conductivity Au, also temp-dependent
    pow = k/x * (tcrit - tenv)  # (in W/m^2)
    return pow

def compute_tcrit(pc, tenv):
    """
    Numerically solves Xiang 2014's model for the critical temperature in the junction given a power per x-section
    dissipated in the junction and an environmental temperature. Xiang's model is the solution to the quasi-1D steady
    state for a rectangular geometry.
    
    :param pc: critical power per x-sectional area dissipated in the junction (in W/m^2)
    :param tenv: environmental temperature that left side is pinned at (in Kelvin)
    :return: critical temperature sustained in the junction given this power and env. temp. (in Kelvin)
    """
    def residuals(theta, pc, tenv):  # Theta is the parameter Tcrit to be fitted
        tc = theta[0]
        pc_estimate = compute_pow(tcrit=tc, tenv=tenv)
        return pc - pc_estimate
    tcrit_0 = [380]  # Initial guess for crital temperature.
    lsq = least_squares(residuals, tcrit_0, args=(pc, tenv), loss="soft_l1")
    return lsq.x[0]  # Fitted estimate of compute_tcrit (in Kelvin)


######################################################################################################################
#   Plot predictions of Xiang Model and compare with linear fit to critical power data.
# Compute and plot some predicted power vs. critical temperature curves with the constant critical temp assumption
tenvs = np.linspace(95, 260, 50)
tcrits = np.linspace(380, 580, 3)
fig, ax = plt.subplots()
snp.labs("Env. Temp. (K)", r"$\frac{P_c}{A}$ ($\mu W/nm^2$)", "Constant Critical Temp Model")
for tc in tcrits:
    ps = [compute_pow(tc, tenv) * 10 ** -12 for tenv in tenvs]
    ax.plot(tenvs, ps, label="Jct Crit Temp = %.0f K" %(tc))

# Plot a straight line fit to 480 degree curve (straight line is what Xiang did for their data)
ps = [compute_pow(480, tenv) * 10 ** -12 for tenv in tenvs]
line = np.polynomial.polynomial.polyfit(tenvs, ps, deg=1)
ys = np.polynomial.polynomial.polyval(tenvs, line)
ax.plot(tenvs, ys, label="Straight Line Pc vs. T")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("XiangWithLinear_Pc_vs_Tenv.png")

# Plot the temperature profiles resulting from the linear fit versus the constant critical temp curve
fig, ax = plt.subplots()
snp.labs("Env. Temp. (K)", r"Solved $T_c$", "Constant Critical Temp Model")
tcs = [compute_tcrit(pc * 10 ** 12, tenv) for pc, tenv in zip(ys, tenvs)]
ax.plot(tenvs, tcs, label="Solved $T_c$ For Straight Line")
tcs = [compute_tcrit(pc * 10 ** 12, tenv) for pc, tenv in zip(ps, tenvs)]
ax.plot(tenvs, tcs, label="Solved $T_c$ For Curve")
# ax.plot(tenvs_data, intensity_units, label="data")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("XiangWithLinear_Tc_vs_Tenv.png")


######################################################################################################################
#   Analysis of Xiang actual data to replicate their fitting results.
xiang_tenvs = [4.2, 60, 100, 150, 195, 250, 298]  # In Kelvin
xiang_ps = np.array([0.185, 0.176, 0.136, 0.135, 0.12, 0.1, 0.063])  # In uW/nm^2
line = np.polynomial.polynomial.polyfit(xiang_tenvs, xiang_ps, deg=1)
ys = np.polynomial.polynomial.polyval(xiang_tenvs, line)

# Plot their measured power and
fig, ax = plt.subplots()
snp.labs("Environmental Temp (K)", r"$\frac{P_c}{A}$ ($\mu W/nm^2$)", "Xiang Data and Fit")
ax.plot(xiang_tenvs, xiang_ps, label="Xiang Data", lw=0)
ax.plot(xiang_tenvs, ys, label="Linear fit to Xiang Data")
plt.tight_layout()
plt.savefig("Xiang_data_and_fit.png")

# Fit the Xiang Data to their Model using exactly their measured critical power values
fig, ax = plt.subplots()
snp.labs("Env. Temp. (K)", r"Solved $T_c$", "Xiang Data $T_C$")
line_realunits = np.polynomial.polynomial.polyfit(xiang_tenvs, xiang_ps*10**12, deg=1)  # Correct units and taking half
ys = np.polynomial.polynomial.polyval(xiang_tenvs, line_realunits)
tcs = [compute_tcrit(pc, tenv) for pc, tenv in zip(ys, xiang_tenvs)]
ax.plot(xiang_tenvs, tcs, label="Fitted $T_c$ using full measured power")

# Fit the Xiang Data to their Model using only half their measured critical power values
line_realunits = np.polynomial.polynomial.polyfit(xiang_tenvs, xiang_ps*10**12/2, deg=1)  # Correct units and taking half
ys = np.polynomial.polynomial.polyval(xiang_tenvs, line_realunits)
tcs = [compute_tcrit(pc, tenv) for pc, tenv in zip(ys, xiang_tenvs)]
ax.plot(xiang_tenvs, tcs, label="Fitted $T_c$ using half measured power")
ax.legend(loc="best")
plt.savefig("Xiang_data_solvedTc.png")

# Make a grid plot of LHS - RHS to visualize the solution points
tenvs = np.linspace(100, 250, 100)
tcs = np.linspace(300, 600, 200)
grid_tenvs, grid_tcs = np.meshgrid(tenvs, tcs)
rhs = 17.55*grid_tenvs**2 - 1.18e4*grid_tenvs - 3.03e6
lhs = grid_tcs**2 - 9.8e3*grid_tcs
diffs = np.abs(rhs - lhs)
plt.imshow(diffs < 10000, extent=[100, 250, 600, 300])
snp.labs("Tenvs", "Tcs", "Visualize Solution to Xiang Model")


######################################################################################################################
#   Generate an expression for thermal conductivity of gold versus temperature.
# Reference = http://www.efunda.com/materials/elements/TC_Table.cfm?Element_ID=Au
gold_k_temps = [300, 250, 200, 150, 100, 90, 80, 70]
gold_k_conds = [317, 321, 323, 325, 327, 328, 332, 348]
fig, ax = plt.subplots()
snp.labs("environmental temperature (K)", "$\kappa$ (W/m*k)", "Au Thermal Conductivity")
ax.plot(gold_k_temps, gold_k_conds, label="data", markersize=10)

# Fit the experimental conductivity in two pieces to capture two different shapes
fit_curve = np.polynomial.polynomial.polyfit(gold_k_temps[-4:], gold_k_conds[-4:], deg=5)
fit_line = np.polynomial.polynomial.polyfit(gold_k_temps[1:5], gold_k_conds[1:5], deg=1)

# Make predictions in the curved region using the fit, then adjust them by hand to capture the shape better
fitxs_curve = np.arange(70, 102, 2)
fitys_curve = np.polynomial.polynomial.polyval(fitxs_curve, fit_curve)
fitys_curve[6:10] = [331, 330.1, 329.3, 328.6]
fitys_curve[-4:] = [327.6, 327.4, 327.1, 327]
fitys_curve[:4] = [349, 345, 341, 337.6]
ax.plot(fitxs_curve, fitys_curve, label="5 degree fit (manually adjusted)")

# Make predictions in the linear region using the fit
fitxs_line = np.arange(100, 500, 10)
fitys_line = np.polynomial.polynomial.polyval(fitxs_line, fit_line)
ax.plot(fitxs_line, fitys_line, label="linear fit (no adjustment)")

# Spline to the points predicted by the two fits in the two regions (points adjusted manually in curved region)
spl_x = np.concatenate((fitxs_curve, fitxs_line))
spl_y = np.concatenate((fitys_curve, fitys_line))
spl = spline(spl_x, spl_y, k=3, s=4)
ax.plot(spl_x, spl(spl_x), 'black', lw=0, label="spline")
ax.legend(loc="best")
plt.savefig("Au_Thermal_Conductivity_Fit.png")


######################################################################################################################
#   Generate an expression for thermal conductivity of gold versus temperature over a larger range.
gold_k_temps = [600, 500, 400, 350, 300, 250, 200, 150, 100, 90, 80, 70]
gold_k_conds = [298, 304, 311, 314, 317, 321, 323, 325, 327, 328, 332, 348]
fig, ax = plt.subplots()
snp.labs("environmental temperature (K)", "$\kappa$ (W/m*k)", "Au Thermal Conductivity")
ax.plot(gold_k_temps, gold_k_conds, label="data", markersize=10)

# Fit the experimental conductivity in two pieces to capture two different shapes
fit_curve = np.polynomial.polynomial.polyfit(gold_k_temps[-4:], gold_k_conds[-4:], deg=5)
fit_line = np.polynomial.polynomial.polyfit(gold_k_temps[0:7], gold_k_conds[0:7], deg=3)

# Make predictions in the curved region using the fit, then adjust them by hand to capture the shape better
fitxs_curve = np.arange(70, 102, 2)
fitys_curve = np.polynomial.polynomial.polyval(fitxs_curve, fit_curve)
fitys_curve[6:10] = [331, 330.1, 329.3, 328.6]
fitys_curve[-4:] = [327.6, 327.4, 327.1, 327]
fitys_curve[:4] = [349, 345, 341, 337.6]
ax.plot(fitxs_curve, fitys_curve, label="5th degree fit (manually adjusted)")

# Make predictions in the linear region using the fit
fitxs_line = np.arange(100, 600, 10)
fitys_line = np.polynomial.polynomial.polyval(fitxs_line, fit_line)
ax.plot(fitxs_line, fitys_line, label="3rd degree fit (no adjustment)")

# Spline to the points predicted by the two fits in the two regions (points adjusted manually in curved region)
spl_x = np.concatenate((fitxs_curve, fitxs_line))
spl_y = np.concatenate((fitys_curve, fitys_line))
spl = spline(spl_x, spl_y, k=3, s=4)
ax.plot(spl_x, spl(spl_x), 'black', lw=0, label="spline")
ax.legend(loc="best")
plt.savefig("Au_Thermal_Conductivity_ExtendedRegion_Fit.png")


######################################################################################################################
#   Load my experimental data and plot against a modified version of xiang's model (short junction, realistic kappa)
# Get my 50x400 data
with open("means_and_mses_50x400_varseries.pickle", "rb") as myfile:
    mus, mses = pickle.load(myfile)
tenvs_data = mus.index.levels[1].values
pows = mus["crit_pow"].values/1000
intensity = pows / AREA  # In watts per meter squared
intensity_units = intensity * 10e-18 * 10e6  # In microwatts per nanometer squared

def compute_pow_shortjct(tcrit, tenv):
    """
    Calculates power per x-section area dissipated in NW given environmental temp & critical junction temp. This is the
    simple model outlined in Xiang2014 but modified to imagine that the wire outside of the junction is pinned at
    environmental temperature and so the temperature gradient occurs over only around 10 nm. Also, using a more
    realistic thermal conductivity from a spline of real experimental thermal conductivity data.

    :param tcrit: The critical temperature of the junction for EM initiation (in Kelvin) 
    :param tenv: The environmental temperature that the left side is pinned at (in Kelvin)
    :return: critical power dissipated in the junction needed to drive compute_tcrit given tenv (in W/m^2)
    """
    x = x = 0.3e-10 + 10e-9  # (in meters) NW length, depends on temp
    # k = -0.069 * (tcrit + tenv) / 2 + 338  # (in W/m*K) Thermal conductivity Au, also temp-dependent
    k = spl((tcrit + tenv)/2)  # A more realistic behavior of thermal conductivity with temp
    pow = k / x * (tcrit - tenv)  # (in W/m^2)
    return pow

# Compute and plot some predicted power vs. critical temperature curves with the constant critical temp assumption
tenvs = np.linspace(95, 260, 50)
tcrits = np.linspace(380, 580, 3)
fig, ax = plt.subplots()
snp.labs("Env. Temp. (K)", r"$\frac{P_c}{A}$ ($\mu W/nm^2$)", "Constant Critical Temp Model")
for tc in tcrits:
    ps = [compute_pow_shortjct(tc, tenv) * 10 ** -12 for tenv in tenvs]
    ax.plot(tenvs, ps, label="Jct Crit Temp = %.0f K" %(tc))
ax.plot(tenvs_data, intensity_units/2, label="data for 50x400 nm wires") # Plot my data on top


######################################################################################################################
#   Visualize a simple critical net mass flux model (influx from colder banks, outflux from hotter junction).
# Plot the next flux for fixed value of junction temperature and different "banks" temperature.
tj = 450
barrier = 450  # The energy-equivalent-temperature barrier in the exponential exp(-E/kT) i.e. barrier = E/k
tbs = np.arange(300, 400)
diff = np.exp(-barrier/tj) - np.exp(-barrier/tbs)
fig, ax = plt.subplots()
snp.labs("$T_{banks}$ (K)", "Net Mass Flux (arb. units)", "Net Mass Flux for Different $T_{banks}$")
ax.plot(tbs, diff, label="barrier = %i K, $T_j$ = %i K" % (barrier, tj))
ax.legend(loc="best")
plt.tight_layout()

# Plot the junction temperature needed to sustain a fixed critical flux for different bnaks temperatures
flux = 0.2
tjs_needed = -barrier/np.log(0.02 + np.exp(-barrier/tbs))
fig, ax = plt.subplots()
snp.labs("$T_{banks}$ (K)", "$T_j$ to sustain flux (K)", "$T_j$ to Sustain Net Mass Flux")
ax.plot(tbs, tjs_needed, label="barrier = %i, Flux = %.2f" % (barrier, flux))
ax.legend(loc="best")
plt.tight_layout()

# Plot a color map of net mass flux for different junction and bank temperatures
tjs = np.arange(300, 550, 5)
tbs = np.arange(200, 500, 5)[::-1]
tjs_grid, tbs_grid = np.meshgrid(tjs, tbs)
diff_grid = np.exp(-barrier/tjs_grid) - np.exp(-barrier/tbs_grid)
fig, ax = plt.subplots()
img = ax.imshow(diff_grid, extent=[300, 550, 200, 500])
snp.labs(r"$T_j$ (K)", r"$T_{banks}$ (K)", r"Net Mass Flux")
plt.colorbar(img)

# Add scatterplot points to the colormap where the flux is equal to some value of interest
success = np.abs(diff_grid.ravel() - flux) < 0.002
tjs_success = tjs_grid.ravel()[success]
tbs_success = tbs_grid.ravel()[success]
ax.scatter(tjs_success, tbs_success, label="Net Flux ~ %.2f" % (flux,))
ax.set_ylim([200, 500])
ax.legend(loc="best")


######################################################################################################################
#   Solve quasi-one-dimensional steady-state heat equation for different geometries with shooting method. Refer to
# https://web.stanford.edu/~fringer/teaching/numerical_methods_02/handouts/lecture9.pdf
def compute_area_areaprime(x):
    """
    Compute the area and it's derivative as a function of independent variable x, return them as a vector.
    
    :param x: independent variable, the domain of the problem is x=0 to L
    :return: a 2-vec holding [A, dA/dx].
    """
    # return [20/(x+2) - 0.166, -20/(x+1)**2]  # A steep funnel geometry
    # return [10/(x+1) - 0.59, -10/(x+1)**2]  # A funnel geometry
    # return [10-0.2*x, -0.2]  # A trapezoid geometry
    return [10, 0]  # Rectangle geometry


def compute_zprime(x, z, areafunction):
    """
    Compute the value of the vector z's derivative at a point given the value of the vector function z and the
    independent variable x. The form of this calculation is specified by the vector ODE. Return a vector for the
    derivative.
    
    :param x: indpendent variable, the domain of the problem is x=0 to L
    :param z: 2-vec of the variable representing system of equations, z = [y, y']
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :return: 2-vec of values for z' = f(x, z)
    """
    zprime_0 = z[1]
    area, areaprime = areafunction(x)
    zprime_1 = - areaprime/area * z[1]
    return [zprime_0, zprime_1]


def integrate_over_domain(z_at_0, integrator, areafunction, length=10, step=0.1, silent=True):
    """
    Call runge-kutta repeatedly to integrate the vector function z over the full domain (specified by length). Return
    a list of 2-vecs which are the value of the vector z at every point in the domain discretized by 'step'. Note that
    runge-kutta object calls x as "t" and z as "y".
    
    :param z_at_0: the value of the vector z=[y, y'] at the left boundary point. should be list or array.
    :param integrator: the runge-kutta numerical integrator object
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :param length: the length of the domain to integrate on
    :param step: the step size with which to discretize the domain
    :param silent: bool indicating whether to print the progress of the integrator
    :return: array of 2-vecs - the value of vector z obtained by integration for each point in the discretized domain.
    """
    initial_conditions = z_at_0
    integrator.set_initial_value(initial_conditions, t=0)  # Set the initial values of z and x
    integrator.set_f_params(areafunction)
    dt = step

    xs, zs = [], []
    while integrator.successful() and integrator.t <= length:
        integrator.integrate(integrator.t + dt)
        xs.append(integrator.t)
        zs.append([integrator.y[0], integrator.y[1]])
        if not silent:
            print("Current x and z values: ", integrator.t, integrator.y)
    return xs, zs


def solve_bvp_pj(y_at_0, yprime_at_length, areafunction, length=10, step=0.1, silent=True):
    """
    Numerically find the value for y'(0) that gives us a propagated (integrated) solution matching most closely with
    with other known boundary condition y'(L) which is proportional to the power dissipated by the junction.
    
    :param y_at_0: the known boundary condition on y for the left point.
    :param yprime_at_length: the known boundary condition on y' for the right point
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :param length: the length of the domain to integrate on
    :param step: the step size with which to discretize the domain
    :param silent: bool indicating whether to print the progress of the integrator
    :return: the optimized estimate of y' at the left boundary point giving the most accurate integrated solution.
    """
    integrator = ode(compute_zprime).set_integrator("dopri5")
    z_at_0 = [y_at_0, 0.5]  # Make an initial guess for yprime at x=0

    def residuals(yprime_at_0, y_at_0, yprime_at_length):

        # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
        z_at_0 = [y_at_0, yprime_at_0]
        xs, zs = integrate_over_domain(z_at_0, integrator, areafunction)
        yprime_at_length_integrated = np.array(zs)[-1, 1]

        # Return the difference between y'(L) found by numerical integrator and the true value
        return yprime_at_length - yprime_at_length_integrated

    yprime_at_0_guess = 0.2
    lsq = least_squares(residuals, yprime_at_0_guess, args=(y_at_0, yprime_at_length), loss="soft_l1")
    yprime_at_0_estimate = lsq.x[0]
    return yprime_at_0_estimate


def solve_bvp_tj(y_at_0, y_at_length, areafunction, length=10, step=0.1, silent=True):
    """
    Numerically find the value for y'(0) that gives us a propagated (integrated) solution matching most closely with
    with other known boundary condition y(L) which is proportional junction temperature.

    :param y_at_0: the known boundary condition y(0) for the left point.
    :param y_at_length: the known boundary condition y(L) for the right point
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :param length: the length of the domain to integrate on
    :param step: the step size with which to discretize the domain
    :param silent: bool indicating whether to print the progress of the integrator
    :return: the optimized estimate of y' at the left boundary point giving the most accurate integrated solution.
    """
    integrator = ode(compute_zprime).set_integrator("dopri5")
    z_at_0 = [y_at_0, 0.5]  # Make an initial guess for yprime at x=0

    def residuals(yprime_at_0, y_at_0, y_at_length):
        # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
        z_at_0 = [y_at_0, yprime_at_0]
        xs, zs = integrate_over_domain(z_at_0, integrator, areafunction)
        y_at_length_integrated = np.array(zs)[-1, 0]

        # Return the difference between y'(L) found by numerical integrator and the true value
        return y_at_length - y_at_length_integrated

    yprime_at_0_guess = 0.2
    lsq = least_squares(residuals, yprime_at_0_guess, args=(y_at_0, y_at_length), loss="soft_l1")
    yprime_at_0_estimate = lsq.x[0]
    return yprime_at_0_estimate


######################################################################################################################
#   Solve heat equation for fixed power with different geometries and plot.
areafuncs = {"funnel1": lambda x: [20/(x+2) - 0.166, -20/(x+1)**2],
             "funnel2": lambda x: [10/(x+1) - 0.59, -10/(x+1)**2],
             "trapezoid": lambda x: [10-0.2*x, -0.2],
             "rectangle": lambda x: [10, 0]
             }
integrator = ode(compute_zprime).set_integrator("dopri5")
yprime_at_length = 10
y_at_0 = 200
fig, ax = plt.subplots()
snp.labs("x (nm)", "T (K)", "Quasi-1D, Steady State, Heat Eqn")
for nm, areaf in areafuncs.items():
    integrator.set_initial_value([y_at_0, 0.5], t=0)  # Set the initial values of z and x
    integrator.set_f_params(areaf)
    yprime_at_0_estimate = solve_bvp_pj(y_at_0, yprime_at_length, areaf)
    xs, zs = integrate_over_domain([y_at_0, yprime_at_0_estimate], integrator, areaf)
    ax.plot(xs, np.array(zs)[:,0], label=nm)

ax.legend(loc="best")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.95, "$y'(L)$ = %.2f" % (yprime_at_length,), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.tight_layout()
plt.savefig("temp_vs_x_diff_geometries.png")



######################################################################################################################
#   Solve heat equation for fixed junction temperature and different environmental temperatures and plot.
integrator = ode(compute_zprime).set_integrator("dopri5")
y_at_length = 300
fig, ax = plt.subplots()
snp.labs("$T_E$ (K)", "$p_c$ (W/m*K)", "Quasi-1D, Steady State, Heat Eqn")
tenvs = np.arange(100, 260, 10)
for nm, areaf in areafuncs.items():
    pjs = []
    for y_at_0 in tenvs:
        integrator.set_initial_value([y_at_0, 0.5], t=0)  # Set the initial values of z and x
        integrator.set_f_params(areaf)
        yprime_at_0_estimate = solve_bvp_tj(y_at_0, y_at_length, areaf)
        xs, zs = integrate_over_domain([y_at_0, yprime_at_0_estimate], integrator, areaf)
        pjs.append(np.array(zs)[-1,1])
    ax.plot(tenvs, pjs, label=nm)

ax.legend(loc="best")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.95, "$T_j$ = %.0f" % (y_at_length,), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.tight_layout()
plt.savefig("pc_vs_tenv_diff_geometries.png")


######################################################################################################################
#   Implement a critical net flux model and plot fluxes for different values of barrier height and environmental temp
def net_flux(barrier, tj, tbank):
    """
    Compute the rate of net outflux of atoms from the junction which is being fed atoms by the adjacent hot banks.
    Assumes the same barrier height for migration from the banks and from the junction.
    
    :param barrier: energey barrier to atomic migration in units of equivalent temperature
    :param tj: temperature of junction
    :param tbank: temperature of adjacent hot bank
    :return: net flux in arbitrary units, where a negative number imples the net flux is in to the junction
    """
    return np.exp(-barrier/tj) - np.exp(-barrier/tbank)

barriers = np.linspace(10, 500, 10)
tbanks = np.linspace(400, 450, 20)
tj = 450
fig, ax = plt.subplots()
snp.labs("$T_{banks}$ (K)", "Net Flux (arb. units)", "Net Flux for Different Barriers")
for barr in barriers:
    fluxes = [net_flux(barr, tj, tbank) for tbank in tbanks]
    ax.plot(tbanks, fluxes, label="barrier = %i K" % (barr,))
ax.legend(loc="best")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.4, 0.95, "$T_j$ = %i K" % (tj,), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig("flux_vs_tenv_differentbarriers.png")


######################################################################################################################
#   Implement a critical net flux model solver which finds power needed to sustain a given flux for some environmental
# temperature, area geometry
def solve_flux(y_at_0, netflux, areafunction, barrier, deltasteps=10, length=10, step=0.1, silent=True):
    """
    Compute the power needed to drive a fixed critical net flux given some environmental temperature. 
    
    :param y_at_0: the known boundary condition y(0) for the left point.
    :param areafunction: a function that takes x and gives back a 2-vec [A(x), dA(x)/dx]
    :param deltasteps: the number of discretized steps between the junction and the bank
    :return: value of y'(0) for the optimized solution to y(x)
    """
    integrator = ode(compute_zprime).set_integrator("dopri5")
    z_at_0 = [y_at_0, 10]  # Make an initial guess for yprime at x=0

    def residuals(yprime_at_0, y_at_0, y_at_length):
        # Use RK to compute [y, y'] over the domain given the values for y, y' at the boundary
        z_at_0 = [y_at_0, yprime_at_0]
        xs, zs = integrate_over_domain(z_at_0, integrator, areafunction)
        y_at_bank = np.array(zs)[-deltasteps, 0]
        y_at_jct = np.array(zs)[-1, 0]
        netflux_estimate = net_flux(barrier, y_at_jct, y_at_bank)

        # Return the difference between y'(L) found by numerical integrator and the true value
        # return abs(netflux - netflux_estimate) + y_at_0  # Regularize to get minimum temp
        return netflux - netflux_estimate # Regularize to get minimum temp

    yprime_at_0_guess = 0.2
    lsq = least_squares(residuals, yprime_at_0_guess, args=(y_at_0, y_at_length), loss="soft_l1")
    yprime_at_0_estimate = lsq.x[0]
    return yprime_at_0_estimate



######################################################################################################################
#   Solve critical flux model for different parameters
barrier = 200
targetflux = 0.1
deltasteps = 20  # Number of steps between junction and banks (step size default is 0.1)
areafuncs = {"funnel1": lambda x: [20/(x+2) - 0.166, -20/(x+1)**2],
             "funnel2": lambda x: [10/(x+1) - 0.59, -10/(x+1)**2],
             "trapezoid": lambda x: [10-0.2*x, -0.2],
             "rectangle": lambda x: [10, 0]
             }
fig, ax = plt.subplots()
snp.labs("$T_E$ (K)", "$p_c$ (arb. units)", "Power Needed to Drive Net Flux")
tenvs = np.arange(100, 260, 10)
for nm, areaf in areafuncs.items():
    pjs = []
    for y_at_0 in tenvs:
        yprime_at_0_estimate = solve_flux(y_at_0, targetflux, areaf, barrier, deltasteps)
        xs, zs = integrate_over_domain([y_at_0, yprime_at_0_estimate], integrator, areaf)
        pjs.append(np.array(zs)[-1, 1])
    ax.plot(tenvs, pjs, label=nm)

ax.legend(loc="best")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.95, "net flux = %.2f \nbarrier = %i \ndeltasteps = %i" % (targetflux, barrier, deltasteps), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.tight_layout()
plt.savefig("power_vs_tenv_netflux_%.2f_%i_%i.png" % (targetflux, barrier, deltasteps))
