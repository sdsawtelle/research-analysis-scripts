"""  Thermal Model for Electromigration (EM) in Nanowires (NW) as in Xiang, APL, 2014.
"""

__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import snips as snp
snp.prettyplot(matplotlib)


def pow(tcrit, tenv):
    '''Calculates power per x-section area dissipated in NW given environemental temp & critical junction temp'''
    x = 3e-9*(tcrit + tenv)/2+ 1.1e-6  # NW length, depends on temp
    # x = 3e-9*tenv+ 1.1e-6  # NW length, depends on tenv
    k = -0.069*(tcrit + tenv)/2 + 338  # Thermal conductivity Au, also temp-dependent
    pow = k/x * (tcrit - tenv)
    return pow

# Compute and plot some predicted power vs. critical temperature curves
tenvs = np.linspace(100, 300, 50)
tcrits = np.linspace(300, 500, 5)

fig, ax = plt.subplots()
snp.labs("environmental temperature (K)", "critical power", "Constant Critical Temp Model")
for tcrit in tcrits:
    ps = [pow(tcrit, tenv) for tenv in tenvs]
    ax.plot(tenvs, ps, label="Jct Crit Temp = %.0f K" %(tcrit))
ax.legend(loc="upper right")