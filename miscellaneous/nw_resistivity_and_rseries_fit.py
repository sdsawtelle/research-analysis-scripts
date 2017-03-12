"""  Here we plot measured resistances as a function of wire length and diameter in order to extract out both the common
series resistance and to fit a value of resistance per unit length for each diameter. This was written for chip 5-10.
"""

__author__ = 'Sonya'
__email__ = "sdsawtelle@gmail.com"
__version__ = "3.5"

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import snips as snp

path = "C:/Users/Sonya/Documents/My Box Files/MolT Project/160730_SDS20_Chip_5-10/"
os.chdir(path)
snp.prettyplot(matplotlib)  # Make a nicer default plot style
plt.close("all")
matplotlib.interactive(True)

# pd.read_table reads a general delimited file rather than expecting a csv - it defaults to sep= tab. If there is no
# header then you explicitly pass that in order to capture starting with the first line.
df = pd.read_table("PS_Resistances.txt", header=None)
df.columns=["devID", "length", "diam", "curr"]
df["res"] = 0.045/(df.curr*1e-6)

# There are three different diameters so I will plot them each as their own color
colors = ["blue", "green", "red"]
fig, ax = snp.newfig("ul")

# Initialize a DF to hold the relevant statistics for each D,L grouping
idx = [df["diam"].unique(), ["mean", "stdev"]]
multindx = pd.MultiIndex.from_product(idx, names=["diam", "stat"])
df_avg_res = pd.DataFrame(columns=multindx, index=df.length.unique())
df_avg_res = df_avg_res.sort_index()  # This is necessary to do proper slicing with multiindexed dataframes

# Initialize a DF to hold the fit statistics
df_fits = pd.DataFrame(columns=["slope", "intercept"], index=df.diam.unique())


# Plot all devices
for dia in df.diam.unique():
    df_diam = df.loc[df.diam == dia]
    diacolor = colors.pop()
    ax.plot(df_diam.length, df_diam.res, label="Diameter = " + str(dia) + " nm", color=diacolor, linestyle="None", marker="o", markersize=6)

    # Calculate and store the mean and variance of resistance for each unique length in this diameter sub-dataframe
    for length in df_diam.length.unique():
        values = df_diam.res.loc[df_diam.length == length]
        values = values[values < 1400]  # Restrict to only non-outlier points
        avg = np.mean(values)
        var = np.var(values)

        # Note! df_avg_res[dia]["mean"].loc[length] just returns a copy of that slice of the DF so it is not sufficient
        # to assign a value there. The set_value avoids this - it takes in the index (label) and the value to be set.
        # But note even df_avg_res[dia]["mean"].set_value(label=length, value=avg) doesn't work b/c the first part
        # [dia]["mean"] gets evaluated and gives us a copy. We would need a full set_value with multindexing. However
        # the recommended way is to use .loc appropriately with multindex and avoid what is called "Chained indexing".
        df_avg_res.loc[length, (dia, "mean")] = avg  # Notice how the column multiindex must be handled as a tuple.
        df_avg_res.loc[length, (dia, "stdev")] = var**(1/2)

    lengths = df_avg_res.index
    res = df_avg_res.loc[:, (dia, "mean")]
    ax.plot(lengths, res, label="Averaged R", color=diacolor, marker='+', markersize=15, markeredgewidth=5)

    # Do a linear LSQ fit to the averages and plot it
    # Recall the 1500 nm length 20 nm diameter wires have a different real diameter than the other "20 nm" wires. We
    # will exclude them from the fit.
    if dia==20:
        res=res[lengths < 1500]
        lengths = lengths[lengths < 1500]
    slope, intercept = np.polyfit(lengths, res, deg=1, rcond=None, full=False, w=None, cov=False)
    df_fits.loc[dia, "slope"] = slope
    df_fits.loc[dia, "intercept"] = intercept
    fitline = np.poly1d([slope, intercept])
    ax.plot(lengths, fitline(lengths), color=diacolor, linestyle="--",
            label="Fit (m=%0.3f, b=%0.2f)" % (slope, intercept), marker=None)

snp.labs("Length (nm)", "Resistance @ 45 mV ($\Omega$)", "Resistance by Geometry")
leg = ax.legend(loc="upper right", numpoints=1)  # numpoints controls num of markers on the legend illustration (default two)
ax.set_ylim([0, 7171])
ax.set_xlim([-25, 1600])

# Zoom in to non-outliers.
leg = ax.legend(loc="upper left", numpoints=1)  # numpoints controls num of markers on the legend illustration (default two)
ax.set_ylim([258.68331599028824, 1397.2745227450835])

with open('res_plot_by_geom.pickle', 'wb') as myfile:
    pickle.dump(fig, myfile)
# To see the file next time:
# with open('res_plot_by_geom.pickle', 'rb') as myfile:
#     fig = pickle.load(myfile)
# fig.show()


print("debug")