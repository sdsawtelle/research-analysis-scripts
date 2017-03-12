

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import pickle
import os
import scipy
from scipy.optimize import least_squares
import datetime
import operator


import snips as snp
snp.prettyplot(matplotlib)





def plt_sweep(fname):
    df = pd.read_csv(fname, header=3)
    if len(df.columns) == 2:
        df.columns = ["voltage","current"]
    else:
        df.columns = ["voltage", "current", "r"]
    fig, ax = snp.newfig()
    snp.labs("Voltage (V)", "Current (nA)", "IV Sweep")
    ax.plot(df["voltage"], df["current"]*10**9, linestyle="-", marker="o", markersize=6, label=fname.strip(".txt"))
    ax.legend(loc="upper left")
    plt.savefig(fname.strip(".txt") + ".png")
    return fig



def plt_sample(fname, interval=1.5):
    df = pd.read_csv(fname, header=3)
    if len(df.columns) == 2:
        df.columns = ["voltage","current"]
    else:
        df.columns = ["voltage", "current", "r"]

    fig, ax = snp.newfig()
    snp.labs("Time (sec)", "Current (nA)", "Const V, Sample I")
    ax.plot(df.index*interval, df["current"]*10**9, linestyle="-", marker="o", markersize=6, label=fname.strip(".txt"), color="red")
    ax.legend(loc="upper left")
    plt.savefig(fname.strip(".txt") + ".png")
    return fig



path = "C:/Users/Sonya/Documents/My Box Files/CBIF_Project/161104_CBIF_glass_dev8_plasma"
os.chdir(path)
#######################################################################
############ PLOTTING ALL ACTUAL RESISTANCE TRACES ############
#######################################################################
fnames = snp.txtfilenames(path)
for fname in fnames:
    plt_sample(fname, interval=1)

for fname in fnames:
    plt_sweep(fname)

fnames = snp.txtfilenames(path)
mt = {}
for fname in fnames:
    mt[fname] = os.path.getmtime(fname)
fnames = sorted(mt.items(), key=operator.itemgetter(1))
fnames = [name for name, mt in fnames]

fig, ax = snp.newfig()
j = 0;
for fname in fnames:
    df = pd.read_csv(fname, header=3)
    if len(df.columns) == 2:
        df.columns = ["voltage","current"]
    else:
        df.columns = ["voltage", "current", "r"]

    labs = fname.strip(".txt").split("_")[3:5]
    lab = "_".join(labs)
    snp.labs("Time (sec)", "Current (nA)", "Const V, Sample I")
    ax.plot(df.index*1.5, df["current"]*10**9, linestyle="-", marker="o", markersize=6, label=lab)
    j += 1

ax.legend(loc='upper right', bbox_to_anchor=(1, 0.84))
ax.legend(loc="upper center")

plt.savefig("Sample.png")






#######################################################################
############ PLOTTING ALL IV SWEEPS ############
#######################################################################
path = "C:/Users/Sonya/Documents/My Box Files/CBIF_Project/161215_CBIF_GlassDev11/combo_am_with_after_no_v"
os.chdir(path)
fnames = snp.txtfilenames(path)
mt = {}
for fname in fnames:
    mt[fname] = os.path.getmtime(fname)
fnames = sorted(mt.items(), key=operator.itemgetter(1))
fnames = [name for name, mt in fnames]



fig, ax = snp.newfig()
NUM_COLORS = len(fnames)
cm = plt.get_cmap('gist_rainbow')
ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

for fname in fnames:
    df = pd.read_csv(fname, header=2)
    if len(df.columns) == 2:
        df.columns = ["voltage","current"]
    else:
        df.columns = ["voltage", "current", "r"]
    snp.labs("|Voltage| (V)", "|Current| (nA)", "IV Sweep")
    ax.plot(np.abs(df["voltage"]), np.abs(df["current"]*10**9), linestyle="-", marker="o", markersize=6, label=fname.strip(".txt"))
# ax.legend(loc='upper left', bbox_to_anchor=(1, 0.84))
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.legend(loc="upper left")
plt.savefig("All_Sweeps_Combined.png")





#######################################################################
############ PLOTTING NOISE ############
#######################################################################
path = "C:/Users/Sonya/Documents/My Box Files/CBIF_Project/161215_CBIF_GlassDev11/combo_am_with_after_no_v"
os.chdir(path)
fnames = snp.txtfilenames(path)
mt = {}
for fname in fnames:
    mt[fname] = os.path.getmtime(fname)
fnames_sorted = sorted(mt.items(), key=operator.itemgetter(1))
fnames = [name for name, mt in fnames_sorted]
times = [datetime.datetime.fromtimestamp(mt) for name, mt in fnames_sorted]
elaps = [abs((time - times[0]).total_seconds())/3660 for time in times]



fig, ax = snp.newfig()
snp.labs("Elapsed Time (hrs)", "Normalized |Current| (nA)", "Normalized Noise / Drift")

vrange = np.arange(-0.38, -0.8, -0.1)
NUM_COLORS = len(vrange)
cm = plt.get_cmap('gist_rainbow')
ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

df = pd.read_csv(fnames[0], header=2)
if len(df.columns) == 2:
    df.columns = ["voltage","current"]
else:
    df.columns = ["voltage", "current", "r"]
df_all = pd.DataFrame(index=fnames, columns=([str(v) for v in df["voltage"].values]))

for idx, fname in enumerate(fnames):
    df = pd.read_csv(fname, header=2)
    if len(df.columns) == 2:
        df.columns = ["voltage","current"]
    else:
        df.columns = ["voltage", "current", "r"]

    df_all.loc[fname, :] = df["current"].values

for v in vrange:
    vals = np.abs(df_all[str(v)].values)
    vals = vals/vals[0]
    ax.plot(elaps, vals, linestyle="-", marker="o", markersize=6, label="V = %s" %(str(v),))

ax.legend(loc="lower center")
plt.savefig("All_Sweeps_Noise_and_Drift.png")




#######################################################################
############ PLOTTING ALL CURRENT TIME TRACES ############
#######################################################################
path = "C:/Users/Sonya/Documents/My Box Files/CBIF_Project/161118_CBIF_GlassDev22/"
os.chdir(path)
fnames = snp.txtfilenames(path)
mt = {}
for fname in fnames:
    mt[fname] = os.path.getmtime(fname)
fnames = sorted(mt.items(), key=operator.itemgetter(1))
fnames = [name for name, mt in fnames]

fig, ax = snp.newfig()
NUM_COLORS = len(fnames)
cm = plt.get_cmap('gist_rainbow')
ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

for fname in fnames:
    mt = os.path.getmtime(fname)

for fname in fnames:
    df = pd.read_csv(fname, header=2)
    if len(df.columns) == 2:
        df.columns = ["voltage","current"]
    else:
        df.columns = ["voltage", "current", "r"]
    v = df.loc[0, "voltage"]
    snp.labs("Time (s)", "|Current| (nA)", "Sampling Continuously at %.2f V" %v)
    ax.plot(df.index*1, np.abs(df["current"]*10**9), linestyle="-", marker="o", markersize=6, label=fname.strip(".txt"))
plt.savefig("All_Samples_Combined.png")









#######################################################################
############ PLOTTING CALIBRATION CURVES ############
#######################################################################

path = "C:/Users/Sonya/Documents/My Box Files/CBIF_Project/161118_CBIF_GlassDev22/flow_n2_2"
os.chdir(path)
fnames = snp.txtfilenames(path)

df = pd.read_csv("optical_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
probe_vals = df["Dissolved Oxygen (mg/L)"].values
probe_times = df["Timestamp"]

mtimes = {}
o2_conc = {}
for fname in fnames:
    mt = os.path.getmtime(fname)
    mtimes[fname] = datetime.datetime.fromtimestamp(mt)
    deltas = [abs((mtimes[fname] - pt).total_seconds()) for pt in probe_times]
    ind = np.argmin(deltas)
    o2_conc[fname] = probe_vals[ind]


# relevant_o2 = df[df["Timestamp"] > min(mtimes.values()) & df["Timestamp"] < max(mtimes.values())]

fig, ax = snp.newfig()
snp.labs("Time Point", "[O2] (mg/L)", "Optical Probe Readings During Flow")
ax.plot(probe_vals)
plt.savefig("Optical_Probe_Readings.png")



fnames, o2 = list(o2_conc.keys()), list(o2_conc.values())

fig, ax = snp.newfig()
snp.labs("[O2] (mg/L)", "|Current| (nA)", "Rough Calibration Curve")

vpoints = np.arange(35, 40, 1)
NUM_COLORS = len(vpoints)
cm = plt.get_cmap('gist_rainbow')
ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

for i in vpoints:
    currs = []
    for fname in fnames:
        df = pd.read_csv(fname, header=3)
        if len(df.columns) == 2:
            df.columns = ["voltage", "current"]
        else:
            df.columns = ["voltage", "current", "r"]
        currs.append(df.loc[i, "current"] * 10 ** 9)

    sens = 10**3*(max(currs) - min(currs))/((max(o2) - min(o2))/0.0469)
    ax.plot(o2, np.abs(currs), marker="o", markersize=6, label="V = %.2f; S = %.2f pA/mmHg"%(df.loc[i, "voltage"], sens), linestyle="None")

    # ax.legend(loc='upper left', bbox_to_anchor=(1, 0.84))
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc="upper left")
    plt.savefig("Calibration.png")


















# ax.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k', "b", "r", "g", "orange", "purple", "pink"]))


    # fig, ax = plt.subplots(2, 1)
    # snp.labs("Time (min)", "|Current| ($\mu$A)", "O2 Sweep", ax = ax[0])
    # ax[0].plot((df.index-10)*0.101, abs(df["current"]*10**6), linestyle="", marker="o", markersize=6, label="sensor")
    # ax[0].legend(loc="lower right")
    # ax[0].set_title("$O_2$ Sweep (htim = 62 ms, integration = 4 PLC)")
    #
    # # ax.vlines([12.5*10.14/12, 13.25*10.14/12, 14.83*10.14/12, 16.91*10.14/12], *ax.get_ylim())
    # # ax.vlines([4*10.14/12], *ax.get_ylim())
    #
    # plt.savefig("O2_Sweep_" + fname.strip(".txt").split("_")[0] + "_" + "_".join(params)+".png")
    # plt.close(fig)

    # 0.101 min / sample
# 9.907 samples/ min


    fig, ax = snp.newfig()
    snp.labs("Time (sec)", "Current (nA)", "Dev6 Across Setup")
    ax.plot(df.index*1.5, df["current"]*10**9, linestyle="-", marker="o", markersize=6, label=fname.strip(".txt"))
    plt.savefig(fname.strip(".txt") + ".png")

    curr = df["current"].as_matrix()
    len(curr)

    curr = curr[0:88000]
    curr = curr.reshape(22, 4000)

    avgcurr = np.apply_along_axis(np.mean, axis=0, arr=curr)
    time = np.arange(0, 4000*33, 33)

    fig, ax = snp.newfig()
    snp.labs("Time (sec)", "Current (nA)", "Dev6 Across Setup")
    ax.plot(time, avgcurr*10**9, linestyle="-", marker="o", markersize=6, label="binned_by_22_points")
    plt.savefig(fname.strip(".txt") + "_Smoothed_.png")




    l = [5.84, 5.8, 5.76, 5.71, 5.63, 5.59, 5.50, 5.42, 5.3, 5.2, 5.13, 5.11, 5.10, 5.05, 4.96, 4.9, 4.88,
         4.92, 5.03, 5.15, 5.3, 5.46, 5.6, 5.73, 5.87, 5.94, 6.14, 6.09, 6.05, 6.01, 5.97, 5.93, 5.88, 5.83, 5.79,
         5.41, 5.06, 4.68, 4.49, 4.44, 4.42, 4.39, 4.40, 4.5, 4.64, 4.8, 4.99, 5.13, 5.29, 5.44, 5.58, 5.74, 5.87]

fig, ax = snp.newfig()
snp.labs("Time (min)", "[$O_2$] ($\frac{mg}{L}$)", "[O2] Sweep - " + param_str)
fig.add_subplot(2, 1, 2)
ax[1].plot(l, linestyle="", marker="o", markersize=6, color="red", label="optical probe")
ax[1].legend(loc="lower right")
ax[1].set_xlabel("Time (min)")
ax[1].set_ylabel("[$O_2$] (mg/L)")





def read_optical(fname):
    df = pd.read_csv(fname, header=1, delim_whitespace=True)
    df.columns = ["time", "do", "a", "b", "c", "d"]
    o2 = df["do"].apply(float)

fig, ax = snp.newfig()
ax.plot(o2)
ax.set_ylim([np.min(o2)-0.01, np.max(o2)+0.01])




def combo_sensor_probe(fname_stem):
    sensor = fname_stem + ".txt"
    df_sens = pd.read_csv(sensor, header=1)
    df_sens.columns = ["voltage","current"]

    optical = fname_stem + ".csv"
    df_opt = pd.read_csv(optical, header=1)
    df_opt.columns = ["time", "do", "a", "b", "c", "d"]
    df_opt["time"] = pd.to_datetime(df_opt["time"])
    df_opt["do"] = df_opt["do"].apply(float)
    span = df_opt.iloc[-1, 0] - df_opt.iloc[0, 0]
    secs = span.total_seconds()

    sens_x = df_sens.index*(secs/len(df_sens))
    opt_x = df_opt.index*(secs/len(df_opt))

    fig, axs = plt.subplots(2, 1, figsize=[12, 9], sharex=True)
    fig.suptitle("Sensor and Optical Probe Response")

    axs[0].plot(sens_x, abs(df_sens["current"]*10**6))


    axs[1].plot(opt_x, df_opt["do"])



#########################################
#########################################
############################# Calibrating Cernox Resistor #########################################

import scipy.io
mat = scipy.io.loadmat("Cernox_Calibration_2016.10.20.mat")
res = mat["R"]
temp = mat["Temp"]

with open("temp.txt", "w") as myfile:
    for i, rest in enumerate(res):
        myfile.write("Point %i: %f,%f\n" %(i+1, np.log10(rest), temp[i]))

temp = [elem[0] for elem in temp]
res = [elem[0] for elem in res]


spl_temp = temp[-500:]
spl_res = res[-500:]



from scipy.interpolate import UnivariateSpline
myspl=UnivariateSpline(spl_temp, spl_res)

myfit = np.polyfit(spl_temp, spl_res, deg=3)
p = np.poly1d(myfit)
plt.plot(spl_temp, spl_res, color="red", markersize=7, linestyle="None", label="data")
plt.plot(spl_temp, p(spl_temp), color="blue", label="cubic fit")


newtemps = np.linspace(94.62, 91.62, 300)
newres = p(newtemps)
plt.plot(newtemps, newres, color="green", markersize=5, linestyle="None", label="extrapolated")
snp.labs("Temp (K)", "Res ($\Omega$)", "Extrapolated Cernox Resistance")
plt.legend(loc="upper right")

with open("temp_extrapolate.txt", "w") as myfile:
    for i, rest in enumerate(newres):
        myfile.write("Point %i: %f,%f\n" %(i+4555, np.log10(rest), newtemps[i]))

res_combo = res + list(newres)
temp_combo = temp + list(newtemps)

r = res_combo[0::25]
t = temp_combo[0::25]
with open("temp_combined.txt", "w") as myfile:
    for i, rest in enumerate(r):
        myfile.write("Point %i: %f,%f\n" %(i+1, np.log10(rest), t[i]))