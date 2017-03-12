# em_const_pow_model

## Overview
This module provides functionality to plot, prepare and fit standard EM traces within the constant-power model for electromigration. The user is able to restrict what parts of the trace should be included in analysis to exclude, for instance, later ramp cycles once the junction resistance is high and the device is unstable. Once the trace has been culled in this fashion, the analysis consists of identifying the current and voltage points where EM occurs, and then fitting that set of points using a constant critical power model.

## Mathematical Model 
Controlled EM traces are generally well described by the "Critical Power Model", which holds that while the junction breaks to increasing resistance in a controlled way EM is initiated at the same critical power in the junction. The underlying idea is that the migration of atoms is a thermally activated process, and if the thermal conductance of the junction structure remains roughly constant then a constant critical power, $p_c$, implies a constant critical temperature, $T_c$, at which EM is initiated. 

The physical model of the system is two lead-in structures of fixed series resistance contacting a junction region which progressively necks down during EM and thus presents an increasing resistance. We neglect any dependence of resistance on current via current-induced heating for both the series resistance and the junction. 

We use the following definitions and conventions:

- $R_0^T$ is the initial total resistance of the device
- $R^L$ is the fixed total series resistance
- $R_0^J$ is the initial resistance of the junction region (which will be small as it is only a few nanometers long)
- We index the ramp cycles by $n$, starting with $n=1$. The very end of each ramp cycle is a critical point where EM was inititiated, and we denote quantities measured at the critical point of the $n^{th}$ ramp cycle by subscript $n$, like $R_n^T$ and $R_n^J$.
- The critical power in the junction at which EM is initiated at the end $n^{th}$ ramp is $P_n^J$.
- The total voltage dropped across the full system, and the current through the system at the critical point at the end of the $n^{th}$ ramp are $V_n$ and $I_n$.

Note that $R_0^T$, $R_n^T$, and  $V_n, I_n \forall n$ are directly measured by us during a controlled EM trace.

As EM proceeds, the series resistance stays fixed while the junction resistance grows so we can write
\begin{align*}
R_n^J = R_n^T - R^L = R_n^T - (R_0^T - R_0^J).\\
\end{align*}

The critical power model holds that $P_n^J = P^*$ for some constant $P^*$. We have:
\begin{align*}
P_n^J = I_n^2R_n^J = I_n^2(R_n^T - R^L) = P^*\\
\implies I_n = \bigg(P^* / (R_n^T - R^L)\bigg)^{1/2}.
\end{align*}

This constitutes a mathematical model for the empirically observed critical currents, $I_n$, as a function of the measured total resistance at critical points, $R_n^T$, with an empirically observed parameter $R_0^T$ and two free parameters $R^L$ and $P^*$. A fitted value of $R^L$ implies a fitted value of $R_0^J$ since $R_0^T = R_0^J + R^L$. This is useful because independent studies which measure the resistance of nanowires with fixed diameter as a function of length can provide a good sanity check on whether fitted values of the initial junction resistance (which is a few-nanometer long section of a nanowire) are reasonable. 

This formulation of the model in terms of $I_n$ as a function of $R_n^T$ is preferred for visualization because we want to check for changing behavior as the breaking proceeds and $R_n^T$ grows. However, the raw data that we actually obtain are the critical points $(V_n, I_n)$ and so for the purposes of fitting the data we can recast the above in terms of these variables by using $R_n^T = V_n/I_n$:
\begin{align*}
V_n = P^*/I_n + I_nR^L.
\end{align*}

\subsection{Variable Temperature Electromigration}
We have conducted variable temperature experiments where we execute electromigration of a device to some small value of $R_n^J$ at environmental temperature, $T_1$,  and then we complete the electromigration at a different temperature $T_2$. Our aim is to determine how the critical power changes as a function of ambient temperature. For these experiments we fit the initial EM trace at $T_1$ in the usual way, to obtain estimates of the two free parameters $R_L$ and $P^*_1$. However, for fitting the final EM trace at $T_2$ we no longer treat $R_L$ as free but rather we tether it to the value estimated from the initial trace. We then just fit a single free parameter $P^*_2$. Because the environmental temperature can significantly affect the series resistance, in fitting the final EM trace we tether the series resistance to an adjusted value: $R_L(T_2) =\alpha R_L(T_1)$. The coefficient $\alpha$ is estimated by comparing IV curves taken at $T_1$ after initial EM is complete with IV curves taken at $T_2$ before proceeding with the final EM. The IV curves sweep to the highest voltage that is safely outside the zone of EM onset, and the last ten steps of each IV is used to calculate $\alpha$ as the average of the ratio of resistances $\alpha = \langle R(T_1) / R(T_2) \rangle$.
\begin{align*}
I_n(T_1) = \bigg(P^*(T_1) / (R_n^T(T_1) - R^L(T_1))\bigg)^{1/2}\\,
I_n(T_2) = \bigg(P^*(T_2) / (R_n^T(T_2) - \alpha R^L(T_1))\bigg)^{1/2}.\\
\end{align*}


## Extracting Model Quantities from Traces
The empirical parameter $R_0^{tot}$ is set by the user while viewing the RV plot. The user should choose a point in the first reasonable ramp cycle that occurs close to the critical voltage, but before any significant curvature presents itself (the hallmark of EM beginning).

The critical points are taken as simply the last $(I, V)$ pair in each valid ramp cycle (the user is able to select which ramp cycles are valid, see next section). 

## Usage for Single Temperature Electromigration
Much of my data consists of a single IV trace for a device, taken at one fixed temperature. In this case the relevant module is `em_pc_prep_and_fit.py`.

Specific usage is to first call pic_all() to make plots of all traces, so that crappy ones can be removed from the folder. Then call prep_all() to do all initial preparation of the good traces like selecting ramps for analysis and setting res_tot_0. This creates and pickles a dict of dicts whose keys are the trace names and whose values are a dict of the prepared DF and the res_tot_0 value. Optionally now you can call revise_prep() if you want to revise the preparation of any specific trace - it will update the pickled dictionary.

Now call fit_all() with kwargs rjmin and rjmax to fit the ramps between rjmin and rjmax for each trace to the constant power model. This function creates and pickles a dataframe whose indices are the trace names and whose values are various demographic info about the device as well as the fit results. It also creates and pickles a dict of dicts with the same information. The pickled dataframe has a column 'conforms' which denotes whether the device appears to conform to the constant power model or not. This is initialized to 1 for all traces but values can later be changed by calling set_pc_conforms(fname). The dict and df pickles are named with the specific rjmin and rjmax values that were used for the fitting, so users can try different values of these kwargs and not overwrite previous results.

# Usage for Variable Temperature Electromigration
In some cases we have performed electromigration of a device to some small value of $R_n^J$ at environmental temperature, $T_1$, and then completed the electromigration at a different temperature $T_2$. The aim is to determine how the critical power changes as a function of ambient temperature.

The data consists of, for each device, an EM trace breaking the junction from 0 to 20 ohms at $T_1$ and then a second EM trace breaking the junction from 20 to 120 ohms at $T_2$. I call these "partial" and "final" EM. In addition, there are two IV traces for each device (one from $T_1$ and one from $T_2$), that may be used in the analysis (not sure yet). 

Devices may belong to different geometry groups. The user should, by hand, sort EM traces into either the proper geometry folder, or the Trash folder if the trace is garbage. Then a full analysis should be conducted on a single geometry at a time. First call prep_all() to select ramps for analysis and to set res_tot_0. This creates and pickles a dict of dicts whose keys are the trace names and whose values are a dict of the prepared DF and the res_tot_0 value. Optionally now you can call `prep_device()` if you want to revise the preparation of any specific trace - it will update the pickled dictionary. Next call `resistance_ratios_from_em` which will compute the parameter $\alpha$ which captures how much the change from $T_1$ to $T_2$ has affected the series resistance. 

Now call fit_all() with to fit the partial and final traces of each device to the constant power model, where the fitting of the final trace is tethered by the fitted series resistance calculated from the partial trace. This function creates and pickles a dataframe whose indices are the trace names and whose values are various demographic info about the device as well as the fit results. It also creates and pickles a dict of dicts with the same information. The pickled dataframe has a column 'conforms' which denotes whether the device appears to conform to the constant power model or not. This is initialized to 1 for all traces but values can later be changed by calling set_pc_conforms(fname). Optionally now you can call `plot_traces_combined()` and `plot_fits_combined` to get PNG images of the traces that were used for fitting, as well as the fits themselves. 

Finally, call `analyze_pc` to create and pickle a dataframe which stores information in the most usable way. Namely, rows are indexed by device, and columns contain information such as $T_1$, $T_2$, fitted critical power at $T_1$, fitted critical power at $T_2$, and device demographics. 

# Brainstorming new varitemp analysis work flow
First sort into trash and geometry folders like before. Now do pic_all() that will plot a zoom in of the split region. Try y_max as 20 ohms plus the resistance on the last point of the first ramp of final EM. Try y_min as the resistance on the last point of the last ramp of partial EM plus 20 ohms. Based on inspecting these plots discard devices that have some obviously strange behavior. 

Now do the "prep": for every device choose the first valid ramp of the final EM and indicate any false ramps that might fall within the fitting range. 


Now do the fits for a given deltaRj. For partial EM discard the last "ramp" (i.e. the partial ramp designed for comparing resistances), then include any ramps whose first point has resistance within deltaRj of the last actual ramp. Take as res_tot_0 the resistance at the first point of the first of these included ramp cycles. For final EM include all ramps whose resistance at the first point of the ramp is within deltaRj of the resistance at the first point of the chosen "first valid ramp". Plot the overlaid traces as we have been doing and create the same dicts and dfs of the results. 


# New Prep for Varitemp Analysis
For each device: choose false ramps for partial, choose last valid ramp for final and then false ramps for final - set "false_ramp" to true for all ramps beyond the last valid ramp. Add in a function that takes rjmin and rjmax and does an unconstrained fit of the non-false-ramps from the final EM trace between rjmin and rjmax (calling rj = 0 at the first non-false-ramp of the final EM trace). In the final aggregate analysis DF there should be a new column(s) which holds to fit info from the unconstrained fit of just the final EM trace for that device. 

To include the current-dependent series resistance add a step in the fitting that takes the first valid ramp included in the fit and uses the first 50 or so points from that ramp to estimate the slope of R(I), and then of course the actual fitting functions need to be modified to account for this. 


# NOTES FROM MOST RECENT REVISION TO CODE
After extracting the ramp cycles for a trace we have a dataframe where each ramp is a row in the frame, and the columns store the following:
- lists of all voltages, currents and resistances in that ramp
- "last_v" and "first_i" which are the voltage and current at the last point in the ramp i.e. where EM is initiated
- "first_r" which is the average of the resistance between the 2nd through 10th point of the ramp which is interpreted as the total resistance of the device on that ramp cycle.
- a boolean indicating whether the ramp is a false ramp cycle and should be excluded from fitting (all initialized to False)

The fit results in two parameters, the fixed series resistance of the device and the critical power in the junction. For the first included ramp in the fit we have "first_r" = (measured total resistance) = r_series + rj0 where rj0 is the initial junction resistance. On each subsequent ramp "first_r" = r_series + rj(t). 

To plot the fit, for each ramp we plot "last_i" (current at which EM is initiated) as a function of "rj(t)" for that ramp. 