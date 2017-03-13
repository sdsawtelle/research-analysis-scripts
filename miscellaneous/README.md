# miscellaneous research scripts

## basic_plotting
Basic plotting functionality for standard current vs. voltage traces of electromigration. Plot in resistance-voltage space or current-voltage space. Also basic text file processing and plotting for constant voltage EM traces.

## keithley_range_investigation
Plot simple current vs. voltage traces obtained from Keithley on different range settings. Indicate where current range and voltage range changes are occurring. Useful for investigating measurement artifacts using sweeps over test resistors of different resistance. Also contains a simple model for a fixed voltage and current offset that can be fit to the data to test for those two effects. 

## nw_resistivity_and_rseries_fit
Plot measured resistances as a function of wire length and diameter in order to extract out both the common series resistance and to fit a value of resistance per unit length for each diameter. We do a linear fit to the average resistance as a function of wire length for each diameter. This was written for a chip type with a wide variety of lengths and diameters. 

## nw_yield_monitoring
This script will take the standard yield measurement files in a directory and make a plot of resistance values.