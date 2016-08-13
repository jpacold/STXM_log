STXM e-logbook generator
Joe Pacold (Lawrence Berkeley Lab)
----------------------------
Generates an easily readable pdf file summarizing one day
of STXM data collection.

Requires numpy and matplotlib (using the Anaconda Python3
distribution is recommended).

Edit STXM_log_config.txt before running to set the default
directories for data and output and to set the name of the
beamline (which will appear at the top of the output file). 

Usage:
python STXM_log.py [directory]

If a directory corresponding to one day of data is specified,
the script will look for data in that folder.

If no directory is specified:
1) If the current time is between 12am and 2am, the script
will check for data and a logbook file corresponding to the
day *before* the current date. If there is no logbook file,
or if it was last modified before the current date, it will
be generated/updated.
2) The script will check for data corresponding to the current
date, and generate a logbook file.
