#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run the adaptive procedure quality measures for a one actual datasets in the 
Oldenburger Measurement Platform xml-markup, Assuming a unix platform with unix paths.
The track is compared for the two model fits:
    concentrated
    two-state (unconcentrated) as zero hypothesis

@author: max scharf maximilian.scharf_at_uol.de
"""
import matplotlib.pyplot as plt
import qualityMeasure.multiStateListenerModelCollection as modelCollection
import qualityMeasure.adaptiveProcedureReader as reader
from pathlib import Path
import os

#run the script from anywhere
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


folder = Path("./exampleData/")

readerDict = reader.OMAparse(folder)
measID = list(readerDict.keys())[0]  # only use the first entry

dat = readerDict[measID]
res = modelCollection.mostLikelyModelA(dat.snr, dat.response, plot=True)
plt.show()

print("measurement: \n\t{}\nlog likelihood difference (PiPGOF): \n\t{}\n\n".format(measID, res[2]-res[1], ))
