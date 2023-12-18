#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run the adaptive procedure quality measures for two measurements in OMA-xml markup
and fit a single psychometric function

@author: max scharf maximilian.scharf_at_uol.de
"""
import matplotlib.pyplot as plt
import qualityMeasure.multiStateListenerModelCollection as modelCollection
import qualityMeasure.adaptiveProcedureReader as reader
from pathlib import Path
import os

# run the script from anywhere
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


folder = Path("./exampleData/")

readerDict = reader.OMAparse(folder)

# iterate through all files
for measID, dat in readerDict.items():
    res = modelCollection.mostLikelyModelSinglePsy(dat["snr"], dat["response"], plot=True)
    plt.show()

    print("measurement: \n\t{}\nSRT50: \n\t{}\nslope: \n\t{}\n\n".format(
        measID, res["snr"], res["slope"]))
