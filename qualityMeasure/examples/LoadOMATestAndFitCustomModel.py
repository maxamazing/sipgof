#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run the adaptive procedure quality measures for a one actual datasets in the 
Oldenburger Measurement Platform xml-markup, Assuming a unix platform with unix paths.
The track is compared for the two model fits:
    concentrated
    two-state (unconcentrated) as zero hypothesis
This commented tutorial shows how to implement a custom psychometric model
@author: max scharf maximilian.scharf_at_uol.de
"""
import matplotlib.pyplot as plt
import qualityMeasure.multiStateListenerModelCollection as modelCollection
import qualityMeasure.adaptiveProcedureReader as reader
from qualityMeasure.multiStateListener import genStates
from qualityMeasure.multiStateListener import mostLikelyStatesList
from pathlib import Path
import os
import numpy as np
import scipy.optimize as opt

# run the script from anywhere
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


folder = Path("./exampleData/")

readerDict = reader.OMAparse(folder)
measID = list(readerDict.keys())[0]  # only use the first entry
dat = readerDict[measID]


def myOwnModelFit(trackSnr, trackResponse, lapsRate=0.0, plotOnce=False):
    """return the log-likelihood difference of a single-state-model
    and a two-state-model to describe the provided track

    Have a look at the implementations in the modelCollection!

    """

    def model(x, plot=False):
        # oldenburger matrixtest: 5 words per sentence
        testItemsPerTrial = 5

        # generate a list of psychometric states that could describe the track, the vecor x determines how many states are generated
        estimStates = genStates(x,
                                guessRate=1 / 10,  # ->this parameter depends on the test type!
                                lapsRate=lapsRate,  # the lapsrate reduces the chance to respond correctly, even though the stimulus was percieved
                                threshold=0.5,  # ->define the threshold percentage
                                )  # -> 2 remaining deg. of freedom per internal state

        if plot:
            estimStates[0].plot(x[0]-5, x[0]+5)

        # calculate which sequence of states best describes the data
        return mostLikelyStatesList(trackSnr, trackResponse, testItemsPerTrial, estimStates)["ll"]

    # initialize a vector which contains the free parameters of the model. here it is:
        # threshold state 1
        # slope state 1
        # threshold state 2
        # slope state 2
    x2 = [np.quantile(trackSnr, 0.25), 0.1425, np.quantile(
        trackSnr, 0.75), 0.1425]

    # Here, we want to set boundry conditions:
    # the first list is the lower bound in the same order as above
    # the second list is the upper bound ine the same order as above
    x2_bounds = opt.Bounds([-np.inf, 0.1425, -np.inf, 0.001],
                           [np.inf, 0.1425, np.inf, 3.0])

    # free parameters of the model are optimized for maximum log-likelihood
    optres2 = opt.minimize(lambda x: -model(x), x2, bounds=x2_bounds)

    # we do the same as above, but now with a smaller model, that contains only one state. The free parameters of the model are:
    # threshold state 1
    # slope state 1
    x1 = [np.quantile(trackSnr, 0.5), 0.1425]
    x1_bounds = opt.Bounds([-np.inf, 0.1425], [np.inf, 0.1425])

    # we optimize these parameters too
    optres1 = opt.minimize(lambda x: -model(x), x1, bounds=x1_bounds)

    # plot the psychometric function of the model
    model(optres1.x, plotOnce)

    # The difference in log-likelihood is a measure of consistence of the track!
    return model(optres2.x)-model(optres1.x)


print("measurement: \n\t{}\nlog likelihood difference (SiPGOF): \n\t{}\n\n".format(
    measID, myOwnModelFit(dat["snr"], dat["response"])))


# %%
# we can now calculate the log likelihood differnce as function of the lapsrate of our model.

lapsRate = np.linspace(0, 0.49, num=10)  # we cant go below 50% when we have data on the SRT50

ll = list(map(lambda lr: myOwnModelFit(dat["snr"], dat["response"], lr, plotOnce=True), lapsRate))
plt.title("psychometric functions with different laps rates")
plt.show()

# %%
# we observe that the likelihood of the single-state model and the likelihood of the two-state model
# get closer. In the limit of a lapsrate=1-guessrate, the two psychometric functions are exactly equal
# you can change the value at line 49 to 1/3 (this would be the guessrate of a sentence test with 3 alternatives, the OLKISA as example has 1/7)

plt.plot(lapsRate*100, ll)
plt.xlabel("lapsrate/%")
plt.ylabel("log.likelihood difference")
plt.title("The two- and single-state model get more\nand more similar when we increase the lapsrate")
