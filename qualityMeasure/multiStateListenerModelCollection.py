#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:41:34 2023

contains some multistateListener models

@author: max scharf Do 8. Mai 13:08:42 CEST 2025 maximilian.scharf_at_uol.de
"""
import numpy as np
import matplotlib.pyplot as plt
from qualityMeasure.multiStateListener import Person, mostLikelyModelA, genStates, baumWelch
import scipy.optimize as opt  # fitting most likely models


def modelB(trackSnr, trackResponse, params, optimizerMode=False):
    """
    generate a listening model with fixed slope for all states.
    The guess-rate is in the range 0.1-0, as would follow for closed-set and open matrix sentence tests

    Prospective users of this library probably have to write their own model
    similar to this one.

    Parameters
    ----------
    trackSnr : list of float
        presented stimulus.
    trackResponse : list of float
        measured response.
    params : list of float
        paramters of states that will be deserialized to the settings
    optimizerMode: bool
        only return likelihood if true

    Returns
    -------
    dict
        st:        
            container of the most likely states with corresponding 
            psychometric function that describes the listener.
        ll:
            likelihood of this sequence
        sl:
            list of most likely state-sequence
        pc:
            dict of Percentual content sorted by statename: 
                "how often the state was present according to this model"

    """
    # errorcheck for common mistakes
    if not (type(params) is list or type(params) is np.ndarray):
        raise Exception("Model init: free parameters are not a list")

    testItemsPerTrial = 5  # ->this parameter depends on the test type!
    """e.g,
        -oldenburger matrixtest: 5 words per sentence
        -audiogram: one stimulus.
        """

    estimStates = genStates(params,
                            lapsRate=0.00,  #
                            threshold=0.5,  # ->define the threshold percentage
                            slope=0.1425,   # average slope for english/cantonese matrix test
                            # -> 2 remaining deg. of freedom per internal state (srt and guessrate)
                            )

    return {"st": estimStates, **baumWelch(trackSnr,
                                           trackResponse,
                                           testItemsPerTrial,
                                           estimStates,
                                           not optimizerMode)}


def mostLikelyModelB(trackSnr, trackResponse, plot=True):
    """return the likelihood of both a single-state-model
    and a two-state-model to describe the provided track.

    The guessrate can be in the interval [0,0.1], which is the theoretical range for open or closed sentece tests

    Parameters
    ----------
    trackSnr : array
        stimuls.
    trackResponse : array
        response.
    plot : bool, optional
        display the data. The default is True.

    Returns
    -------
    dict
        1:
            logLikelihood of 1-state model
        2: 
            logLikelihood of 2-state model
        deltaSnr :
            estimated snr difference of the two state model ("deltaSnr")
        1stModel:
            1 state model
        2stModel
            2 state model


    """

    model = modelB

    # twoState
    def model2(x):
        return model(trackSnr, trackResponse, params=x, optimizerMode=True)["ll"]
    x2 = [np.quantile(trackSnr, 0.25), 0.1, np.quantile(
        trackSnr, 0.75), 0.1]  # turningpoint and guuess-rate
    x2_bounds = opt.Bounds([-np.inf, 0, -np.inf, 0],
                           [np.inf, 0.1, np.inf, 0.1])  # [lower bound],[upper bound]
    optres2 = opt.minimize(lambda x: -model2(x), x2, bounds=x2_bounds)

    # singleState
    def model1(x):
        return model(trackSnr, trackResponse, params=x, optimizerMode=True)["ll"]
    x1 = [np.quantile(trackSnr, 0.5), 0.1]  # turningpoint and slope
    x1_bounds = opt.Bounds([-np.inf, 0], [np.inf, 0.1])
    optres1 = opt.minimize(lambda x: -model1(x), x1, bounds=x1_bounds)

    fig = None
    if plot:
        plt.close('all')  # important if inlinefigures are disabled
        fig, axs = plt.subplots(
            2, 2, gridspec_kw={"height_ratios": [1, 1.5]}, figsize=[20, 10])
        dummy = Person.fromData(trackSnr, trackResponse, "singleState")
        dummy.states = model(trackSnr, trackResponse, optres1.x)["st"]
        dummy.trackState = model(trackSnr, trackResponse, optres1.x)["sl"]
        dummy.plot(axs=axs[:, 0])

        dummy = Person.fromData(trackSnr, trackResponse, "twoState")
        dummy.states = model(trackSnr, trackResponse, optres2.x)["st"]
        dummy.trackState = model(trackSnr, trackResponse, optres2.x)["sl"]
        dummy.plot(axs=axs[:, 1])

    deltaThres = model(trackSnr, trackResponse, optres2.x)["st"][0].thresStim -\
        model(trackSnr, trackResponse, optres2.x)["st"][0].thresStim
    return {
        1: model1(optres1.x),
        2: model2(optres2.x),
        "deltaThresh": deltaThres,
        "1stModel": model(trackSnr, trackResponse, optres1.x),
        "2stModel": model(trackSnr, trackResponse, optres2.x),
        "plot": fig,
        "solver1st": optres1,
        "solver2st": optres2

    }
