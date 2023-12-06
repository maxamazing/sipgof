#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:41:34 2023

contains multistateListener models

@author: max scharf maximilian.scharf_at_uol.de
"""
import numpy as np
import matplotlib.pyplot as plt
from qualityMeasure.multiStateListener import Person
from qualityMeasure.multiStateListener import modelA
from qualityMeasure.multiStateListener import mostLikelyModelA
from qualityMeasure.multiStateListener import genStates
from qualityMeasure.multiStateListener import mostLikelyStatesList
import scipy.optimize as opt  # fitting most likely models


def modelB(trackSnr, trackResponse, params):
    """
    generate a listening model with fixed slope for the lower/concentrated state.
    The amount of internal states is deduced from
    the number of parameters in the container params.

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
                "how often the state was present according to this prediction"

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
                            guessRate=1 / 10,  # ->this parameter depends on the test type!
                            lapsRate=0.00,  #
                            threshold=0.5,  # ->define the threshold percentage
                            )  # -> 2 remaining deg. of freedom per internal state

    return {"st": estimStates, **mostLikelyStatesList(trackSnr, trackResponse, testItemsPerTrial, estimStates)}


def mostLikelyModelB(trackSnr, trackResponse, plot=True):
    """return the likelihood of both a single-state-model
    and a two-state-model to describe the provided track.


    The slope is in the typical range for matrix sentence tests for the concentrated state only
    (see Kollmeier 2015: http://dx.doi.org/10.3109/14992027.2015.1020971)

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
        return model(trackSnr, trackResponse, params=x)["ll"]
    x2 = [np.quantile(trackSnr, 0.25), 0.1425, np.quantile(
        trackSnr, 0.75), 0.1425]  # turningpoint and slope
    x2_bounds = opt.Bounds([-np.inf, 0.1425, -np.inf, 0.001],
                           [np.inf, 0.1425, np.inf, 3.0])
    optres2 = opt.minimize(lambda x: -model2(x), x2, bounds=x2_bounds)

    # singleState
    def model1(x):
        return model(trackSnr, trackResponse, params=x)["ll"]
    x1 = [np.quantile(trackSnr, 0.5), 0.1425]  # turningpoint and slope
    x1_bounds = opt.Bounds([-np.inf, 0.1425], [np.inf, 0.1425])
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


def modelC(trackSnr, trackResponse, params):
    """
    generate a listening model with slopes as open paramters.
    The Naming converntion is (see multiStateListenerModelCollection.py):
        A:fixed slopes
        B:unconcentrated slope fixed
        C:both slopes are open paramters

    The amount of internal states is deduced from
    the number of parameters in the container params.

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
                "how often the state was present according to this prediction"

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
                            guessRate=1 / 10,  # ->this parameter depends on the test type!
                            lapsRate=0.00,  #
                            threshold=0.5,  # ->define the threshold percentage
                            )  # -> 2 remaining deg. of freedom per internal state

    return {"st": estimStates, **mostLikelyStatesList(trackSnr, trackResponse, testItemsPerTrial, estimStates)}


def mostLikelyModelC(trackSnr, trackResponse, plot=True, slopeInit=0.1425, slopeLowerBound=0.05, slopeUpperBound=0.5, maxIter=[100, 100]):
    """return the likelihood of both a single-state-model
    and a two-state-model to describe the provided track.

    The slope is initialized witht the typical range for matrix sentence tests 
    (see Kollmeier 2015: http://dx.doi.org/10.3109/14992027.2015.1020971).
    It may take on values in a specified range (see below)

    Parameters
    ----------
    trackSnr : array
        stimuls.
    trackResponse : array
        response.
    plot : bool, optional
        display the data. The default is True.
    slopeInit : float, optional
        initialize optimizer with this slope for all psychometric functions 
    slopeLowerBound : float, optional
        lower bound for the slope. The optimizer can not go below
    slopeUpperBound : float, optional
        upper bound for the slope, the Optimizer can not go below
    maxIter : unsigned list of two , optional
        use to see the evolution of the optimization process first value: single state, second value twostate


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
    # make interchanging the model easier
    model = modelC
    slopeInit = 0.1425
    slopeLowerBound = 0.05  # probability/dB
    slopeUpperBound = 0.5

    # twoState
    def model2(x):
        return model(trackSnr, trackResponse, params=x)["ll"]
    x2 = [np.quantile(trackSnr, 0.25), slopeInit, np.quantile(
        trackSnr, 0.75), slopeInit]  # turningpoint and slope
    x2_bounds = opt.Bounds([-np.inf, slopeLowerBound]*2, [np.inf, slopeUpperBound]*2)
    optres2 = opt.minimize(lambda x: -model2(x), x2, bounds=x2_bounds,
                           options={"maxiter": maxIter[1]})

    # singleState
    def model1(x):
        return model(trackSnr, trackResponse, params=x)["ll"]
    x1 = [np.quantile(trackSnr, 0.5), slopeInit]  # turningpoint and slope
    x1_bounds = opt.Bounds([-np.inf, slopeLowerBound], [np.inf, slopeUpperBound])
    optres1 = opt.minimize(lambda x: -model1(x), x1, bounds=x1_bounds,
                           options={"maxiter": maxIter[0]})

    fig = []
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


def modelD(trackSnr, trackResponse, params):
    """
    generate a listening model with fixed slope for all states.
    The amount of internal states is deduced from
    the number of parameters in the container params.

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
                "how often the state was present according to this prediction"

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
                            guessRate=1 / 10,  # ->this parameter depends on the test type!
                            lapsRate=0.00,  #
                            threshold=0.5,  # ->define the threshold percentage
                            slope=0.171,     # average slope for german male matrix test
                            )  # -> 1 remaining deg. of freedom per internal state

    return {"st": estimStates, **mostLikelyStatesList(trackSnr, trackResponse, testItemsPerTrial, estimStates)}


def mostLikelyModelD(trackSnr, trackResponse, plot=True):
    """return the likelihood of both a single-state-model
    and a two-state-model to describe the provided track.

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

    model = modelD

    # twoState
    def model2(x):
        return model(trackSnr, trackResponse, params=x)["ll"]
    x2 = [np.quantile(trackSnr, 0.25), np.quantile(
        trackSnr, 0.75), ]  # turningpoint
    optres2 = opt.minimize(lambda x: -model2(x), x2)

    # singleState
    def model1(x):
        return model(trackSnr, trackResponse, params=x)["ll"]
    x1 = [np.quantile(trackSnr, 0.5)]  # turningpoint
    optres1 = opt.minimize(lambda x: -model1(x), x1)

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


def model_varSlope(trackSnr, trackResponse, params, slope):
    """
    generate a listening model with fixed slope for all states.
    The amount of internal states is deduced from
    the number of parameters in the container params.

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
                "how often the state was present according to this prediction"

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
                            guessRate=1 / 10,  # ->this parameter depends on the test type!
                            lapsRate=0.00,  #
                            threshold=0.5,  # ->define the threshold percentage
                            slope=slope,     # average slope for german male matrix test
                            )  # -> 1 remaining deg. of freedom per internal state

    return {"st": estimStates, **mostLikelyStatesList(trackSnr, trackResponse, testItemsPerTrial, estimStates)}


def mostLikelyModel_varSlope(trackSnr, trackResponse, plot=True, slope=0.16):
    """return the likelihood of both a single-state-model
    and a two-state-model to describe the provided track.

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

    model = model_varSlope

    # twoState
    def model2(x):
        return model(trackSnr, trackResponse, params=x, slope=slope)["ll"]
    x2 = [np.quantile(trackSnr, 0.25), np.quantile(
        trackSnr, 0.75), ]  # turningpoint
    optres2 = opt.minimize(lambda x: -model2(x), x2)

    # singleState
    def model1(x):
        return model(trackSnr, trackResponse, params=x, slope=slope)["ll"]
    x1 = [np.quantile(trackSnr, 0.5)]  # turningpoint
    optres1 = opt.minimize(lambda x: -model1(x), x1)

    fig = None
    if plot:
        plt.close('all')  # important if inlinefigures are disabled
        fig, axs = plt.subplots(
            2, 2, gridspec_kw={"height_ratios": [1, 1.5]}, figsize=[20, 10])
        dummy = Person.fromData(trackSnr, trackResponse, "singleState")
        dummy.states = model(trackSnr, trackResponse, optres1.x, slope)["st"]
        dummy.trackState = model(trackSnr, trackResponse, optres1.x, slope)["sl"]
        dummy.plot(axs=axs[:, 0])

        dummy = Person.fromData(trackSnr, trackResponse, "twoState")
        dummy.states = model(trackSnr, trackResponse, optres2.x, slope)["st"]
        dummy.trackState = model(trackSnr, trackResponse, optres2.x, slope)["sl"]
        dummy.plot(axs=axs[:, 1])

    deltaThres = model(trackSnr, trackResponse, optres2.x, slope)["st"][0].thresStim -\
        model(trackSnr, trackResponse, optres2.x, slope)["st"][0].thresStim
    return {
        1: model1(optres1.x),
        2: model2(optres2.x),
        "deltaThresh": deltaThres,
        "1stModel": model(trackSnr, trackResponse, optres1.x, slope),
        "2stModel": model(trackSnr, trackResponse, optres2.x, slope),
        "plot": fig,
        "solver1st": optres1,
        "solver2st": optres2

    }


def mostLikelyModelSinglePsy(trackSnr, trackResponse, plot=False):
    """return the likelihood and estimated threshold of the single psychometric function fit.


    The slope is in the typical range for matrix sentence tests for the concentrated state only
    (see Kollmeier 2015: http://dx.doi.org/10.3109/14992027.2015.1020971)

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
        Snr :
            estimated snr the model
        slope:
            estimated slope of the model


    """

    model = modelB

    # singleState

    def model1(x):
        return model(trackSnr, trackResponse, params=x)["ll"]
    x1 = [np.quantile(trackSnr, 0.5), 0.1425]  # turningpoint and slope
    x1_bounds = opt.Bounds([-np.inf, 0.1425], [np.inf, 0.1425])
    optres1 = opt.minimize(lambda x: -model1(x), x1, bounds=x1_bounds)

    fig = None
    if plot:
        plt.close('all')  # important if inlinefigures are disabled
        fig, axs = plt.subplots(
            1, 1, figsize=[20, 10])
        dummy = Person.fromData(trackSnr, trackResponse, "singleState")
        dummy.states = model(trackSnr, trackResponse, optres1.x)["st"]
        dummy.trackState = model(trackSnr, trackResponse, optres1.x)["sl"]
        dummy.plot(axs=axs[:, 0])

    if optres1.success == True:
        return {
            1: model1(optres1.x),
            "opt": optres1,
            "snr": optres1.x[0],
            "slope": optres1.x[1],
            "plot": fig,

        }
    else:
        return None
