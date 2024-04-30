#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python 3.12
Multistate-listener model

@author: max scharf maximilian.scharf_at_uol.de
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy.optimize as opt  # fitting most likely models
import warnings


class undefState:
    """container for an undefined state"""

    def __init__(self, name):
        self.name = name


class State:
    def __init__(self, guessRate, lapsRate, threshold,
                 slope, thresStim, name="undef. state"):
        """
        single state of a person, which is modelled by a psychometric function

        Parameters
        ----------
        guessRate  : double
            context free forced choice: propto 1/number of choices
        lapsRate: double
            pobabbility of making errors at optimal presentation
        threshold : double
            percentage correct (0,1) at which the Threshold is probed
        slope : double
            slope of psychometric function at threshold stmulation
        thresStim : double
            value of the stimulus at the threshold
        name : string
            name of the state

        Returns
        -------
        state-object of a person

        see: four-parameter transformed Logistic model
        (Lam et al., 1996; King-Smith and Rose, 1997; Shen and Richards, 2012)
        """

        # errorcheck
        tmp = np.array([guessRate, lapsRate])
        if tmp.any() < 0 or tmp.any() > 1:
            raise Exception("guessRate,lapsRate: invalid Settings")
        tmp = np.array([guessRate, lapsRate, threshold])
        if tmp.any() < 0 or tmp.any() > 1:
            raise Exception("threshold: invalid Settings")
        if slope <= 0:
            raise Exception("slope: invalid Settings [{}]".format(slope))
        if (threshold-guessRate) <= 0:
            raise Exception(
                """threshold [{}] undefined: <=guessRate [{}]
                : invalid Settings""".format(threshold, guessRate))
        if (1-guessRate-lapsRate) <= 0:
            raise Exception(
                """guessRate [{}]>=1-lapsRate [{}]
                : invalid Settings""".format(guessRate, lapsRate))
        if (1-lapsRate-threshold) <= 0:
            raise Exception(
                """threshold [{}] undefined: >=1-lapsRate [{}]
                : invalid Settings""".format(threshold, lapsRate))

        self.gues = guessRate
        self.laps = lapsRate
        phi = threshold
        # see http://dx.doi.org/10.1121/1.4979580
        self.beta = slope*(1-self.gues-self.laps) / \
            ((phi-self.gues)*(1-self.laps-phi))
        self.alpha = thresStim + \
            np.log((1-self.laps-phi)/(phi-self.gues))/self.beta

        # keep information for later reference
        self.name = name
        self.threshold = threshold
        self.slope = slope
        self.thresStim = thresStim

    def psychFunc(self, snr):
        '''psychometric function: item-specific intelligibility function'''
        return self.gues+(1-self.gues-self.laps)/(1+np.exp(-self.beta*(snr-self.alpha)))

    def get(self, snr):
        """
        do a trial at given SNR and return hit or miss

        Parameters
        ----------
        snr : double
            signal to noise ratio of test.

        INFORMATION: the number of alternative choices has an influence on the psychometric function
            (depending on the setup of the Test e.g. open/closed focesd choice) and has to be considered
            when initializing a person.

        Returns
        -------
        boolean: hit=true; miss=false

        """
        prob = self.psychFunc(snr)
        rand = np.random.uniform()
        return (rand < prob)

    def likely(self, snr, hit):
        '''return the likelihood of given state to occur at given snr
        ARGUMENTS:
            snr     signal to noise ratio
            hit     boolian: hit=True /miss=False'''
        prob = self.psychFunc(snr)
        if hit:
            return prob
        return 1-prob

    def plot(self, start, stop):
        '''plot the psychometric function
        ARGUMENTS:
            start:  start SNR range to plot
            stop:  stop SNR range to plot'''
        snr = np.linspace(start, stop, 100)
        plt.plot(snr, self.psychFunc(snr), label="{}".format(self.name))
        plt.xlabel("SNR /dB")
        plt.ylabel("prob. correct")
        plt.vlines(self.thresStim, 0, 1, color="red")
        plt.title("internal state")
        plt.ylim(self.gues, 1)


"""#######################################################################
            functions for maximum likelihood estimation
   #######################################################################"""


def transMatFromStatesList(states):
    """
    generate the transitionmatrix from a list of states

    Parameters
    ----------
    states: list
        list of states, containing IDs

    Returns
    -------
    transitionmatrix

    """

    #                |transprob to A |transprob to B
    # -------------------------------------------------
    # coming from A  |  [[  x        |     x  ],
    # -------------------------------------------------
    # coming from B  |   [  x        |     x  ]]

    transMat = np.zeros([max(states)+1, max(states)+1])
    for curState, nextState in zip(states, states[1:]):
        transMat[curState, nextState] += 1

    # normalize
    ret = []
    for n, row in enumerate(transMat):
        with np.errstate(invalid='raise'):
            try:
                ret.append((row/sum(row)).tolist())
            except:
                tmp = np.zeros(max(states)+1).tolist()
                # make the transitionmatrix a valid matrix with the assumption that state n stays in n
                tmp[n] = 1
                ret.append(tmp)
    return ret


def logLikelihoodFromTransmat(transMat, states):
    """
    calculate the likelihood that the list of state-IDs was the result
    of the transition matrix

    Parameters
    ----------
    transMat:matrix (optional)
        transitionmatrix of the states, if undefined:
        equal transition probabilities are assumed.
    states: list of unsigned
        contains index of previous states in order of occurance
        states[0] is the state that corresponds to the first trial
        index must be a valid index of statesList

    Returns
    -------
    logLikelihood

    """
    if transMat is not None:
        checkTransMat(transMat, max(states)+1)

    #                |transprob to A |transprob to B
    # -------------------------------------------------
    # coming from A  |  [[  x        |     x  ],
    # -------------------------------------------------
    # coming from B  |   [  x        |     x  ]]

    tmpll = 0
    for curState, nextState in zip(states, states[1:]):
        with np.errstate(divide='raise'):
            try:
                tmpll += np.log(transMat[curState][nextState])
            except FloatingPointError:
                raise Exception("The transition matrix does not permit the state-"
                                "transition ID{} to ID{}".format(curState, nextState))
    return tmpll


def mostLikelyState(snr, numHit, numMiss, statesList):
    """
    return the most likely internal state given a response.
    The influence of the transitionmatrix can not be included
    at this point, because it is not defined.
    ----------
    snr : double
        signal to noise ratio.
    numHit : unsigned interger
        response # correct anwser during a single trial.
    numMiss : unsigned interger
        response # incorrect answer during a single trial.
    statesList: list
        list of states.

    Returns
    -------
    index of most likely state, logLikelihood

    """

    if statesList == []:
        raise Exception("no valid states defined")
    np.seterr(divide="ignore")

    ll = []  # container for loglikelihood of each possible state
    for i, s in enumerate(statesList):
        tmpll = 0
        for n in range(numHit):
            tmpll += np.log(s.likely(snr, True))  # may smoetimes produce nan
        for n in range(numMiss):
            tmpll += np.log(s.likely(snr, False))  # may smoetimes produce nan

        # include the binomial coefficient
        tmpll += np.log(np.math.factorial(numHit+numMiss) /
                        (np.math.factorial(numHit)*np.math.factorial(numMiss)))

        ll.append(tmpll)

    nMax = np.nanargmax(ll)
    np.seterr(divide="warn")
    return nMax, ll[nMax]


def mostLikelyStatesList(trackSnr, trackResp, testItemsPerTrial,
                         statesList, useTransMat=False):
    """
    Calculate the most likely list of states for the given data

    -------
    tackSnr : double-list
        list of snr of an adaptive procedure
    trackResp : double list on interval [0,1]
        list of responses of an adaptive procedure
    testItemsPerTrial : int
        number of testitems for one adaptive step="trial"
    statesList: list
        list of states
    useTransMat:boolean, optional
        use the transitionmatrix of the states to caluculate the likelihood 
        that the estimated stateslist is the results of the transition matrix

    Returns
    -------
    dict
    ll:
        likelihood of this sequence
    sl:
        list of most likely state-sequence
    pc:
        dict of Percentual content sorted by statename:
            "how often the state was present according to this prediction"

    """
    if max(trackResp) > 1 or max(trackResp) < 0 or len(trackSnr) == 0:
        raise Exception("ivalid response Track.")
    if len(trackResp) != len(trackSnr):
        raise Exception("tracks do not have the same length.")

    def conv(resp):  # covert response to number Hit/Miss
        numHit = resp*testItemsPerTrial
        if not numHit.is_integer():
            raise Exception(
                "response does not match the number of testitems per trial.")
        return int(numHit), int(testItemsPerTrial-numHit)

    # calculate the log-likelihood from each trial
    sl = []
    ll = 0
    for snr, resp in zip(trackSnr, trackResp):
        s, tmpll = mostLikelyState(
            snr, *conv(resp), statesList)
        sl.append(s)
        ll += tmpll

    # add log-likelihood from transitionMatrix
    if useTransMat:
        transMat = transMatFromStatesList(sl)
        ll += logLikelihoodFromTransmat(transMat, sl)

    pc = {}
    for e, s in enumerate(statesList):
        pc[s.name] = sl.count(e)
    return {"ll": ll, "sl": sl, "pc": pc}


def genStates(params, thresholdStimulus=None, slope=None, guessRate=None, lapsRate=None, threshold=None):
    """
    deserialize the container params, while keeping the named arguments as constants
    return a list of states-objects

    Parameters (order of deserialization)
    ----------
    thresholdStimulus : list of float
        the simulus at the threshold of the psychometric funtion. e.g., srt, pure-tone-threshold
        also referred to as the turning point of the psychometric function
    slope : list of float
        the slope of the psychometric function.
    guessRate : list of float
        chance of guessing corretly during one trial
    lapsRate : list of float
        chance of a random incorrect answer during high level presentation
    threshold : float (0,1)
        threshold definition in percentage correct



    Returns
    -------
    states : list of states objects
        represents the states of a listener.

    """

    states = []
    params = list(params)

    def paramsToSettings(parms, n):
        names = "state_{}".format
        try:
            if thresholdStimulus is None:
                tp = parms.pop(0)
        except IndexError:
            # the first deserialization item may fail and cause serailization to stop
            # once a new state is deserialized, a failure is considered fatal
            return None

        if slope is None:
            slp = parms.pop(0)
        else:
            slp = slope

        if guessRate is None:
            gr = parms.pop(0)
        else:
            gr = guessRate

        if lapsRate is None:
            lr = parms.pop(0)
        else:
            lr = lapsRate

        if threshold is None:
            th = parms.pop(0)
        else:
            th = threshold

        return dict(
            slope=slp,
            thresStim=tp,
            guessRate=gr,
            lapsRate=lr,
            threshold=th,
            name=names(n))

    n = 0
    while len(params) != 0:
        settings = paramsToSettings(params, n)
        if settings is not None:
            states.append(State(**settings))
        n += 1

    return states


def modelA(trackSnr, trackResponse, params):
    """
    generate a listening model with fixed slope for all states.
    The amount of internal states is deduced from
    the number of parameters in the container params.

    Prospective users of this library probably have to write their own model
    similar to this one. See also multiStateListenerModelCollection.py

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
                            slope=0.1425,     # average slope for english/cantonese matrix test
                            )  # -> 1 remaining deg. of freedom per internal state (SRT50)

    return {"st": estimStates, **mostLikelyStatesList(trackSnr, trackResponse, testItemsPerTrial, estimStates)}


def mostLikelyModelA(trackSnr, trackResponse, plot=True):
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

    model = modelA

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


def checkTransMat(transMat, dim):
    """
    raises warning or exception if mistakes are found

    Parameters
    ----------
    transMat : matrix
    dim : needed dimensions

    Returns
    -------
    None.

    """

    # errorcheck transitionmatrix
    for n, s in enumerate(np.sum(transMat, 1)):
        if s != 1:
            raise Exception(
                "transitionmatrix not a probability matrix:"
                "check column {}: [{}]".format(n, transMat))
    # if dim != np.size(transMat, 0) or dim != np.size(transMat, 1):
    #     warnings.warn("transitionMatrix<->statesList dimension mismatch")


class Person:
    '''person, or subject who is modeled by
    several states and an old measurement'''

    def __init__(self, statesList, startState, transMat, fromData=False,
                 trackResponse=[], trackSnr=[], lastTestName="undefined"):

        self.transMat = transMat
        self.startState = startState
        self.states = statesList
        self.fromData = fromData
        self.reset()

        # if init from Data:
        self.trackResponse = trackResponse
        self.trackSnr = trackSnr
        self.trackState = [startState]+[0]*len(self.trackResponse)
        self.lapsastTestName = lastTestName

        checkTransMat(self.transMat, len(self.states))

    @ classmethod
    def fromData(cls, trackSnr, trackResponse, label=""):
        return cls([undefState("undef. state")], 0, [[1]],
                   True, trackResponse, trackSnr, label)

    def add_fit(self, fitFunction):
        """add a fitting function for use in the plots"""
        self.fittingProcedures.append(fitFunction)

    def reset(self):
        '''reset the person to initialized state'''
        # default settings
        self.cmap = plt.get_cmap("tab10")
        self.resFormat = "{0:1.2f}"  # response Formatting
        self.numberOfAxes = 2  # used by interactive Plot
        self.showStatsInPlot = True  # can turn of the latex rendering
        self.fittingProcedures = []

        # internal variables
        self.curState = self.startState
        self.trackState = [self.startState]
        self.trackResponse = []  # store response of person
        self.trackSnr = []  # store snr that was ures to probe
        self.latestTestName = ""  # name of latest test for later reference
        self.fittingProcedures = []  # list of fitting procedures

    def probe(self, snr, nTestItem=1):
        """
        do a trial at given SNR, 
        return the percent correct e[0,1]
        and transition to the next internal state

        Parameters
        ----------
        snr : double
            signal to noise ratio of test.
        nTestItem : integer
            how many trials per test. default=1

        Returns
        -------
        double
            ratio of correct answers.

        """
        # errorcheck
        if self.fromData:
            raise Exception(
                "Person recovered from Data: unknown internal states: simulation not possible")

        self.itemsPerTest = nTestItem
        response = 0
        for i in range(nTestItem):
            response += 1.*self.states[self.curState].get(snr)
        response /= nTestItem
        self.trackResponse.append(response)
        self.trackSnr.append(snr)

        # tarnsition to next internal state
        draw = np.random.uniform()
        transVec = self.transMat[self.curState]
        self.curState = 0
        for t in transVec:
            draw -= t
            if draw <= 0:
                break
            self.curState += 1
        self.trackState.append(self.curState)
        return response

    def printSettings(self, toConsole=True, tex=False):
        """
        Print a summary of the settings

        Parameters
        ----------
        toConsole : bool, optional
            print information to console too. The default is True.
        tex:    bool, optional
            return data formatted as simple LaTex-table. The default is True.

        Returns
        -------
            all settings of the person as formatted table

        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "you need to install pandas to print a table of the parameters of the psychometric function")

        # errorcheck
        if self.states == []:  # there are no known internal states
            return "data: {}".format(self.lastTestName)
        else:
            if self.fromData:
                # calculate the transiotnmatrix
                self.transMat = transMatFromStatesList(self.trackState)
        checkTransMat(self.transMat, len(self.states))

        labelIn = [s.name for s in self.states[0:np.shape(self.transMat)[0]]]  # make it more robust
        labelOut = ["trans prob. to "+s.name for s in self.states[0:np.shape(self.transMat)[1]]]
        mat = pd.DataFrame(np.transpose(self.transMat), labelOut, labelIn)

        def getSettings(state):
            return pd.DataFrame([state.thresStim,
                                 state.threshold,
                                 state.slope*100,
                                 state.gues,
                                 state.laps,
                                 state.alpha,
                                 state.beta],
                                ["stimulus at threshold/ dB",
                                 "threshold/ perc.",
                                 "slope/ perc./dB",
                                 "guessRate/ perc.",
                                 "lapsRate/ perc.",
                                 "alpha",
                                 "beta"], [state.name])

        dat = pd.concat([getSettings(s) for s in self.states], axis=1)

        if toConsole:
            if self.transMat != [[1]]:
                print("-"*80+"\ntransition matrix:")
                print(mat)
            print("-"*80+"\nsettings of internal states:\n")
            print(dat)
            print("-"*80)

        dat = pd.concat([dat, mat])

        dat.style.format("{{:2.2f}}".format)
        if tex:
            text = dat.style.to_latex()
            # romove some stuff so that it compiles with no hassle
            return text.replace('\\toprule', ' ').replace('\\midrule', ' ').replace('\\bottomrule', ' ').replace("\n", " ")
        else:
            return dat.style.to_string()

    def plot(self, axs=None, figsize=[17, 14], loc="upper right", stimulusUnits="SNR/dB", showStdBinom=True, showStats=False, snrRange=None, showLegend=True, heightRatios=[1, 1.5]):
        """
        plot the full history of the person: Phaseplane and adaptive track.

        Parameters
        ----------
        axs : list of two figure-axes, default=None
            If provided, the plot is drawn onto
            the two axes and nothing is returned
            else: a new figure is created and returned.
        figsize : x-y size of the figure
            DESCRIPTION. The default is [17,14] inches.
        loc : string
            location of the informational matrix in the bottom plot.
        stimulusUnits: optional, default:SNR/dB
            unist of the stimulus signal
        showStdBinom: bool,optional, default=True
            show the standard deviation of a binomial distribution in dotted lines
        showStats: bool, optional, default=True
            show the table with the properties of the psychometric functions
        snrRange: float, limit the snr range to have constant scale for the axis 
            (uesful for comparison betwen plots), None= disabled
        showLegend: bool, show the legend of the scatter-plot
        heightRatios:list, use it to change the heightratio of the plots


        Returns
        -------

        handle to the plots which contain the drawn lines
        figure if axs is not provided

        """
        # does the person contain data to plot?
        if len(self.trackSnr) == 0:
            warnings.warn(
                "person contains no data for plotting."
                "Did you forget to add data?")

        # prepare the plot
        if axs is None:
            fig, axs = plt.subplots(
                2, gridspec_kw={"height_ratios": heightRatios}, figsize=figsize)
        else:
            fig = None
            # errorcheck
            if len(axs) != self.numberOfAxes:
                raise Exception("incorrect axes specified")
        plots = []

        # some settings
        lapsegendLoc = ["lower right", "upper center"]  # plot 0 and 1
        snrMargin = 1
        alpha = 0.2

        # phaseplane
        lines = []
        for i, s in enumerate(self.states):
            # colorcode each state
            snr = [snr for snr, state in zip(
                self.trackSnr, self.trackState) if state == i]
            response = [resp for resp, state in zip(
                self.trackResponse, self.trackState) if state == i]
            lines.append(axs[0].scatter(
                snr, response, label="{}".format(s.name), marker="X", color=self.cmap(i)))
        for n, (snr, res) in enumerate(zip(self.trackSnr, self.trackResponse)):
            # add the number of each state
            lines.append(axs[0].annotate("{}".format(n), (snr, res)))
        # add psychometric functions
        for i, s in enumerate(self.states):
            if s.name != "undef. state":
                if snrRange is not None:
                    x, y, std, label = self.plotPsy(
                        min(self.trackSnr)-snrMargin,
                        min(self.trackSnr)+snrRange+snrMargin,
                        state=s,)
                else:
                    x, y, std, label = self.plotPsy(
                        min(self.trackSnr)-snrMargin,
                        max(self.trackSnr)+snrMargin,
                        state=s,)
                lines.append(axs[0].plot(x, y, label=label, color=self.cmap(i))[0])
                if showStdBinom:
                    lines.append(axs[0].plot(
                        x, y+std, "--", color=lines[-1].get_color(), alpha=alpha)[0])
                    lines.append(axs[0].plot(
                        x, y-std, "--", color=lines[-1].get_color(), alpha=alpha)[0])

        axs[0].set_xlabel("stimulus: "+stimulusUnits)
        if snrRange is not None:
            if self.trackSnr != []:
                axs[0].set_xlim(min(self.trackSnr)-snrMargin,
                                min(self.trackSnr)+snrRange+snrMargin)
            else:
                axs[0].set_xlim(-snrMargin, snrRange+snrMargin)  # assume start at zero

        axs[0].set_ylabel("fraction correct")
        axs[0].set_ylim(-0.05, 1.05)
        # add custom entry to the legend
        lines.append(axs[0].plot(
            [], [], "--", color="black",
            label="standard deviation\nof binomial distribution", alpha=alpha)[0])
        if showLegend:
            axs[0].legend(loc=lapsegendLoc[0])
        axs[0].grid()
        plots.append(lines)

        # adaptive track
        lines = []
        lines.append(axs[1].plot(self.trackSnr, color="black")[0])
        for n, (snr, state, res) in enumerate(
                zip(self.trackSnr, self.trackState, self.trackResponse)):
            # one by one for better order of the lines handle
            lines.append(axs[1].scatter(n, snr, color=self.cmap(state)))
        for n, (snr, res) in enumerate(zip(self.trackSnr, self.trackResponse)):
            lines.append(axs[1].annotate(self.resFormat.format(res), (n, snr)))
        fits = [f(self) for f in self.fittingProcedures]
        for key, val in fits:
            lines.append(axs[1].hlines(val, 0, len(self.trackSnr) -
                         1, linestyles="dashed", label=key, color=self.cmap(i)))
        axs[1].set_ylabel("stimulus: "+stimulusUnits)
        axs[1].set_xlabel("trial ID")
        # axs[1].set_title("track")
        if len(fits) > 0:
            axs[1].legend(loc=lapsegendLoc[1])
        plots.append(lines)
        if snrRange is not None:
            if self.trackSnr != []:
                axs[1].set_ylim(min(self.trackSnr)-snrMargin,
                                min(self.trackSnr)+snrRange+snrMargin)
            else:
                axs[1].set_ylim(-snrMargin, snrRange+snrMargin)  # assume start at zero

        axs[1].set_xticks(np.arange(stop=len(self.trackSnr), step=5))
        axs[1].grid()

        # information on the states
        if self.showStatsInPlot and showStats:
            plt.sca(axs[1])
            plt.rc('text', usetex=True)
            artist = AnchoredText(r'{}'.format(self.printSettings(False)), loc=loc)
            artist.set_alpha(0)
            axs[1].add_artist(artist)

        plt.tight_layout()
        if fig is None:
            return plots
        return fig

    def plotPsy(self, start, stop, state):
        """
        return plottable data of the psychometric functions
        corresponding to the given state in range (start,stop)
        aditionally the variance of the binomial distribution of measurements with "testItemNum"
        number of testitems is given for confidence intervals

        Parameters
        ----------
        start : double
            startRange
        stop : double
            stopRange
        state : state object

        Returns
        -------
        x-values,y-values,varinance,label

        """
        snr = np.linspace(start, stop, 100)
        resp = state.psychFunc(snr)
        try:
            return snr, resp, resp*(1-resp), "psychometric function \n{}".format(state.name)
        except:
            return [], [], [], ""
