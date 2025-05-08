#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python 3.12
Multistate-listener model

@author: max scharf Do 8. Mai 13:11:32 CEST 2025 maximilian.scharf_at_uol.de
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy.optimize as opt  # fitting most likely models
import warnings
import math


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
        guessRate  : float
            context free forced choice: propto 1/number of choices
        lapsRate: float
            pobabbility of making errors at optimal presentation
        threshold : float
            percentage correct (0,1) at which the Threshold is probed
        slope : float
            slope of psychometric function at threshold stmulation
        thresStim : float
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
        snr : float
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


def stateLikelihood(snr, numHit, numMiss, state):
    """
    calculate the likelihood that the observed number of hits and misses was emitted by the state

    Parameters
    ----------
    snr : float
        signal to noise ratio.
    numHit : unsigned interger
        response # correct anwser during a single trial.
    numMiss : unsigned interger
        response # incorrect answer during a single trial.
    statesList: state object
        list of states.

    Returns
    -------
    float
        likelihood.

    """
    tmpl = 1
    for n in range(numHit):
        tmpl *= state.likely(snr, True)  # may sometimes produce nan
    for n in range(numMiss):
        tmpl *= state.likely(snr, False)  # may sometimes produce nan

    # include the binomial coefficient
    return tmpl*math.factorial(numHit+numMiss)/(math.factorial(numHit)*math.factorial(numMiss))


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


def forwardBackward(trackSnr, trackResponse, states, transMat, firstStateDistr, testItemsPerTrial):
    """
    implements the forward-backward algorithm to calculate the forward and backward matrices
    given the list of states-objects, initial state-distribution and transitionmatrix

    Parameters
    ----------
    trackSnr : list of float
        presented stimulus.
    trackResponse : list of float
        measured response.
    states : states object
        HMM states.
    transMat : estimate of the transition matrix
        has to match the numer of provided states.
    firstStateDistr : probability of being in a given state at the first observation
        The asumed starting state

    Returns
    -------
    alpha and beta matrix

    """

    #                |transprob to A |transprob to B
    # -------------------------------------------------
    # coming from A  |  [[  x        |     x  ],
    # -------------------------------------------------
    # coming from B  |   [  x        |     x  ]]

    # errorcheck
    checkTransMat(transMat, len(states))
    if len(trackSnr) != len(trackResponse):
        raise Exception("number of Observations and Stimuly noes not match!")
    if sum(firstStateDistr) >= 1+1e-9 or sum(firstStateDistr) <= 1-1e-9:
        raise Exception("first State distr not a probability vector(sum({})={})".format(
            firstStateDistr, sum(firstStateDistr)))
    if len(firstStateDistr) != len(states):
        raise Exception("first State distr does not match the number of provided states")

    # shorthands
    def b(stim, resp, state):
        numHit = resp*testItemsPerTrial
        if not numHit.is_integer():
            raise Exception(
                "response does not match the number of testitems per trial.")
        return stateLikelihood(stim, int(numHit), int(testItemsPerTrial-numHit), state)

    # a: transMat[startState_id, endState_id]

    # ---------------

    # construct forward matrix
    alpha = np.ones([len(trackResponse), len(states),])  # [T,states]

    # init at t=0
    alpha[0] = list(
        map(lambda s: b(trackSnr[0], trackResponse[0], s), states))*firstStateDistr

    # iteratively fill the matrix from t=1->t=N-1
    for i_t, (prev_alpha_t, stim, resp) in enumerate(zip(alpha, trackSnr[1:], trackResponse[1:]), start=1):
        for i_s, s in enumerate(states):
            alpha[i_t, i_s] = b(stim, resp, s)*np.inner(prev_alpha_t, transMat[:, i_s])

    # ---------------

    # construct backward matrix and init at t=T,
    beta_rev = np.ones([len(trackResponse), len(states),])  # [T,states]

    # iteratively fill the matrix from t=N-2->t=0
    # define t and beta backwards!
    for i_t, (prev_beta_t, stim, resp) in enumerate(zip(beta_rev, reversed(trackSnr[1:]), reversed(trackResponse[1:])), start=1):
        for i_s, s in enumerate(states):
            beta_rev[i_t, i_s] = np.inner(prev_beta_t*transMat[i_s, :],
                                          list(map(lambda s: b(stim, resp, s), states)))

    # log-likelihood of observation: np.log(sum(alpha[-1, :]))
    return alpha, beta_rev[::-1]


def viterbi(a, b, pi):
    """
    viterbi algorithm
    updated 7.05

    Parameters
    ----------
    a : transition matrix
    b : emition probability at each time step
    pi : initial distribution

    Returns
    -------
    most likely state sequence.
    log-likelihood of this sequence (should match the result of the forward algorithm)

    """

    #                |transprob to A |transprob to B
    # -------------------------------------------------
    # coming from A  |  [[  x        |     x  ],
    # -------------------------------------------------
    # coming from B  |   [  x        |     x  ]]

    # init
    delta = [pi*b[0]]
    psi = []

    # inductive
    for b_t in b:
        psi.append([int(np.argmax([delta[-1][i]*a[i, j] for i in range(len(pi))]))
                   for j in range(len(pi))])
        delta.append(np.array([max([delta[-1][i]*a[i, j] for i in range(len(pi))])
                     for j in range(len(pi))])*b_t)

    # ending
    stateSeq = []  # in reverse
    stateSeq.append(int(np.argmax(delta[-1])))

    # backtrack in reverse
    psi.reverse()
    for p in psi:
        stateSeq.append(p[stateSeq[-1]])

    stateSeq.reverse()
    return stateSeq, np.log(max(delta[-1]))


# global variables of the transition matrix and initial distribution for baumwelch:
# hold results from previous run to initialize next run with a better estimate for faster convergence
bw_a = None
bw_pi = None


def baumWelch(trackSnr, trackResponse, testItemsPerTrial, states, useViterbi=False):
    """
    implement the Baum-Welch algorithm to compute (some of) the maximum likelihood HMM parameters:
        transition matrix and imitial state-distribution

    Parameters
    ----------
    trackSnr : list of float
        presented stimulus.
    trackResponse : list of float
        measured response.
    states : states object
        HMM states.
    transMat : estimate of the transition matrix
        has to match the numer of provided states.
    firstStateDistr : probability of being in a given state at the first observation
        The asumed starting state
    useViterbi: bool DEFAULT=False
        include an estimate of the most likely state sequence in the results.

    Returns
    -------
    dict
    ll:
        log-likelihood of this sequence (with forward algorithm)

    sl:
        list of most likely state-sequence (only if viterbi==True)
    pc:
        dict of Percentual content sorted by statename:(only if viterbi==True)
            "how often the state was present according to this prediction"
    vll:
        log-likelihood of this sequence (with viterbi algorithm, only if viterbi==True)
    a:
        Transitionmatrix
    """

    # shorthands
    def b(stim, resp, state):
        numHit = resp*testItemsPerTrial
        if not numHit.is_integer():
            raise Exception(
                "response does not match the number of testitems per trial.")
        return stateLikelihood(stim, int(numHit), int(testItemsPerTrial-numHit), state)
    # a: transMat[startState_id, endState_id]

    def prob(t):
        # shorthand notation for the probability to be in state i at time t and j at time t+1
        # index with prob(t)[i,j]
        tmp = np.ones([len(states)]*2)
        for i, s_i in enumerate(states):
            for j, s_j in enumerate(states):
                tmp[i, j] = alpha[t, i]*bw_a[i, j]*beta[t+1, j] * \
                    b(trackSnr[t+1], trackResponse[t+1], s_j)
        return tmp

    # initialize with defaults or use results from previous run
    global bw_a
    if bw_a is not None and np.shape(bw_a) == (len(states), len(states)):
        pass
    else:
        bw_a = np.ones([len(states)]*2)/len(states)
    global bw_pi
    if bw_pi is not None and len(bw_pi) == len(states):
        pass
    else:
        bw_pi = np.ones(len(states))/len(states)

    def runViterbi(a, pi):
        sl, vll = viterbi(a, np.array([[b(stim, resp, state) for state in states]
                                       for stim, resp in zip(trackSnr, trackResponse)]), pi)
        pc = {}
        for e, s in enumerate(states):
            pc[s.name] = sl.count(e)
        return {"ll": logLikelihood, "sl": sl, "pc": pc, "vll": vll, "a": bw_a}

    # remaining parameters
    epsilon = 1e-5  # smallest logLikelihood increase that counts as convergence
    nMax = 100
    logLikelihood = None

    for n in range(nMax):

        alpha, beta = forwardBackward(trackSnr, trackResponse, states,
                                      bw_a, bw_pi, testItemsPerTrial)

        # calculate intermediate results
        gamma = [alpha_t*beta_t/np.inner(alpha_t, beta_t) for alpha_t, beta_t in zip(alpha, beta)]
        xi = []
        for t in range(len(alpha)-1):
            divident = prob(t)
            xi.append(divident/np.sum(divident))

        # update variables:
        bw_pi = gamma[0]
        bw_a = np.sum(xi, axis=0)/np.sum(gamma[: -1], axis=0)[:, np.newaxis]  # [i,j]

        # normalize (pi and a are not probability matrices because of machine precision)
        bw_pi /= sum(bw_pi)
        bw_a /= np.sum(bw_a, 1)[:, np.newaxis]

        # convergence criterion
        logLikelihood_new = np.log(sum(alpha[-1, :]))
        if n != 0 and abs(logLikelihood_new-logLikelihood) < epsilon:
            # print("converged in n={} steps".format(n))
            if not useViterbi:
                return {"ll": logLikelihood_new, "sl": None, "pc": None, "vll": None, "a": bw_a}
            else:
                return runViterbi(bw_a, bw_pi)
        else:
            logLikelihood = logLikelihood_new

    # print("not converged")
    if not useViterbi:
        return {"ll": logLikelihood, "sl": None, "pc": None, "vll": None, "a": bw_a}
    else:
        return runViterbi(bw_a, bw_pi)


def modelA(trackSnr, trackResponse, params, optimizerMode=False):
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
                            guessRate=1 / 10,  # ->this parameter depends on the test type!
                            lapsRate=0.00,  #
                            threshold=0.5,  # ->define the threshold percentage
                            slope=0.1425,     # average slope for english/cantonese matrix test
                            )  # -> 1 remaining deg. of freedom per internal state (SRT50)

    return {"st": estimStates, **baumWelch(trackSnr,
                                           trackResponse,
                                           testItemsPerTrial,
                                           estimStates,
                                           not optimizerMode)}


def mostLikelyModelA(trackSnr, trackResponse, plot=True):
    """return the likelihood of both a single-state-model
    and a two-state-model to describe the provided track.
    This is the procedure, which is used in the JASA article:
        A consistency measure for psychometric measurements

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
        return model(trackSnr, trackResponse, params=x, optimizerMode=True)["ll"]
    x2 = [np.quantile(trackSnr, 0.25), np.quantile(
        trackSnr, 0.75), ]  # turningpoint
    optres2 = opt.minimize(lambda x: -model2(x), x2)

    # singleState
    def model1(x):
        return model(trackSnr, trackResponse, params=x, optimizerMode=True)["ll"]
    x1 = [np.quantile(trackSnr, 0.5)]  # turningpoint
    optres1 = opt.minimize(lambda x: -model1(x), x1)

    fig = None
    if plot:
        plt.close('all')  # important if inlinefigures are disabled
        fig, axs = plt.subplots(
            2, 2, gridspec_kw={"height_ratios": [1, 1.5]}, figsize=[20, 10])
        dummy = Person.fromData(trackSnr, trackResponse, "singleState")
        tmp = model(trackSnr, trackResponse, optres1.x)
        dummy.states = tmp["st"]
        dummy.trackState = tmp["sl"]
        dummy.transMat = tmp["a"]
        dummy.plot(axs=axs[:, 0])

        dummy = Person.fromData(trackSnr, trackResponse, "twoState")
        tmp = model(trackSnr, trackResponse, optres2.x)
        dummy.states = tmp["st"]
        dummy.trackState = tmp["sl"]
        dummy.transMat = tmp["a"]
        dummy.plot(axs=axs[:, 1])

    deltaThres = model(trackSnr, trackResponse, optres2.x)[
        "st"][0].thresStim - model(trackSnr, trackResponse, optres2.x)["st"][0].thresStim
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


def checkTransMat(transMat, dim, epsilon=1e-9):
    """
    raises warning or exception if mistakes are found

    Parameters
    ----------
    transMat : matrix
    dim : needed dimensions
    epsilon: float
        if the sums of rowas equal 1 with an accuracy of epsilon, raise no error

    Returns
    -------
    None.

    """

    # errorcheck transitionmatrix
    for n, s in enumerate(np.sum(transMat, 1)):
        if s < 1-epsilon or s > 1+epsilon:
            raise Exception(
                "transitionmatrix not a probability matrix (epsilon={}):".format(epsilon) +
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
        snr : float
            signal to noise ratio of test.
        nTestItem : integer
            how many trials per test. default=1

        Returns
        -------
        float
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

    def plot(self, axs=None, figsize=[17, 14], loc="upper right", stimulusUnits="SNR/dB", showStdBinom=True, showStats=True, snrRange=None, showLegend=True, heightRatios=[1, 1.5]):
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
            try:
                plt.sca(axs[1])
                plt.rc('text', usetex=True)
                artist = AnchoredText(r'{}'.format(self.printSettings(False)), loc=loc)
                artist.set_alpha(0)
                axs[1].add_artist(artist)
            except:
                pass  # may fail if latex interpreter or pandas is not present: fail gently

        plt.tight_layout()
        if fig is None:
            return plots
        return fig

    def printSettings(self, toConsole=True, tex=True):
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
        import pandas as pd

        # errorcheck
        if self.states == []:  # there are no known internal states
            return "data: {}".format(self.lastTestName)

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
                                ["stimulus at threshold",
                                 "threshold",
                                 "slope/ \%/1",
                                 "guessRate",
                                 "lapsRate",
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

        if tex:
            # dat.style.format("{{:2.2f}}".format)
            text = dat.style.to_latex()
            # romove some stuff so that it compiles with no hassle
            return text.replace('\\toprule', ' ').replace('\\midrule', ' ').replace('\\bottomrule', ' ').replace("\n", " ")
        else:
            return dat.style.to_latex()

    def plotPsy(self, start, stop, state):
        """
        return plottable data of the psychometric functions
        corresponding to the given state in range (start,stop)
        aditionally the variance of the binomial distribution of measurements with "testItemNum"
        number of testitems is given for confidence intervals

        Parameters
        ----------
        start : float
            startRange
        stop : float
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
