#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python 3.12
Multistate-listener model

@author: max scharf Do 8. Mai 13:11:32 CEST 2025 maximilian.scharf_at_uol.de

credits to Arne Leyon for his help on the implementation of a correct and more efficient EM 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy.optimize as opt  # fitting most likely models
from scipy.special import comb  # binomial coefficient

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
        # see http://dx.doi.org/10.1121/1.4979580
        self.beta = slope*(1-self.gues-self.laps) / \
            ((threshold-self.gues)*(1-self.laps-threshold))
        self.alpha = thresStim + \
            np.log((1-self.laps-threshold)/(threshold-self.gues))/self.beta

        # keep information for later reference
        self.name = name
        self.threshold = threshold
        self.slope = slope
        self.thresStim = thresStim
        
    def __str__(self):
        message = (
            "\tName: {}\n"
            "\tThreshold: {:.5f}\n"
            "\tSlope: {:.5f}\n"
            "\tthresStim: {:.5f}\n"
            "\tGuessrate: {:.5f}\n"
            "\tLapsrate: {:.5f}\n"
            "\tBeta: {:.5f}\n"
            "\tAlpha: {:.5f}\n"
        ).format(
            self.name,
            self.threshold,
            self.slope,
            self.thresStim,
            self.gues,
            self.laps,
            self.beta,
            self.alpha
        )
        return message

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


class StateMatrixTestFreeThr(State):
    testItemsPerTrial = 5  # class constant for Matrix test
    @classmethod
    def initialize(cls, snr50, name=''):
        """
        Self-adapting state model for matrix-test results, with only ONE free parameter:
        the threshold of the psychometric function (proportional to alpha)
        that is optimized to the observed track.
    
        Parameters
        ----------
        snr50 : float
            SNR value at threshold (50% correct response rate).
        name : string, optional
            Name or label for the state (for informational purposes).
    
        Returns
        -------
        instance of the class
            Instance with the given snr50 for 50% correct and standard values for all other parameters.
        """
        return cls(thresStim=snr50,
                   guessRate=1 / 10,    # -> this parameter depends on the test type!
                   lapsRate=0.00,       #
                   threshold=0.5,       # -> define the threshold percentage
                   slope=0.1425,        # average slope for english/cantonese matrix test
                   name=name,
                   )

    def _nhit(self, track_response):
        """
        Convert float-valued correct-response rate to integer number of hits
        for this type of test procedure.
    
        Parameters
        ----------
        track_response : array-like of float
            Sequence of correct-response rates as floats.
    
        Returns
        -------
        nh : array of int
            Corresponding array of integer number of hits.
        """
        fh = self.testItemsPerTrial * np.asarray(track_response)
        nh = np.round(fh)  # still float valued
        if np.any(nh != fh):
            raise RuntimeError('Response rate does not match number of test items per trial.')
        return nh.astype(int)

    def prob(self, track_snr, track_response):
        """
        Probability mass of observed response, given track stimuli,
        assuming binomial probability mass of response rates.
    
        Parameters
        ----------
        track_snr : array-like of float
            Sequence of track SNR values for each trial.
        track_response : array-like of float
            Corresponding correct response rates for each trial.
            np.asarray(track_response).shape == np.asarray(track_snr).shape
    
        Returns
        -------
        p : array of float
            Array with p[n] = P{track_response[n] | track_snr[n], self}
        """
        
        nhit = self._nhit(track_response)
        nmiss = self.testItemsPerTrial - nhit
        phi = self.psychFunc(np.asarray(track_snr))
        
        #no need for exact flag for small numer of testitems here
        return comb(self.testItemsPerTrial, nhit) * phi**nhit * (1.- phi)**nmiss

    def adapt(self, s, r, w=1.):
        """
        Adapt self to a given track by maximum likelihood (ML) estimation of the free parameter.
        
        Parameters
        ----------
        s : 1D array-like of float
            Stimulus values for each trial.
        r : 1D array-like of float
            Corresponding correct response rates for each trial, each value in [0., 1.].
            Must satisfy ``len(r) == len(s)``.
        w : float or 1D array-like of float, optional
            Weight factors for each trial, values in [0, 1]. If array-like, must match length of `s` and `r`.
            ``w[n] == prob{ r[n] generated by model self at stim s[n] }``.
            Use ``w != 1`` only if there are other states. Default is 1 for all trials.
        
        Returns
        -------
        ll : float
            Scalar log-likelihood at the optimized self.
        
        Notes
        -----
        The object's properties are updated in-place according to the optimization.
        """
        
        def neg_ll(alfa):
            """
            Negative (weighted with w) sum log-likelihood of a given track, to be minimized.
            
            Parameters
            ----------
            alfa : float or array-like of float
                Tentative alpha value(s) of the psychometric function, broadcast-compatible with array `s`.
                Optimizing alpha is equivalent to optimizing the threshold
            
            Returns
            -------
            neg_log_likelihood : float
                Weighted negative sum log-likelihood of the given track.
            
            """
            phi = self.gues + (1 - self.gues - self.laps) / (1 + np.exp(-self.beta * (s - alfa)))
            lp = nhit * np.log(phi) + nmiss * np.log(1. - phi)
            
            #no need for exact flag for small numer of testitems here
            lp += np.log(comb(self.testItemsPerTrial, nhit))
            return - np.sum(w * lp)
        
        # -----------------------------------------------
        s = np.asarray(s)
        nhit = self._nhit(r)
        nmiss = self.testItemsPerTrial - nhit
        res = opt.minimize(neg_ll, self.alpha)
        if res.success:
            self.alpha = res.x[0] 
            phi = self.threshold
            self.thresStim = (self.alpha
                              - np.log((1 - self.laps - phi) / (phi - self.gues)) / self.beta)
            # Keeping self.thresStim and self.alpha consistent
            return -res.fun  # the maximal likelihood
        else:
            raise RuntimeError(res.message)

def stateLikelihood(snr, numHit, numMiss, state):
    """
    calculate the likelihood that the observed number of hits and misses was emitted by the state

    Parameters
    ----------
    snr : float
        signal to noise ratio.
    numHit : unsigned integer
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
    
    #no need for exact flag for small numer of testitems here
    return comb(numHit+numMiss, numHit)*state.likely(snr, True)**numHit * state.likely(snr, False)**numMiss

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


class PsychHMM:
    def __init__(self, A, pi, states):
        """
        Hidden Markov Model to model a response track from an adaptive psychophysical procedure,
        assuming responses are determined by more than one psychometric function. All parameters
        can be optimized to an observed track using the standard EM/Baum-Welch algorithm.
    
        Parameters
        ----------
        A : square array
            Transition probability matrix; A[i, j] = P[state_{t+1} == j | state_t == i]
        pi : array-like
            Initial state probability vector; pi[i] = P[state_0 == i]
        states : sequence of State objects
            Sequence of state objects (subclass of State). Length N = len(states) == len(pi).
            shape(A) == (N, N)
    
        """
        
        # errorcheck
        checkTransMat(A, len(states))
        if sum(pi) >= 1+1e-9 or sum(pi) <= 1-1e-9:
            raise Exception("first State distribution (pi) is not a probability vector(sum({})={})".format(
                pi, sum(pi)))
        if len(pi) != len(states):
            raise Exception("first State distribution (pi) does not match the number of provided states")
        
        self.A = np.asarray(A)
        self.pi = np.asarray(pi)
        self.states = states

    @classmethod
    def initialize(cls, states):
        """
        Create initial instance with given initial states and default initial settings
        for Markov transition and initial state probabilities.
    
        Parameters
        ----------
        states : sequence of State subclass objects
            Sequence of State objects to define the model states
    
        Returns
        -------
        instance of the class
            An instance of the class initialized with provided states and default settings
        """
        bw_a = np.ones([len(states)] * 2) / len(states)
        bw_pi = np.ones(len(states)) / len(states)
        return cls(bw_a, bw_pi, states)
    
    def __str__(self):
        message = (
            "Transition matrix:\n{}\n\n"
            "Initial state probabilities:\n{}\n\n"
            "Internal state(s):\n\n"
        ).format(
            np.array2string(self.A, precision=5, floatmode='fixed'),
            np.array2string(self.pi, precision=5, floatmode='fixed')
        )

        for s in self.states:
            message+=str(s)+"\n"
        return message
    
    def forwardBackward(self,trackSnr, trackResponse):
        """
        implements the forward-backward algorithm to calculate the forward and backward matrices

        Parameters
        ----------
        trackSnr : list of float
            presented stimulus.
        trackResponse : list of float, each value in [0., 1.].
            measured response.

        Returns
        -------
        alpha and beta matrix
        alpha.shape == beta.shape == (nTrials, nStates)
        """

        #                |transprob to A |transprob to B
        # -------------------------------------------------
        # coming from A  |  [[  x        |     x  ],
        # -------------------------------------------------
        # coming from B  |   [  x        |     x  ]]
        
        # Some alpha and beta have very small values, not scaled inside forwardBackward.
        # This is OK for short tracks, but scaling is needed in general application!

        # errorcheck
        if len(trackSnr) != len(trackResponse):
            raise Exception("number of Observations and Stimuly noes not match!")
       

        # ----------------shorthands-------------
        testItemsPerTrial=self.states[0].testItemsPerTrial
        def b(stim, resp, state):
            numHit = resp*testItemsPerTrial
            if not numHit.is_integer():
                raise Exception(
                    "response does not match the number of testitems per trial.")
            return stateLikelihood(stim, int(numHit), int(testItemsPerTrial-numHit), state)

        # ---------------------------------------

        # construct forward matrix
        alpha = np.ones([len(trackResponse), len(self.states),])  # [T,states]

        # init at t=0
        alpha[0] = list(
            map(lambda s: b(trackSnr[0], trackResponse[0], s), self.states))*self.pi

        # iteratively fill the matrix from t=1->t=N-1
        for i_t, (prev_alpha_t, stim, resp) in enumerate(zip(alpha, trackSnr[1:], trackResponse[1:]), start=1):
            for i_s, s in enumerate(self.states):
                alpha[i_t, i_s] = b(stim, resp, s)*np.inner(prev_alpha_t, self.A[:, i_s])

        # construct backward matrix and init at t=T,
        beta_rev = np.ones([len(trackResponse), len(self.states),])  # [T,states]

        # iteratively fill the matrix from t=N-2->t=0
        # define t and beta backwards!
        for i_t, (prev_beta_t, stim, resp) in enumerate(zip(beta_rev, reversed(trackSnr[1:]), reversed(trackResponse[1:])), start=1):
            for i_s, s in enumerate(self.states):
                beta_rev[i_t, i_s] = np.inner(prev_beta_t*self.A[i_s, :],
                                              list(map(lambda s: b(stim, resp, s), self.states)))

        # likelihood of observation: sum(alpha[-1, :])
        return alpha, beta_rev[::-1]


    def viterbi(self,trackSnr, trackResponse):
        """
        viterbi algorithm

        Parameters
        ----------
        trackSnr : list of float
            presented stimulus.
        trackResponse : list of float, each value in [0., 1.].
            measured response.

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
        
        # ----------------shorthands-------------
        testItemsPerTrial=self.states[0].testItemsPerTrial
        def _b(stim, resp, state):
            numHit = resp*testItemsPerTrial
            if not numHit.is_integer():
                raise Exception(
                    "response does not match the number of testitems per trial.")
            return stateLikelihood(stim, int(numHit), int(testItemsPerTrial-numHit), state)
        b=np.array([[_b(stim, resp, state) for state in self.states]
                     for stim, resp in zip(trackSnr, trackResponse)])
        # ---------------------------------------

        # init
        delta = [self.pi*b[0]]
        psi = []

        # inductive
        for b_t in b[1:]:
            psi.append([int(np.argmax([delta[-1][i]*self.A[i, j] for i in range(len(self.pi))]))
                       for j in range(len(self.pi))])
            delta.append(np.array([max([delta[-1][i]*self.A[i, j] for i in range(len(self.pi))])
                         for j in range(len(self.pi))])*b_t)

        # ending
        stateSeq = []  # in reverse
        stateSeq.append(int(np.argmax(delta[-1])))

        # backtrack in reverse
        psi.reverse()
        for p in psi:
            stateSeq.append(p[stateSeq[-1]])

        stateSeq.reverse()
        return stateSeq, np.log(max(delta[-1]))

    def learn(self, trackSnr, trackResponse,
              min_step=1e-5, max_iter=200):
        """
        Optimize parameters using the standard Baum-Welch algorithm.
    
        Parameters
        ----------
        trackSnr : list of float
            Presented stimulus for each trial.
        trackResponse : list of float
            Corresponding responses for each trial, coded as fraction of correct trial responses.
        min_step : float, optional
            Minimum data log-likelihood improvement required for the learning iterations to continue.
        max_iter : int, optional
            Maximum number of iterations, regardless of result.
    
        Returns
        -------
        log_likelihood : float
            Scalar log-likelihood of the observed track given the optimized model.
    
        Notes
        -----
        All internal parameters are optimized for the given track.
        Might be further simplified, but works OK
        """

        # -------------------- shorthands ---------------
        testItemsPerTrial = self.states[0].testItemsPerTrial
        def b(stim, resp, state):
            numHit = resp * testItemsPerTrial
            if not numHit.is_integer():
                raise Exception(
                    "response does not match the number of testitems per trial.")
            return stateLikelihood(stim, int(numHit),
                                   int(testItemsPerTrial - numHit),
                                   state)

        # a: transMat[startState_id, endState_id]

        def prob(t):
            # shorthand notation for the probability to be in state i at time t and j at time t+1
            # index with prob(t)[i,j]
            # calculated using OLD Markov A matrix
            tmp = np.ones([len(self.states)] * 2)
            for i, s_i in enumerate(self.states):
                for j, s_j in enumerate(self.states):
                    tmp[i, j] = alpha[t, i] * self.A[i, j] * beta[t + 1, j] * \
                                b(trackSnr[t + 1], trackResponse[t + 1], s_j)
            return tmp
        # -------------------------------------------------

        logLikelihood = -np.inf  # needed only for first iteration
        for n in range(max_iter):
            alpha, beta = self.forwardBackward(trackSnr, trackResponse)

            # b_0 = [b(trackSnr[0], trackResponse[0], state_i)
            #        for state_i in self.states]
            # sum_tot_alpha = np.sum(alpha[-1])
            # sum_tot_beta = np.sum(beta[0] * self.pi * b_0)
            # if not np.isclose(sum_tot_alpha, sum_tot_beta):
            #     raise RuntimeError('Error in forward-backward function!')

            # NOTE: All these results calculated with OLD model properties, before any update!

            # calculate intermediate results
            gamma = np.array([alpha_t*beta_t/np.inner(alpha_t, beta_t)
                              for alpha_t, beta_t in zip(alpha, beta)])

            xi = []
            for t in range(len(alpha)-1):
                divident = prob(t)
                xi.append(divident/np.sum(divident))
            # xi = list of 2D arrays
            # xi[t][i, j] == P[state_t = i AND state_{t+1} = j | complete track]

            # update Markov properties:
            bw_pi = gamma[0]
            # bw_pi[i] = P[state_0 = i | complete track]
            bw_a = np.sum(xi, axis=0)/np.sum(gamma[: -1], axis=0)[:, np.newaxis]  # [i,j]
            # bw_a[i,j] = P[state_{t+1} == j | state_t == i, complete track, old model]

            # normalize (pi and a are not EXACT probability matrices because of machine precision)
            # but they are very close already, if everything is correct!
            bw_pi /= sum(bw_pi)
            bw_a /= np.sum(bw_a, 1)[:, np.newaxis]
            self.pi = bw_pi
            self.A = bw_a

            # update state properties; NOTE: Independently for each state.
            for (state_i, gamma_i) in zip(self.states, gamma.T):
                state_i.adapt(trackSnr, trackResponse, gamma_i)
            # Here, we need not know how the states do it!

            # convergence criterion
            logLikelihood_new = np.log(sum(alpha[-1, :]))
            if n != 0 and abs(logLikelihood_new-logLikelihood) < min_step:
                return logLikelihood_new
            else:
                logLikelihood = logLikelihood_new
        warnings.warn(f'Baum-Welch needed more than {max_iter} iterations. '
                      + 'Result may be inaccurate.')
        return logLikelihood_new



def SiPGOF(trackSnr, trackResponse, plot=True):
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
        1stModel:
            1 state model
        2stModel
            2 state model


    """

    # Two-state model: using standard Baum-Welch
    x2 = [np.quantile(trackSnr, 0.25), np.quantile(trackSnr, 0.75)]
    st_2 = [StateMatrixTestFreeThr.initialize(x2_i,name = f'TwoState_{i}')
            for (i, x2_i) in enumerate(x2)]
    model_2 = PsychHMM.initialize(st_2)
    ll_2 = model_2.learn(trackSnr, trackResponse)

    # Single-state model: no HMM needed for training, faster without Baum-Welchm-Welch
    x1=np.quantile(trackSnr, 0.5)
    st1 = StateMatrixTestFreeThr.initialize(x1,name='SingleState')
    model_1 = PsychHMM(np.ones((1,1)), np.ones(1), [st1])
    # = equivalent one-state HMM with trivial Markov properties
    ll_1 = st1.adapt(trackSnr, trackResponse)

    fig = None
    if plot:
        plt.close('all')  # important if inlinefigures are disabled
        fig, axs = plt.subplots(
            2, 2, gridspec_kw={"height_ratios": [1, 1.5]}, figsize=[20, 10])
        dummy = Person.fromData(model_1, ll_1,trackSnr, trackResponse, "singleState")
        dummy.plot(axs=axs[:, 0])
        dummy = Person.fromData(model_2, ll_2,trackSnr, trackResponse, "twoState")
        dummy.plot(axs=axs[:, 1])

    return {
        1: ll_1,
        2: ll_2,
        "1stModel": model_1,
        "2stModel": model_2,
        "plot": fig,
    }




class Person:

    def __init__(self, statesList, trackState, transMat,
                 trackResponse, trackSnr, label="undefined"):
        self.reset()
        self.transMat = transMat
        self.states = statesList
        self.trackResponse = trackResponse
        self.trackSnr = trackSnr
        self.trackState = trackState
        self.label = label

        checkTransMat(self.transMat, len(self.states))

    @ classmethod
    def fromData(cls, hmm, ll,trackSnr, trackResponse, label=""):
        """
        Create Display module for the analysis of a PsychHMM instance and related track data.
        
        Parameters
        ----------
        hmm : PsychHMM instance
            Hidden Markov Model optimized to the given track.
        ll : float
            Log-likelihood for this model.
        trackSnr : 1D array-like of float
            SNR values for the observed track.
        trackResponse : 1D array-like of float
            Observed responses for the corresponding SNR values.
            The track is needed here only to run Viterbi for display.
        label : name of the model (optional)
        
        Returns
        -------
        result : dict
            Dictionary with elements as generated by the function `modelA`.
        
        """
        if len(hmm.states) == 1:  # one-state model
            best_state_seq = np.zeros(len(trackSnr), dtype=int)
            viterbi_ll = ll
        else:  # two-state model
            best_state_seq, viterbi_ll = hmm.viterbi(trackSnr, trackResponse)
        return cls(hmm.states, best_state_seq, hmm.A, trackResponse, trackSnr, label)
    

    def add_fit(self, fitFunction):
        """add a fitting function for use in the plots"""
        self.fittingProcedures.append(fitFunction)

    def reset(self):
        '''reset the person to initialized state'''
        # default settings
        self.cmap = plt.get_cmap("tab10")
        self.resFormat = "{0:1.2f}"  # response Formatting
        self.numberOfAxes = 2
        self.showStatsInPlot = True  # can turn of the latex rendering
        self.fittingProcedures = [] # list of fitting procedures

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
    
    def __str__(self):
        return self.printSettings(toConsole=False,tex=False)

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
            return str(dat)

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
