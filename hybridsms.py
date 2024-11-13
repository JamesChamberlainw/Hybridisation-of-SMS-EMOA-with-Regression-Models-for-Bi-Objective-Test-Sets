import numpy as np
import copy 

from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.population import Population
from pymoo.indicators.hv.exact import ExactHypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
from pymoo.indicators.hv import Hypervolume 
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize


# SMS-EMOA Pymoo Implementation  - this project is extended from this class and files 
# ref. https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/sms.py#L26 

# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================


from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival

class MyLeastHypervolumeContributionSurvival(LeastHypervolumeContributionSurvival):

    __model__ = None
    __model_initialised__ = False
    
    __average_error__ = 0.0

    # missing evaluations hypervolume contribution - NOT USED AT ALL BUT STILL CALCULATED FOR STATISTICAL PURPOSES
    __missing_evals_unused__ = []

    # switch between model and actual hypervolume evaluation``
    __n_model_swap__ = 20           # number before swapping
    __bool_model_eval__ = False     # current state
    __counter_model_eval__ = 1      # counter (default: 1) - lower values mean more data for first model evaluation

    # statistic variables 
    eval_counter__ = 0                  # all evaluations perfromed by the model
    eval_counter_all_potential__ = 0    # all potential evaluations even those that are not performed due to the model
    # variables for storing actual evaluations and missing evaluations that would have taken palce
    __hv_actual_over_time__ = []
    __missing_evals_unused__ = []
    # USED hypervolume contribution
    __hv_actual_contrib_over_time__ = []
    __hv_aprox_contrib_over_time__ = []
    __hv_mean_contrib_over_time__ = []

    def __init__(self, eps=10.0, num_between_model_swaps=10, model = None) -> None:
        """
            if model is None -> default behaviour is set to True
                this only exists for test purposes for comparison with original implementation
        """

        if num_between_model_swaps is not None and num_between_model_swaps > 0: # has to be greater than 0
            self.__n_model_swap__ = num_between_model_swaps

        if model is not None: # default model is None so default behaviour is set to True
            self.__model__ = model
            self.__model_initialised__ = True

        # finish with the original constructors behaviour
        super().__init__(eps=eps)

    def _do(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # get the actual front as individuals
            front = pop[front]
            front.set("rank", k)

            if len(survivors) + len(front) > n_survive:

                # normalize all the function values for the front
                F = front.get("F")
                F = normalize(F, ideal, nadir)

                # define the reference point and shift it a bit - F is already normalized!
                ref_point = np.full(problem.n_obj, 1.0 + self.eps)

                # choose the suitable hypervolume method
                clazz = ExactHypervolume # default behaviour for training the model

                hv = None

                if self.__model_initialised__:
                    if self.__bool_model_eval__ and self.__counter_model_eval__ <= self.__n_model_swap__: # true
                        hv = self.__model__.predict(F) + self.__average_error__ # - average error to get a better approximation
                        self.__counter_model_eval__ += 1

                        # counters for statistics
                        self.eval_counter_all_potential__ += 1    # counter for testing purposes
                        self.__hv_actual_over_time__.append(Hypervolume(ref_point = ref_point).do(F))       # append the actual hypervolume
                        self.__hv_actual_contrib_over_time__.append(np.NaN)                                 # append NAN as no prediction is made
                        self.__hv_aprox_contrib_over_time__.append(sum(hv))                                 # append contribution of the predicted points hypervolume (whats being calculated here)
                        self.__missing_evals_unused__.append(sum(clazz(ref_point).add(F).hvc))                         # append actual hyper volume contribution (whats being calculated by hv here hv.hvc)
                        self.__hv_mean_contrib_over_time__.append(np.mean(hv))                               # append the mean of the predicted points hypervolume

                        if self.__counter_model_eval__ >= self.__n_model_swap__:
                            self.__bool_model_eval__ = False
                            self.__counter_model_eval__ = 0
                        self.___current_hv___ = False

                    else: 
                        hv = clazz(ref_point).add(F)
                        self.__model__.fit(F, hv.hvc)
                        self.__counter_model_eval__ += 1

                        pred_hv = self.__model__.predict(F) 
                        self.__average_error__ = np.mean(hv.hvc - pred_hv) # calculate the average error for the model

                        self.__hv_actual_over_time__.append(Hypervolume(ref_point = ref_point).do(F))       # append the actual hypervolume
                        self.__hv_actual_contrib_over_time__.append(sum(hv.hvc))                            # append actual hyper volume contribution (whats being calculated by hv here hv.hvc)
                        self.__hv_mean_contrib_over_time__.append(np.mean(hv.hvc))                          # append the mean of the predicted points hypervolume
                        self.__hv_aprox_contrib_over_time__.append(np.NaN)                                  # append NAN as no prediction is made
                        self.__missing_evals_unused__.append(np.NaN)                                        # append NAN as its actually calculated so no need to append extra data

                        # counters for statistics
                        self.eval_counter__ += 1    # counter for testing purposes
                        self.eval_counter_all_potential__ += 1 # all potential evaluations even those that are not performed due to the model

                        if self.__counter_model_eval__ >= self.__n_model_swap__:
                            self.__bool_model_eval__ = True
                            self.__counter_model_eval__ = 0
                        self.___current_hv___ = True
                else:
                    hv = clazz(ref_point).add(F)
                    self.___current_hv___ = True
                    self.__hv_actual_over_time__.append(Hypervolume(ref_point = ref_point).do(F))
                    self.__hv_aprox_contrib_over_time__.append(np.NaN)
                    self.__missing_evals_unused__.append(np.NaN)

                    self.eval_counter__ += 1    # counter for testing purposes


                # current front sorted by crowding distance if splitting
                while len(survivors) + len(front) > n_survive:
                    if self.___current_hv___:
                        k = hv.hvc.argmin()
                        hv.delete(k)
                        front = np.delete(front, k)
                    else:
                        k = hv.argmin()
                        hv = np.delete(hv, k)
                        front = np.delete(front, k)

            # extend the survivors by all or selected individuals
            survivors.extend(front)

        return Population.create(*survivors)

class Hy_SMSEMOA(SMSEMOA):

    def get_extra_stats(self):
        return self.survival.__hv_actual_over_time__, self.survival.__hv_aprox_contrib_over_time__, self.survival.__hv_actual_contrib_over_time__, self.survival.__missing_evals_unused__, self.survival.__hv_mean_contrib_over_time__

# custom minminise function
def hy_minimize(problem, algorithm=Hy_SMSEMOA, termination=None, copy_algorithm=True, copy_termination=True, **kwargs):
    """
        A custom version of the minimize function from pymoo 
        
            ADDED: extra test data and statistics 
    """

    # create a copy of the algorithm object to ensure no side-effects
    if copy_algorithm:
        algorithm = copy.deepcopy(algorithm)

    # initialize the algorithm object given a problem - if not set already
    if algorithm.problem is None:
        if termination is not None:

            if copy_termination:
                termination = copy.deepcopy(termination)

            kwargs["termination"] = termination

        algorithm.setup(problem, **kwargs)

    # actually execute the algorithm
    res = algorithm.run()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm

    extra_stats = algorithm.get_extra_stats()

    res.hv_actual = extra_stats[0]                      # actual hypervolume        

    res.hv_aprox_contrib = extra_stats[1]               # approximate calculated hypervolume contribution
    res.hv_actual_contrib = extra_stats[2]              # actual      calculated hypervolume contribution
    res.hv_missing_evals_unused = extra_stats[3]        # missing   evaluations  hypervolume contribution
    res.hv_mean_contrib = extra_stats[4]                # mean of the calculated hypervolume contribution

    return res