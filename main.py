import numpy as np
import copy 

from pymoo.problems import get_problem
# from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.visualization.scatter import Scatter

# non dominated sorting
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import time

# models
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor

# randomForrestReg
from sklearn.ensemble import RandomForestRegressor

# SVR polynomial
from sklearn.svm import SVR

# stockastic gradient descent
from sklearn.linear_model import SGDRegressor

# plotting
import matplotlib.pyplot as plt

# LeastHypervolumeContributionSurvival
from pymoo.core.population import Population
from pymoo.indicators.hv.exact import ExactHypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize

# indicator
from pymoo.indicators.hv import Hypervolume 

# hypervolume
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

    # hypervolume actual - NOT USED AT ALL BUT STILL CALCULATED FOR STATISTICAL PURPOSES
    __hv_actual_over_time__ = []
    
    # USED hypervolume contribution
    __hv_actual_contrib_over_time__ = []
    __hv_aprox_contrib_over_time__ = []
    __hv_mean_contrib_over_time__ = []

    # missing evaluations hypervolume contribution - NOT USED AT ALL BUT STILL CALCULATED FOR STATISTICAL PURPOSES
    __missing_evals_unused__ = []


    __n_model_swap__ = 20

    __bool_model_eval__ = False
    __counter_model_eval__ = 1 # stats at so it swaps on the 10th evaluation

    # statistic variables 
    eval_counter__ = 0                  # all evaluations perfromed by the model
    eval_counter_all_potential__ = 0    # all potential evaluations even those that are not performed due to the model

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
                        hv = self.__model__.predict(F)
                        self.__counter_model_eval__ += 1

                        # counters for statistics
                        self.eval_counter_all_potential__ += 1    # counter for testing purposes
                        self.__hv_actual_over_time__.append(Hypervolume(ref_point = ref_point).do(F))       # append the actual hypervolume
                        self.__hv_actual_contrib_over_time__.append(np.NaN)                                 # append NAN as no prediction is made
                        self.__hv_aprox_contrib_over_time__.append(sum(hv))                                 # append contribution of the predicted points hypervolume (whats being calculated here)
                        temp_hvc_values = clazz(ref_point).add(F).hvc
                        self.__missing_evals_unused__.append(sum(temp_hvc_values))                         # append actual hyper volume contribution (whats being calculated by hv here hv.hvc)
                        self.__hv_mean_contrib_over_time__.append(np.mean(hv))                               # append the mean of the predicted points hypervolume

                        # print(clazz(ref_point).add(F).hvc.shape, ", ", hv.shape)  # tested and they are all the same shape - only difference is error builds up over time and with 62 with a minor amount of error will result in a large error in the end 
                        # aprox / actual
                        _____temp = np.subtract(temp_hvc_values, hv)
                        print(_____temp.max(), ", ", _____temp.min(), ", ", _____temp.mean(), ", ", _____temp.std())
                        print(sum(abs(_____temp)))
                        

                        if self.__counter_model_eval__ >= self.__n_model_swap__:
                            self.__bool_model_eval__ = False
                            self.__counter_model_eval__ = 0
                            print("Now Based on Original Hypervolume Evaluation")
                        self.___current_hv___ = False

                    else: 
                        hv = clazz(ref_point).add(F)
                        self.__model__.fit(F, hv.hvc)
                        self.__counter_model_eval__ += 1

                        self.__hv_actual_over_time__.append(Hypervolume(ref_point = ref_point).do(F))       # append the actual hypervolume
                        self.__hv_actual_contrib_over_time__.append(sum(hv.hvc))                            # append actual hyper volume contribution (whats being calculated by hv here hv.hvc)
                        self.__hv_mean_contrib_over_time__.append(np.mean(hv.hvc))                               # append the mean of the predicted points hypervolume
                        self.__hv_aprox_contrib_over_time__.append(np.NaN)                                  # append NAN as no prediction is made
                        self.__missing_evals_unused__.append(np.NaN)                                        # append NAN as its actually calculated so no need to append extra data

                        # counters for statistics
                        self.eval_counter__ += 1    # counter for testing purposes
                        self.eval_counter_all_potential__ += 1 # all potential evaluations even those that are not performed due to the model

                        if self.__counter_model_eval__ >= self.__n_model_swap__:
                            self.__bool_model_eval__ = True
                            self.__counter_model_eval__ = 0
                            print("Now Based on Model Evaluation")
                        
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
            # actual hypervolume, approximate calculated hypervolume, actual hypervolume contribution, missing evaluations hypervolume contribution


# custom minminise function
def hy_minimize(problem, algorithm, termination=None, copy_algorithm=True, copy_termination=True, **kwargs):
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


# The Problem 
problem = get_problem("zdt3", n_var = 30)

# model = GaussianProcessRegressor()
# model = linear_model.BayesianRidge()
# model = RandomForestRegressor()
# model = SVR(kernel='linear')
# model = SVR(kernel='poly')          # incredible resutls with 4/5 with major coverage (tiny gap) with 1/5 with minor coverage
# model = SVR(kernel='rbf')         # all found full coverage on 40% and the rest with partial with one poor coverage
# model = SVR(kernel='sigmoid')   # good results 4/5 zdt3 n_var = 30
model = SGDRegressor()
# model = linear_model.LinearRegression()
algorithm = Hy_SMSEMOA(survival=MyLeastHypervolumeContributionSurvival(model=model, num_between_model_swaps=10)) # key argument for this project DO NOT FORGET!!! 
# algorithm = SMSEMOA() # default 

# Run the Optimization
res = hy_minimize(problem,
               algorithm,
               termination=('n_gen', 100),
               seed=1,
               save_history=True,
               verbose=True)

# Result Visualization
plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()

# decision space values
X = res.pop.get("X")
# objective space values
F = res.pop.get("F")


# plot hv actual and hv aprox
x = np.arange(0, len(res.hv_actual), 1)
plt.plot(x, res.hv_actual_contrib, label="Hypervolume Contrib Actual")
plt.plot(x, res.hv_aprox_contrib, label="Hypervolume Contrib Aprox")
plt.plot(x, res.hv_missing_evals_unused, label="Real Hypervolume Contrib Missing Evals")
plt.plot(x, res.hv_mean_contrib, label="(used values) Mean Hypervolume Contrib")
plt.legend()
plt.show()


    # res.hv_actual = extra_stats[0]                      # actual hypervolume        
    # res.hv_aprox_contrib = extra_stats[1]               # approximate calculated hypervolume contribution
    # res.hv_actual_contrib = extra_stats[2]              # actual      calculated hypervolume contribution
    # res.hv_missing_evals_unused = extra_stats[3]        # missing   evaluations  hypervolume contribution


# D = [5, 10, 30] # number of decision variables (hav to test for each value)
# M = 2           # number of objectives (fixed due to the benchmark problem)

# for d in D:
#     problem = get_problem("zdt6", n_var=d)
#     # algorithm = NSGA2() # NSGA-II
#     algorithm = HySMSEMOA() # SMS-EMOA
#     res = minimize(problem,
#                    algorithm,
#                    ('n_gen', 100),
#                    seed=1,
#                    verbose=True,
#                    save_history=True)
    
#     plot = Scatter()
#     plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
#     plot.add(res.F, color="red")
#     plot.show()


# treat evaluations as time - as less evals = better 