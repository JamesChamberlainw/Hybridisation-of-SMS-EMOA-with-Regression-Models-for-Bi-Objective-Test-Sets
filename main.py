import numpy as np

from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.visualization.scatter import Scatter

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

    _ot_evals = []

    __n_model_swap__ = 10

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
                        self._ot_evals.append(Hypervolume(ref_point = ref_point).do(F))
                        if self.__counter_model_eval__ >= self.__n_model_swap__:
                            self.__bool_model_eval__ = False
                            self.__counter_model_eval__ = 0
                            print("Now Based on Original Evaluated")
                        self.___current_hv___ = False

                    else: 
                        hv = clazz(ref_point).add(F)
                        self.__model__.fit(F, hv.hvc)
                        self.__counter_model_eval__ += 1

                        self._ot_evals.append(Hypervolume(ref_point = ref_point).do(F))


                        # counters for statistics
                        self.eval_counter__ += 1    # counter for testing purposes
                        self.eval_counter_all_potential__ += 1 # all potential evaluations even those that are not performed due to the model

                        if self.__counter_model_eval__ >= self.__n_model_swap__:
                            self.__bool_model_eval__ = True
                            self.__counter_model_eval__ = 0
                            print("Now Based on Model Evaluated")
                        
                        self.___current_hv___ = True
                else:
                    hv = clazz(ref_point).add(F)
                    self.___current_hv___ = True

                    self.eval_counter__ += 1    # counter for testing purposes


                # current front sorted by crowding distance if splitting
                while len(survivors) + len(front) > n_survive:
                    if self.___current_hv___:
                        k = hv.hvc.argmin()
                        hv.delete(k)
                        front = np.delete(front, k)
                    else:
                        k = hv.argmin()
                        # remove the individual from the front array
                        hv = np.delete(hv, k)
                        # hv = hv.delete(hv.indexof(k))
                        front = np.delete(front, k)

            # extend the survivors by all or selected individuals
            survivors.extend(front)

            # print(f"Current Front: {k} - Survivors: {len(survivors)} - Eval Counter: {self.eval_counter__} out of {self.eval_counter_all_potential__}")

            # if last display overtime evaluations
            if self.eval_counter_all_potential__ == 120:
                print("Overtime Evaluations: ")
                # 139
                x = np.arange(0, len(self._ot_evals), 1)
                plt.plot(x, self._ot_evals)
                plt.show()


        return Population.create(*survivors)
        # return super()._do(problem, pop, *args, n_survive=n_survive, ideal=ideal, nadir=nadir, **kwargs) # original implementation 

# The Problem 
problem = get_problem("zdt3", n_var = 30)

# model = GaussianProcessRegressor()
# model = linear_model.BayesianRidge()
# model = RandomForestRegressor()
# model = SVR(kernel='linear')
model = SVR(kernel='poly')          # incredible resutls with 4/5 with major coverage (tiny gap) with 1/5 with minor coverage
# model = SVR(kernel='rbf')         # all found full coverage on 40% and the rest with partial with one poor coverage
# model = SVR(kernel='sigmoid')   # good results 4/5 zdt3 n_var = 30
# model = SGDRegressor()
# model = linear_model.LinearRegression()
algorithm = SMSEMOA(survival=MyLeastHypervolumeContributionSurvival(model=model, num_between_model_swaps=10)) # key argument for this project DO NOT FORGET!!! 
# algorithm = SMSEMOA() # default 

# Run the Optimization
res = minimize(problem,
               algorithm,
               termination=('n_gen', 140),
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