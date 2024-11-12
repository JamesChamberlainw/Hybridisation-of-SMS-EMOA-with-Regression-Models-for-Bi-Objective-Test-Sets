import numpy as np
import pymoo as pm

from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.visualization.scatter import Scatter

# models
from sklearn import linear_model

# ridge 
from sklearn.linear_model import RidgeClassifier


# needed for MyLeastHypervolumeContributionSurvival
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.docs import parse_doc_string
from pymoo.indicators.hv.exact import ExactHypervolume
from pymoo.indicators.hv.exact_2d import ExactHypervolume2D
from pymoo.indicators.hv.monte_carlo import ApproximateMonteCarloHypervolume
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function
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

    __bool_model_eval__ = False
    __counter_model_eval__ = 0

    def __init__(self, eps=10.0) -> None:
        super().__init__(eps=eps)

    def _do(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):

        if not self.__model_initialised__:
            print("Model Initialised")
            self.__model__ = linear_model.LinearRegression()
            self.__model_initialised__ = True

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # the number of objectives
        _, n_obj = F.shape

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

                _, n_obj = F.shape

                # choose the suitable hypervolume method
                clazz = ExactHypervolume
                if n_obj == 2:
                    clazz = ExactHypervolume2D
                elif n_obj > 3:
                    clazz = ApproximateMonteCarloHypervolume


                ## CHANGES MADE HERE 
                hv = None

                if self.__model_initialised__:
                    if self.__bool_model_eval__ and self.__counter_model_eval__ <= 10: # true
                        print("Model Evaluated")
                        hv = self.__model__.predict(F)
                        self.__counter_model_eval__ += 1
                        if self.__counter_model_eval__ <= 10:
                            self.__bool_model_eval__ = False
                            self.__counter_model_eval__ = 0
                    else: 
                        # print("Fitting New Data", self.__counter_model_eval__)
                        # print(ref_point)
                        # print(F.shape)
                        # print(len(F))

                        hv = clazz(ref_point).add(F)

                        # print("hv:", hv.hv)
                        # print("hvc:", len(hv.hvc))

                        # print(type(F))
                        # print(type(hv.hvc))

                        # crash here (as incompatible at the moment)
                        self.__model__.fit(F, hv.hvc)
                        self.__counter_model_eval__ += 1
                

                        if self.__counter_model_eval__ >= 10:
                            print("over 10")
                            self.__bool_model_eval__ = True
                            self.__counter_model_eval__ = 0
                else:
                    print("Model Not Initialised: Original Behaviour")
                    hv = clazz(ref_point).add(F)



                ## END OF CHANGES

                # calculate the hypervolume contribution 
                # hv = clazz(ref_point).add(F)

                # current front sorted by crowding distance if splitting
                while len(survivors) + len(front) > n_survive:
                    k = hv.hvc.argmin()
                    hv.delete(k)
                    front = np.delete(front, k)

                # 

            # extend the survivors by all or selected individuals
            survivors.extend(front)

        return Population.create(*survivors)

        
        # return super()._do(problem, pop, *args, n_survive=n_survive, ideal=ideal, nadir=nadir, **kwargs)

# The Problem 
problem = get_problem("zdt1")
# algorithm = HySMSEMOA()
algorithm = SMSEMOA(survival=MyLeastHypervolumeContributionSurvival()) # key argument for this project DO NOT FORGET!!! 

# Run the Optimization
res = minimize(problem,
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
