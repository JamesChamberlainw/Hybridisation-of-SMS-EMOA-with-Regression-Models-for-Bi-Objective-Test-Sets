import numpy as np

# other 
import pymoo as pm
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
# sms 
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA

# random forest
from sklearn.ensemble import RandomForestRegressor


# SMS-EMOA Pymoo Implementation  - this project is extended from this class and files 
# ref. https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/sms.py#L26 


# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================
# ref. https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/sms.py#L14 
# ==============================================================================================================
# ==============================================================================================================

# custom survival function - this is the main class that is extended from the pymoo library

from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival

class MyLeastHypervolumeContributionSurvival(LeastHypervolumeContributionSurvival):

    def __init__(self, eps=10.0) -> None:
        super().__init__(eps=eps)

    # def _do # todo

# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================

class HySMSEMOA(SMSEMOA):

    __model__ = None
    __model_initialized__ = False

    # def next(self):
    #     return super().next()

    def __init__(self):
        super().__init__()
        # super().__init__(survival=MyLeastHypervolumeContributionSurvival())
  
    def suggest_solutions(self, num_suggestions, n_var, xu, xl):
        xu = np.array(xu).reshape(1, n_var)
        xl = np.array(xl).reshape(1, n_var)

        # Generate random values within bounds
        X_new = np.random.rand(num_suggestions, n_var) * (xu - xl) + xl
        return X_new


    def _advance(self, infills=None, **kwargs):
        # messing with infills to understand the data structure
        if infills is not None:
            # further training of the model
            if not self.__model_initialized__:
                # initialize the model
                self.__model_initialized__ = True
                self.__model__ = RandomForestRegressor(n_estimators=100)
                self.__model__.fit(infills.get("X"), infills.get("F"))
            else:
                # update the model
                self.__model__.fit(infills.get("X"), infills.get("F"))


            print(infills.get("X").shape) # (total num of offspring, D (i.e., num of decision variables))

            # predict the scores
            _predicted_scores = self.__model__.predict(infills.get("X"))

            # suggest new solutions
            X_new = self.suggest_solutions(1000, 30, problem.xu, problem.xl)
            y_new = self.__model__.predict(X_new)
            # y_new = self.__model__.predict(X_new)

            # plot actual vs predicted scores
            plot = Scatter()
            plot.add(infills.get("F"), color="blue")
            plot.add(_predicted_scores, color="red")
            plot.add(y_new, color="green")
            # problem (may not be available so default is commented out)
            plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            plot.show()

        return super()._advance(infills, **kwargs) # type: ignore
    
    def _initialize_advance(self, **kwargs):
        return super()._initialize_advance(**kwargs)
    
    def _infill(self):
        return super()._infill()
    
# The Problem 
problem = get_problem("zdt1")
algorithm = HySMSEMOA()
# algorithm = SMSEMOA()

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

# get objective space values
F = res.pop.get("F")
print(F)

# get decision space values
X = res.pop.get("X")
print(X)
# print(res.pop.get("X"))


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

# TODO look into survival function - as this could be useful for improving this algorithm (or atleast attempting to for this assignment)
