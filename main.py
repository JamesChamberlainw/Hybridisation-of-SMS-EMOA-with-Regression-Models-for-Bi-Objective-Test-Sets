import numpy as np
import pymoo as pm

from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.visualization.scatter import Scatter

# SMS-EMOA Pymoo Implementation  - this project is extended from this class and files 
# ref. https://github.com/anyoptimization/pymoo/blob/main/pymoo/algorithms/moo/sms.py#L26 


# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================


from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival

class MyLeastHypervolumeContributionSurvival(LeastHypervolumeContributionSurvival):
    def __init__(self, eps=10.0) -> None:
        super().__init__(eps=eps)

    def _do(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):
        return super()._do(problem, pop, *args, n_survive=n_survive, ideal=ideal, nadir=nadir, **kwargs)

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
