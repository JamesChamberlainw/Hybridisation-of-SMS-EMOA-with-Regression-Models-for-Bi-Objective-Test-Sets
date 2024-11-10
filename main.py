import pymoo as pm 
import numpy as np
import seaborn as sns

# The benchmark problems 
# my problems - ZDT1, ZDT3, ZDT4 and ZDT6
# ref. https://pymoo.org/problems/multi/zdt.html
from pymoo.problems import get_problem
from pymoo.util.plotting import plot
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

# algorithms 
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.algorithms.moo.nsga2 import NSGA2  # NSGA-II

# termination criteria
from pymoo.core.termination import Termination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination.robust import RobustTermination

# The following loads in and shows the pareto optimal fronts. 
# zdt1 = get_problem("zdt1")
# plot(zdt1.pareto_front(), no_fill=True)

# zdt3 = get_problem("zdt3")
# plot(zdt3.pareto_front(), no_fill=True)

# zdt4 = get_problem("zdt4")
# plot(zdt4.pareto_front(), no_fill=True)

# zdt6 = get_problem("zdt6")
# plot(zdt6.pareto_front(), no_fill=True)

# ref. https://pymoo.org/algorithms/moo/sms.html
# problem = get_problem("zdt6")

D = [5, 10, 30] # number of decision variables (hav to test for each value)
M = 2           # number of objectives (fixed due to the benchmark problem)

for d in D:
    problem = get_problem("zdt6", n_var=d)
    # algorithm = NSGA2() # NSGA-II
    algorithm = SMSEMOA() # SMS-EMOA
    # termination = get_termination("n_gen", 200) # number of generations
    termination = RobustTermination(DesignSpaceTermination(tol=0.001), period=20) # xtol termination
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True,
                   save_history=True)
    
    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()