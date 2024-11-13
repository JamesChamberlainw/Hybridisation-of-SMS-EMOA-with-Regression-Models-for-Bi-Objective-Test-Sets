# my files - hybridsms.py
from hybridsms import MyLeastHypervolumeContributionSurvival
from hybridsms import Hy_SMSEMOA
from hybridsms import hy_minimize

# plotting and maths
import numpy as np
import matplotlib.pyplot as plt

# pymoo 
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

# sklearn 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor



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
algorithm = Hy_SMSEMOA(survival=MyLeastHypervolumeContributionSurvival(model=model, num_between_model_swaps=10)) 

# Run the Optimization
res = hy_minimize(problem,
               algorithm=algorithm,
               termination=('n_gen', 100),
               seed=1,
               save_history=True,
               verbose=False)

# Run the Optimization
# res = minimize(problem, algorithm=SMSEMOA(), termination=('n_gen', 100), seed=1, save_history=True, verbose=False)

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