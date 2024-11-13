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
from pymoo.indicators.hv import HV

# sms for comparison
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA

# sklearn 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor

from sklearn.neural_network import MLPClassifier
import sklearn.svm 


D = [5, 10, 30]                                                 # number of decision variables (hav to test for each value)
# M = 2                                                           # number of objectives (fixed due to the benchmark problem)
problems = ["zdt1", "zdt3", "zdt4", "zdt6"]                     # benchmark problems to test
ref_points = [[1.1, 1.1], [1.1, 1.1], [1.1, 1.1], [1.1, 1.1]]   # reference points for hypervolume calculation (todo find the best ref point)

# generate model dataset
models = []
models.append([SVR(kernel='poly'), "svr_poly"])
# models.append([GaussianProcessRegressor(), "gaussian_process"])
# models.append([RandomForestRegressor(), "random_forest"])
# models.append([linear_model.LinearRegression(), "linear_regression"])
# model.append([SVR(kernel='rbf'), "svr_rbf"])
# model.append([SVR(kernel='sigmoid'), "svr_sigmoid"])
# model.append([linear_model.BayesianRidge(), "bayesian_ridge"])
# model.append([linear_model.SGDRegressor(), "sgd_regressor"])

# nameing convention for the models MODELNAME_PROBLEMNAME_NVAR_HYPERVOLUME_+/-_CONTRIBUTION

for model in models:
    m = model[0]
    row = None

    for problem in problems:
        # the problem itself in order
        for i in range(len(D)):
            # the number of decision variables
            p = get_problem(problem, n_var=D[i])

            # The Algorithm
            algorithm = Hy_SMSEMOA(survival=MyLeastHypervolumeContributionSurvival(model=m, num_between_model_swaps=10))

            # Run the Optimization
            res = hy_minimize(p,
                            algorithm=algorithm,
                            seed=1,
                            termination=('n_gen', 140),
                            save_history=True,
                            verbose=False)

            # calculate hypervolume
            ref_point = ref_points[i]
            ind = HV(ref_point)
            hv_value = ind(res.F)

            row = f"{model[1]},  {problem},  {D[i]} : {hv_value}"
            print(row)

            # # Result Visualization
            # plot = Scatter()
            # plot.add(p.pareto_front(), plot_type="line", color="black", alpha=0.7)
            # plot.add(res.F, color="red")
            # plot.show()