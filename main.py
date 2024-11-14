# my files - hybridsms.py and visulisation.py
from hybridsms import MyLeastHypervolumeContributionSurvival
from hybridsms import Hy_SMSEMOA
from hybridsms import hy_minimize
from visulisation import vis   

# deepcopy 
import copy

# pymoo 
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

# sklearn 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# decimal rounding
rounding = 4

# list of where default sms-emoa converges for each problem (max value)
convergance_dict = {
    "zdt1": [40, 70, 140],
    "zdt3": [40, 70, 120],
    "zdt4": [130, 190, 670],
    "zdt6": [130, 230, 600]
}

# comparison values for hypervolume for each problem (similarity)
smsdefualt_hypervolume_dict = {
    "zdt1": [0.8706010094731254, 0.8701004215097737, 0.8667542151989153],
    "zdt3": [1.326932799989349, 1.3251498767866068, 1.3132467347404106],
    "zdt4": [0.8711174340219447, 0.8630376835162556, 0.8508551733759164],
    "zdt6": [0.4959617673719027, 0.4903102556919033, 0.4898010767283024]
}

D = [5, 10, 30]                                                 # number of decision variables (hav to test for each value)
# M = 2                                                           # number of objectives (fixed due to the benchmark problem)
problems = ["zdt1", "zdt3", "zdt4", "zdt6"]                     # benchmark problems to test
ref_points = [[1.1, 1.1], [1.1, 1.1], [1.1, 1.1], [1.1, 1.1]]   # reference points for hypervolume calculation (todo find the best ref point)

# generate model dataset
models = []
models.append([RandomForestRegressor(), "random forest"])
models.append([GaussianProcessRegressor(), "gaussian regression"])
models.append([linear_model.LinearRegression(), "linear regression"])
models.append([SVR(kernel='poly'), "svr poly"])
models.append([SVR(kernel='rbf'), "svr rbf"])
models.append([SVR(kernel='sigmoid'), "svr sigmoid"])
models.append([SVR(kernel='linear'), "svr linear"])
models.append([linear_model.BayesianRidge(), "bayesian ridge"])
models.append([linear_model.SGDRegressor(), "sgd regressor"])
# linear_model.SGDOneClassSVM()

for model in models:
    m = copy.deepcopy(model[0])
    row = None

    for problem in problems:
        # the problem itself in order
        for i in range(len(D)):
            # the number of decision variables
            p = get_problem(problem, n_var=D[i])

            # The Algorithm
            algorithm = Hy_SMSEMOA(survival=MyLeastHypervolumeContributionSurvival(model=m, num_between_model_swaps=10))

            n_gen = convergance_dict[problem][D.index(D[i])]    

            # Run the Optimization
            res = hy_minimize(p,
                            algorithm=algorithm,
                            seed=1,
                            termination=('n_gen', n_gen),
                            save_history=True,
                            verbose=False)

            # calculate hypervolume
            ref_point = ref_points[i]
            ind = HV(ref_point)
            hv_value = ind(res.F)

            # row = f"{model[1]}, {problem},  {D[i]} : {round(hv_value, rounding)} with similarity {round(hv_value/smsdefualt_hypervolume_dict[problem][D.index(D[i])], rounding)} at {n_gen} generations, {res.total_evals} evaluations performed out of {res.total_evals_potential} potential evaluation calls"
            
            # latex table row:
            row = ""
            if D[i] == 5 and problem == "zdt1":
                row = f"{model[1]} & {problem.upper} & {D[i]} & {n_gen} & {round(hv_value, rounding)} & {IGD(p.pareto_front())(res.F)}& {res.total_evals} & {res.total_evals_potential} & {round(hv_value/smsdefualt_hypervolume_dict[problem][D.index(D[i])], rounding)} \\\\"
            elif D[i] == 5 and (problem == "zdt3" or problem == "zdt4" or problem == "zdt6"):
                row = f" & {problem} & {D[i]} & {n_gen} & {round(hv_value, rounding)} & {IGD(p.pareto_front())(res.F)}& {res.total_evals} & {res.total_evals_potential} & {round(hv_value/smsdefualt_hypervolume_dict[problem][D.index(D[i])], rounding)} \\\\"
            else: 
                row = f" & & {D[i]} & {n_gen} & {round(hv_value, rounding)} & {IGD(p.pareto_front())(res.F)}& {res.total_evals} & {res.total_evals_potential} & {round(hv_value/smsdefualt_hypervolume_dict[problem][D.index(D[i])], rounding)} \\\\"

            print(row)

            # # # Result Visualization
            # plot = vis(res, p.pareto_front())
            # plot.display_front()
            # plot.display_hypervolume_overtime()
            # plot.display_aprox_contrib_overtime()