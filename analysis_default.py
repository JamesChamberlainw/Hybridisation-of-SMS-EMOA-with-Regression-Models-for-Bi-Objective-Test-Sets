from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV

# plot the original data 

D = [5, 10, 30]                                                 # number of decision variables (hav to test for each value)
problems = ["zdt1", "zdt3", "zdt4", "zdt6"]                     # benchmark problems to test

convergance_dict = {
    "zdt1": [40, 70, 140],
    "zdt3": [40, 70, 120],
    "zdt4": [130, 190, 670],
    "zdt6": [130, 230, 600]
}

# SMS-EMOA
# for problem in problems:
#     for d in D:
#         p = get_problem(problem, n_var=d)
#         n_gen = convergance_dict[problem][D.index(d)]
#         algorithm = SMSEMOA() 
#         res = minimize(p,
#                        algorithm,
#                        ('n_gen', n_gen),
#                        seed=1,
#                        verbose=False,
#                        save_history=False)
        
#         ind = HV(ref_point=[1.1, 1.1])
#         # if problem == "zdt4" or problem == "zdt6":
#         #     plot = Scatter()
#         #     plot.add(p.pareto_front(), plot_type="line", color="black", alpha=0.7)
#         #     plot.add(res.F, color="red")
#         #     plot.show()
#         print(f"{problem}, {d}: {ind(res.F)},  at {n_gen} generations")

# Checking Convergance Manually
p = get_problem("zdt3", n_var=10)
algorithm = SMSEMOA() 
res = minimize(p,
               algorithm,
               ('n_gen', 230),
               seed=1,
               verbose=False,
               save_history=False)

plot = Scatter()
plot.add(p.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.title("ZDT6 SMS-EMOA 10 Decision Variables 230 generations")
plot.add(res.F, color="red")
plot.show()