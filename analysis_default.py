from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV

# plot the original data 

D = [5, 10, 30]                                                 # number of decision variables (hav to test for each value)
problems = ["zdt1", "zdt3", "zdt4", "zdt6"]                     # benchmark problems to test

# SMS-EMOA
for problem in problems:
    for d in D:
        p = get_problem(problem, n_var=d)
        algorithm = SMSEMOA() 
        res = minimize(p,
                       algorithm,
                       ('n_gen', 100),
                       seed=1,
                       verbose=False,
                       save_history=False)
        
        ind = HV(ref_point=[1.1, 1.1])
        if problem == "zdt4":
            plot = Scatter()
            plot.add(p.pareto_front(), plot_type="line", color="black", alpha=0.7)
            plot.add(res.F, color="red")
            plot.show()
        print(f"{problem}, {d}: {ind(res.F)}")