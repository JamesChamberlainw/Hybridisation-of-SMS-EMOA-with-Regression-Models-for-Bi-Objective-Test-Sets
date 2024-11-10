from pymoo.problems import get_problem
# algorithms 
from pymoo.algorithms.moo.sms import SMSEMOA  # SMS-EMOA
from pymoo.algorithms.moo.nsga2 import NSGA2  # NSGA-II
from pymoo.optimize import minimize
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

from pymoo.visualization.scatter import Scatter

# Define problem and algorithm
problem = get_problem("zdt1")
algorithm = SMSEMOA(100) #get_algorithm("sms-emoa", pop_size=100)

# Initial phase: Run a few generations to gather initial training data
initial_res = minimize(problem, algorithm, termination=('n_gen', 5), verbose=True)
train_X = initial_res.X  # Decision variables of initial population
_copy = initial_res.F
train_y = initial_res.F  # Objective values of initial population

# Train the initial SVR models
svr_models = [GaussianProcessRegressor() for _ in range(train_y.shape[1])]
for i in range(train_y.shape[1]):
    svr_models[i].fit(train_X, train_y[:, i])

# Main loop: iteratively optimize, evaluate, and update
for gen in range(10):  # Run for 10 more generations or until termination condition

    # Generate new candidate solutions
    new_solutions = np.random.rand(50, problem.n_var)
    
    # Predict objectives for new solutions
    predicted_objectives = np.column_stack([svr.predict(new_solutions) for svr in svr_models])

    # Select top candidates based on predicted objectives for exact evaluation
    selected_solutions = new_solutions[:10]  # For example, select top 10 solutions
    selected_objectives = problem.evaluate(selected_solutions)

    # Add new data points (exact evaluations) back into the population
    train_X = np.vstack([train_X, selected_solutions])
    train_y = np.vstack([train_y, selected_objectives])

    # Update population with exact solutions for further optimization
    population = np.vstack([initial_res.X, selected_solutions])  # Include exact solutions in population
    pop_objectives = np.vstack([initial_res.F, selected_objectives])
    
    # Continue running the algorithm using the updated population as the initial point
    res = minimize(problem, algorithm, termination=('n_gen', 1), verbose=True, X=population, F=pop_objectives)

    # Retrain SVR models with the expanded dataset
    for i in range(train_y.shape[1]):
        svr_models[i].fit(train_X, train_y[:, i])

    print(f"Generation {gen+1} complete.")

# Visualize the final Pareto front approximation
plot = Scatter(title="Final Pareto Front Approximation")
plot.add(res.F)
plot.add(_copy, color="red")
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.show()
