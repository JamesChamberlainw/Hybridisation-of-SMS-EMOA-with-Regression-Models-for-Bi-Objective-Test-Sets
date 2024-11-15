from pymoo.visualization.scatter import Scatter
import numpy as np
# matplotlib
import matplotlib.pyplot as plt

class vis:
    __res__ = None
    __pareto_front__ = None

    def __init__(self, ref, pareto_front=None):
        self.__res__ = ref

        if pareto_front is not None:
            self.__pareto_front__ = pareto_front
    
    def display_hypervolume_overtime(self, best_found=None):
        """
            Display a visualisation of the hypervolume over time for the default SMS-EMOA
        
            ref_point will always be [1.1, 1.1] for all problems SO THIS IS FIXED
        """
        x = np.arange(0, len(self.__res__.hv_actual), 1)
        y = self.__res__.hv_actual 

        plt.plot(x, y, label="Hypervolume over time")
        plt.xlabel("Evaluations")
        plt.ylabel("Hypervolume")
        plt.title("Hypervolume over time")
        plt.show()


    def display_aprox_contrib_overtime(self):

        # # plot hv actual and hv aprox
        x = np.arange(0, len(self.__res__.hv_actual_contrib), 1)
        plt.title("Sum of Hypervolume Contribution against Evaluation Number")
        plt.xlabel("Evaluations")
        plt.ylabel("Hypervolume Contribution")
        plt.plot(x, self.__res__.hv_actual_contrib, label="Actual Contribution (training data)")
        plt.plot(x, self.__res__.hv_aprox_contrib, label="Model Contribution (aprox)")
        plt.plot(x, self.__res__.hv_missing_evals_unused, label="Actual Contribution (unused values)")
        plt.plot(x, self.__res__.hv_mean_contrib, label="Mean Contribution (for used values)")
        plt.legend()
        plt.show()

    def display_front(self):
        plot = Scatter()
        plot.add(self.__pareto_front__, plot_type="line", color="black", alpha=0.7)
        plot.add(self.__res__.F, color="red")
        plot.show()



# use:
# plot = vis(res, p.pareto_front())
# plot.display_front()
# plot.display_hypervolume_overtime()
# plot.display_aprox_contrib_overtime()
