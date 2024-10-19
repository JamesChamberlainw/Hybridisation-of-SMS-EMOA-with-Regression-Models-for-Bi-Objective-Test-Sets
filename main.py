import pymoo as pm 
import numpy as np
import seaborn as sns

# The benchmark problems 
# my problems - ZDT1, ZDT3, ZDT4 and ZDT6
# ref. https://pymoo.org/problems/multi/zdt.html
from pymoo.problems import get_problem
from pymoo.util.plotting import plot

zdt1 = get_problem("zdt1")
plot(zdt1.pareto_front(), no_fill=True)

zdt3 = get_problem("zdt3")
plot(zdt3.pareto_front(), no_fill=True)

zdt4 = get_problem("zdt4")
plot(zdt4.pareto_front(), no_fill=True)

zdt6 = get_problem("zdt6")
plot(zdt6.pareto_front(), no_fill=True)