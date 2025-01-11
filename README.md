# COMM510:  Hybridisation of SMS-EMOA with Regression Models for bi-objective test sets

The use of hypervolume as an indicator for the survival of individuals in a population in SMS-EMOA can be considered computationally expensive. This study as such explores the use of a variety of regression models to replace approximately half of the calculations for the hypervolume contribution made by the EA for the *ZDT1*, *ZDT3*, *ZDT4* and *ZDT6* test sets.

# This project? 

This was a coursework project for COMM510 2024-25. Here is my implementation where we explore the use of various regression models to replace the selection function, aiming to reduce real hypervolume calculations by approximately half. The focus is on minimizing calculation calls rather than computational time, and the implementation reflects this emphasis. Additionally, as the computations are derived from test sets and are only two-dimensional, the results are not expected to scale to higher dimensions due to the test sets being all bi-objective.

# References 

This project utilized the pymoo library:
J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access, vol. 8, pp. 89497-89509, 2020, doi: 10.1109/ACCESS.2020.2990567

