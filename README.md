# COMM510:  Hybridisation of SMS-EMOA with Regression Models for bi-objective test sets

This coursework project for COMM510 (2024-25) extends the pymoo library by replacing the selection function with various regression models, aiming to reduce real hypervolume calculations by approximately half. The use of hypervolume as an indicator for the survival of individuals in SMS-EMOA is computationally expensive, and this study explores how regression models can reduce these calculations. The focus is on minimizing calculation calls rather than computational time, with the implementation reflecting this approach. The results are based on bi-objective test sets (ZDT1, ZDT3, ZDT4, and ZDT6) and are not expected to scale to higher dimensions.

# References 

This project utilized the **pymoo** library:
J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access, vol. 8, pp. 89497-89509, 2020, doi: 10.1109/ACCESS.2020.2990567

