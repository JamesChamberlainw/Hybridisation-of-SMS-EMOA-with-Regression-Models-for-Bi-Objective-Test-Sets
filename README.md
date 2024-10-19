# COMM510:  Hybridisation of indicator-based MOEA and Machine Learning Corusework 
The aim of this coursework is to explore how regression or classification techniques can be used to build a model of indicators in indicator-based MOEAs. Select an indicator-based MOEA of your choice and use one of the following indicators when building a machine learning model: Hypervolume, IGD, R2. Test the algorithm on either benchmark set 3 or real-world problems.

This will be completed working on the following benchmark problems: ZDT1, ZDT3, ZDT4, ZDT6 with number of variables, D = [5, 10, 30] and number of objectives,
M = 2. You should test the algorithm on all possible values of D. 

## End of Day 1 findings: initial understanding of the coursework problem (pre-starting the coursework)
*My understanding of what this cousework is* \\ 

In this coursework i will be working with ZDT1, ZDT3, ZDT4 and ZDT6 problems, these are all dual objective problems (only two objectives) where the objectives can be anything, and my simulation itself is a black-box - I don't know yet where I will be getting my solutions from as i don't know what the data itself is so $x_objs$ and $f_1$ and $f_2$ values are all unknown as of so far. 

The pymoo libary can get me the Pareto fronts for this. 

I have to use IBEAs - meaning that my solution has to use performance indicators as a guide in the selection process. 

The ultiamte aim of this coursework is that I use some ML technique falling into either regression or classification categories. 

I don't know how yet classificaiton would fall into this? Maybe to worok as early stop to reduce additional computational cost? 

Regression for aiding in finding the front? idk 

I have to fit this inside of my IBEA algorithm (yet to be decided maybe SMS-EMOA as it was introduced in lecture)

SMS-EMOA works well for ZDT1, ZDT3 and ZDT4 but ZDT6 seems to get as close to the Pareto front as it does on the other problems (seed 1, from the example provided by pymoo)

**AIM:** use classifcaiton or regression inside of the IBEA to reduced unessusary evaluation function runs - either by increasing convergance speed or early stopping? 

# Keywords 
**IBEA** - Indicator-Based Multi-Objective Evolutionary Algorithm
