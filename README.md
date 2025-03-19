# InterpML-DPD-Param

## Motivations and Application

This repository contains the code for the data-driven approach decribed in the work "Interpretable machine-learning enhanced parametrization methodology for Pluronics-Water Mixtures in DPD simulations" for accurately determining model parameters tailored to perform DPD simulations of different Pluronic systems.

## Workflow Overview

This repository contains the workflow for building a **Gaussian Process Regression (GPR)-based surrogate model**. The goal of this model is to replicate the results of **Dissipative Particle Dynamics (DPD)** simulations, significantly reducing computational costs while maintaining accuracy.  

The workflow follows these steps:  

1. **Data Collection:** The process begins by collecting data from DPD simulations, which serve as the basis for training the surrogate model.  

2. **Model Construction:** At the core of the workflow, there is the building of a **Gaussian Process Regression (GPR)-based surrogate model**. The goal of this model is to replicate the results of **Dissipative Particle Dynamics (DPD)** simulations, significantly reducing computational costs while maintaining accuracy.  

3. **Model Training:** The **GPR** model learns the relationships between input parameters and predicted properties.  

4. **Model Interpretation:** Once the surrogate model is built, **SHAP** (SHapley Additive exPlanations) analysis is performed to enhance model interpretability. This analysis helps identify the most relevant input parameters and understand their causal impact on the predicted properties.  

5. **Optimization:** By combining **GPR** and **SHAP** analysis, the workflow achieves an interpretable machine learning approach that enables efficient parameter optimization, minimizing the need for exhaustive simulations.  

**Technical information**


