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

## Technical information

### shapley_values.py

**Input**:
- The script expects the paths to the `.pkl` files containing the GPR models for CMC, As, and Rg, which will be provided by the user.
  
**Output**:
- The output consists of three `.pkl` files: `cmc_shapley.pkl`, `agg_shapley.pkl`, and `rg_shapley.pkl`. These files contain the computed Shapley values, which are required for subsequent optimization.

**Key Functionality**:
- This Python script calculates the Shapley values for each of the provided GPR models. These values reflect the contribution of each input feature to the predictions made by the models. The computation of Shapley values is done using functions defined in `definitions.py`, helping to keep the code modular and clean.

---

### shap_optimization.py

**Input**:
- This script takes the Shapley value files generated by `shapley_values.py` (i.e., `cmc_shapley.pkl`, `agg_shapley.pkl`, and `rg_shapley.pkl`).

**Key Functionality**:
- The main purpose of this Python script is to perform grid confinement optimization. It uses the provided Shapley values to guide the optimization process, helping to adjust model parameters based on the significance of each feature as indicated by the Shapley values.


