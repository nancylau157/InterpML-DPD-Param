# shapley_values.ipynb file:
## INPUT: GPR models from the CMC, As, Rg GPR models.
Please input the relevant paths of the pkl files from the GPR models.
## OUTPUT: Shap objects which will be required for optimization.
cmc_shapley.pkl, agg_shapley.pkl, rg_shapley.pkl.

The required functions for shapley_values.py file are available in the definitions.py
This is done to make the code cleaner.

shap_optimization.ipynb file:
## INPUT: Shap objects from shapley_values.py
cmc_shapley.pkl, agg_shapley.pkl, rg_shapley.pkl
## Performs grid confinement.
