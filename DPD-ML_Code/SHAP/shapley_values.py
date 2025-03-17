#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import nshap
import shap
import time
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('run', 'my_definitions.ipynb')
from my_definitions import *
# Loading the file paths of gpr models of different qunatites
## CMC
cmc_gpr_model_path = "gp3_model.pkl"
## Aggreation number 
agg_gpr_model_path = "gp2_model.pkl"
## Micelle radius
rg_gpr_model_path = "gp1_model.pkl"

import joblib
# loading gpr models for differen quantities
## cmc
cmc_model = joblib.load(cmc_gpr_model_path)
## Aggregation number (AN)
agg_model = joblib.load(agg_gpr_model_path)
## Micelle radius 
rg_model = joblib.load(rg_gpr_model_path)


# In[8]:


# Defining the optimal point
optimal_point_scaled = np.array([0.        , 0.35835714])
# Evaluating the Shapley values
cmc_shapley = shapley(cmc_model,[20,20])
# Shifting the Shapley values
## Evalauting the new base value
# b_cmc = cmc_shapley.evaluate_gpr_at(optimal_point_scaled)
b_cmc = 0.025
## Evalauting the mean value
y0_cmc = cmc_shapley.shap_object.base_values[0]
## Shifting the Shapley values
cmc_shapley.shap_object.values = cmc_shapley.shap_object.values + (y0_cmc-b_cmc)/2
# Evaluating Shap at the optimal point
cmc_shapley.evaluate_shap_at(optimal_point_scaled)


# In[10]:


# Defining the optimal point
optimal_point_scaled = np.array([0.        , 0.35835714])
# Evaluating the Shapley values
agg_shapley = shapley(agg_model,[20,20])
# Shifting the Shapley values
## Evalauting the new base value
#b_agg = agg_shapley.evaluate_gpr_at(optimal_point_scaled)
b_agg = 10
## Evalauting the mean value
y0_agg = agg_shapley.shap_object.base_values[0]
## Shifting the Shapley values
agg_shapley.shap_object.values = agg_shapley.shap_object.values + (y0_agg-b_agg)/2
# Evaluating Shap at the optimal point
agg_shapley.evaluate_shap_at(optimal_point_scaled)


# In[11]:


# Defining the optimal point
optimal_point_scaled = np.array([0.        , 0.35835714])
# Evaluating the Shapley values
rg_shapley = shapley(rg_model,[20,20])
# Shifting the Shapley values
## Evalauting the new base value
# b_rg = rg_shapley.evaluate_gpr_at(optimal_point_scaled)
conversion_factor = 1.95*10**-9
b_rg = 2.28*10**-9
b_rg = b_rg/conversion_factor
## Evalauting the mean value
y0_rg = rg_shapley.shap_object.base_values[0]
## Shifting the Shapley values
rg_shapley.shap_object.values = rg_shapley.shap_object.values + (y0_rg-b_rg)/2
# Evaluating Shap at the optimal point
rg_shapley.evaluate_shap_at(optimal_point_scaled)


# In[12]:


b_rg


# In[13]:


# Evaluating the shapley values at the optimal point
# Evaluate Shapley values at the optimal point
print("CMC")
print(r"The Shapley values at the optimized a_AW and a_BW are:")
print(cmc_shapley.shap_object_at_point.values)
print("AGG")
print(r"The Shapley values at the optimized a_AW and a_BW are:")
print(agg_shapley.shap_object_at_point.values)
print("RG")
print(r"The Shapley values at the optimized a_AW and a_BW are:")
print(rg_shapley.shap_object_at_point.values)


# In[19]:


import dill
# saving the SHAP objects
with open('cmc_shapley.pkl', 'wb') as f:
    dill.dump(cmc_shapley, f)

with open('agg_shapley.pkl', 'wb') as f:
    dill.dump(agg_shapley, f)

with open('rg_shapley.pkl','wb') as f:
    dill.dump(rg_shapley,f)


# In[15]:


show_plots(cmc_shapley)


# In[17]:


show_plots(agg_shapley)


# In[18]:


show_plots(rg_shapley)

