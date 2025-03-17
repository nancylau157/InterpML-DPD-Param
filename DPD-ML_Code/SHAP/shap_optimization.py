#!/usr/bin/env python
# coding: utf-8

# In[50]:


import dill
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from my_definitions import *
# Importing the scaler to unscale aaw and abw
scaler = joblib.load("scaler.pkl")

#get_ipython().run_line_magic('run', 'my_definitions.ipynb')
# loading the shapley values of all the quantities
## CMC
with open('cmc_shapley.pkl', 'rb') as f:
    cmc_shapley = dill.load(f)
## AGG
with open("agg_shapley.pkl",'rb') as f:
    agg_shapley = dill.load(f)
## Rg
with open("rg_shapley.pkl","rb") as f:
    rg_shapley = dill.load(f)
    

def confine_indices(quantity):
    '''
    input: cmc_shapley or agg_shapley or rg_shapley
    output: The indices of the domain where optimal value might occur
    '''
    ## getting the shap values
    shap_values = quantity.shap_object.values
    ## Shap values corresponding to aaw and abw
    aaw_shap_values = shap_values[:,0]
    abw_shap_values = shap_values[:,1]
    ### finding absolute (max SHAP value)
    mod_max_aaw_shap = np.abs(np.max(aaw_shap_values))
    mod_max_abw_shap = np.abs(np.max(abw_shap_values))
    ### finding absolute (min SHAP value)
    mod_min_aaw_shap = np.abs(np.min(aaw_shap_values))
    mod_min_abw_shap = np.abs(np.min(abw_shap_values))

    # Finding necessary indices
    ## comparing the right arm of aaw with the left arm of abw
    if mod_max_aaw_shap >= mod_min_abw_shap:
        necessary_indices_1 = np.where((aaw_shap_values>=0) & (aaw_shap_values<=mod_min_abw_shap))
    else:
        necessary_indices_1 = np.where((abw_shap_values<=0) & (abw_shap_values>=-mod_max_aaw_shap))
    ## comparing the left arm of aaw with the right arm of abw
    if mod_min_aaw_shap >= mod_max_abw_shap:
        necessary_indices_2 = np.where((aaw_shap_values<=0) & (aaw_shap_values>=-mod_max_abw_shap))
    else:
        necessary_indices_2 = np.where((abw_shap_values>=0) & (abw_shap_values<=mod_min_aaw_shap))
    ## combining incides
    necessary_indices = np.hstack([necessary_indices_1[0],necessary_indices_2[0]])
    
    return necessary_indices

# Confining the domain
## Using CMC:
necessary_indices_cmc = confine_indices(cmc_shapley)
## Using agg:
necessary_indices_agg = confine_indices(agg_shapley)
## Using rg:
necessary_indices_rg = confine_indices(rg_shapley)
## common indices
necessary_indices_overall = np.intersect1d(necessary_indices_cmc,
                                            np.intersect1d(necessary_indices_agg,
                                            necessary_indices_rg))

# Showing the plot:
aaw_all = scaler.inverse_transform(cmc_shapley.X_mesh_column)[:,0]
abw_all = scaler.inverse_transform(cmc_shapley.X_mesh_column)[:,1]
plt.scatter(aaw_all,abw_all,c="k",s = 100,label="All grid")
## showing the agg necessary aaw,abw
aaw_agg = aaw_all[necessary_indices_agg]
abw_agg = abw_all[necessary_indices_agg]
plt.scatter(aaw_agg,abw_agg,c="b",s = 60,label="$A_s$")
## showing the cmc necessary aaw,abw
aaw_cmc = aaw_all[necessary_indices_cmc]
abw_cmc = abw_all[necessary_indices_cmc]
plt.scatter(aaw_cmc,abw_cmc,c="r",s = 40,label="$CMC$")

## showing the rg necessary aaw,abw
aaw_rg = aaw_all[necessary_indices_rg]
abw_rg = abw_all[necessary_indices_rg]
plt.scatter(aaw_rg,abw_rg,c="y",s = 10,label="$R_g$")

plt.xlabel("$a_{AW}$",fontsize=15)
plt.ylabel("$a_{BW}$",fontsize=15)
# plt.title("Allowable grid points from Shapley analysis",fontsize=20,fontweight="bold")
# plt.legend(fontsize=12, labelcolor="black", 
#            edgecolor="pink", facecolor="pink",framealpha=1,loc="upper right")
plt.gca().set_aspect(0.3)
plt.savefig("all_grid.pdf")


# In[48]:


plt.scatter(aaw_all,abw_all,c="k",s = 100,label="All grid")

## showing the common necessary aaw,abw
aaw_overall = aaw_all[necessary_indices_overall]
abw_overall = abw_all[necessary_indices_overall]
plt.scatter(aaw_overall,abw_overall,c="g",s = 90,label="confined grid")
plt.xlabel("$a_{AW}$",fontsize=15)
plt.ylabel("$a_{BW}$",fontsize=15)
# plt.title("Allowable grid points from Shapley analysis",fontsize=20,fontweight="bold")
# plt.legend(fontsize=12, labelcolor="black", 
#            edgecolor="pink", facecolor="pink",framealpha=1,loc="upper right")
plt.gca().set_aspect(0.3)
plt.savefig("necessary_grid.pdf")


# In[51]:


# Showing the plot:
plt.scatter(aaw_all,abw_all,c="k",s = 100,label="All grid")

## showing the rg necessary aaw,abw
plt.scatter(aaw_rg,abw_rg,c="y",s = 10,label="$R_g$")
plt.xlabel("$a_{AW}$",fontsize=15)
plt.ylabel("$a_{BW}$",fontsize=15)
# plt.title("Allowable grid points from Shapley analysis og $R_g$",fontsize=20,fontweight="bold")
# plt.legend(fontsize=12, labelcolor="black", 
#            edgecolor="pink", facecolor="pink",framealpha=1,loc="upper right")
plt.gca().set_aspect(0.3)
plt.savefig("rg_grid.pdf")

