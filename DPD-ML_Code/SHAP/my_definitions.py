#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import time
import nshap
import shap
plt.rcParams.update({"text.usetex":True,"font.family":"Helvetica"})
from matplotlib import rcParams

plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.formatter.limits']=[-3,4]
plt.rcParams['ps.usedistiller']='xpdf'
plt.rcParams['figure.autolayout']=True
# Set color for ticks and axis labels
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
# Define a consistent style
# plt.style.use('seaborn-whitegrid')


# Define a consistent colormap
cmap = 'spring'

# Set color for ticks and axis labels
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
# Set tick parameters globally
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.major.size'] = 12  # Major tick size in points
plt.rcParams['xtick.minor.size'] = 12  # Minor tick size in points
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['ytick.minor.size'] = 12

# Defining a shapley class
## input: gpr model, the number of nodes in the aaw,abw directions respectively.
## methods: generates mesh, evaluates the shapley values at the mesh points.
## useful attributes: shapobject for plotting, attribution mesh
class shapley:
    def __init__(self,gpr_model,n_nodes):
        self.gpr_model = gpr_model
        self.n_nodes_aaw, self.n_nodes_abw = n_nodes # Defining the number of nodes in aaw, abw
        self.aaw_mesh,self.abw_mesh,self.X_mesh_column, self.Y_mesh = self.generate_mesh()
        (self.shap_object,
         self.aaw_attribution_mesh,
         self.abw_attribution_mesh,
         self.shap_explainer,
         self.time) = self.evaluate_shap()
        
    def generate_mesh(self):
        # dividing the aaw axis into n_nodes_aaw and abw axis into n_nodes_abw
        aaw_nodes,abw_nodes = [np.linspace(0,1,self.n_nodes_aaw), np.linspace(0,1,self.n_nodes_abw)] 
        # Generating a mesh in the domain
        aaw_mesh,abw_mesh = np.meshgrid(aaw_nodes,abw_nodes)
        X_mesh_column = np.hstack([aaw_mesh.reshape(-1,1), abw_mesh.reshape(-1,1)])
        # generating mesh in the range
        Y_mesh = self.gpr_model.predict(X_mesh_column,return_std= False).reshape(np.shape(aaw_mesh))
        return aaw_mesh,abw_mesh,X_mesh_column, Y_mesh
    
    def evaluate_shap(self):
        # redefining the gpr model suitable for shap
        def func(x):
            value = self.gpr_model.predict(x,return_std=False)
            return value
        # Evaluate start time
        start_time = time.time()
        # loading the explainer
        explainer = shap.KernelExplainer(func, self.X_mesh_column,feature_names=[r"$a_{AW}$",r"$a_{BW}$"])
        # using the explainer at diffent points
        shap_object = explainer(self.X_mesh_column)
        # Evaluate end time
        end_time = time.time()
        aaw_attribution_mesh = shap_object.values[:,0].reshape(self.aaw_mesh.shape)
        abw_attribution_mesh = shap_object.values[:,1].reshape(self.abw_mesh.shape)
        return shap_object,aaw_attribution_mesh,abw_attribution_mesh,explainer,end_time-start_time
    
    def evaluate_shap_at(self,point):
        shap_object_at_point = self.shap_explainer(point)
        self.shap_object_at_point = shap_object_at_point
        
    def evaluate_gpr_at(self,point):
        gpr_at_point = self.gpr_model.predict([point],return_std=False)
        return gpr_at_point
    
# Visualizing the results for the cmc model
def show_plots(quantity):
    fig_bs = plt.figure()  
    shap.plots.beeswarm(quantity.shap_object, show=False, color_bar=False)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(0.05, 0.95, num=5))  # 11 ticks between 0 and 1
    # Get the current tick labels and modify the last one
    tick_labels = ["0.00","0.25","0.50","0.75","1.00"]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Feature Value', rotation=90, labelpad=15, fontsize=12)
    fig_bs.savefig("1_beeswarm_plot.pdf", format='pdf', bbox_inches='tight')

    
    # Create the bar plot without immediately displaying it
    fig_bar_all = plt.figure() 
    shap.plots.bar(quantity.shap_object)
    fig_bar_all.savefig("2_bar_plot_overall.pdf", format='pdf',bbox_inches='tight')
    
    # Shapley values at the optimal point
    fig_bar_opti_point = plt.gcf() 
    shap.plots.bar(quantity.shap_object_at_point)
    fig_bar_opti_point.savefig("3_fig_bar_opti_point.pdf", format='pdf',bbox_inches='tight')
    
    # Creating z axis limits
    min_lim = min(np.min(quantity.aaw_attribution_mesh),np.min(quantity.abw_attribution_mesh))
    max_lim = max(np.max(quantity.aaw_attribution_mesh),np.max(quantity.abw_attribution_mesh))
    
    
     # create a 2d plot attribution of aaw vs aaw
    fig_1 = plt.figure(figsize=[8,6])
    ax_1 = fig_1.add_subplot(111)
    #ax_1.set_title(r"Attribution of $a_{AW}$ wrt $a_{AW}$",fontweight="bold",fontsize=18)
    ax_1.set_xlabel(r"$a_{AW}$",fontweight="bold", fontsize=18, color='black')
    ax_1.set_ylabel(r"Attribution of $a_{AW}$",fontweight="bold", fontsize=18, color='black')
    ax_1.grid(True)
    sc_1 = ax_1.scatter(quantity.aaw_mesh.ravel(),
               quantity.aaw_attribution_mesh.ravel(),
               c =quantity.aaw_attribution_mesh.ravel(),
               cmap="viridis", s= 10)
    cbar_1 = fig_1.colorbar(sc_1, ax=ax_1)
    cbar_1.set_label(r"Attribution of $a_{AW}$", fontsize=18, fontweight="bold")  # Label for colorbar
    fig_1.savefig("4_fig_att_aaw_aaw.pdf", format='pdf',bbox_inches='tight')
    
    # create a 2d plot attribution of aaw vs abw
    fig_2 = plt.figure(figsize=[8,6])
    ax_2 = fig_2.add_subplot(111)
    #ax_2.set_title(r"Attribution of $a_{AW}$ wrt $a_{BW}$",fontweight="bold",fontsize=18)
    ax_2.set_xlabel(r"$a_{BW}$",fontweight="bold", fontsize=18, color='black')
    ax_2.set_ylabel(r"Attribution of $a_{AW}$",fontweight="bold", fontsize=18, color='black')
    ax_2.grid(True)
    sc_2 = ax_2.scatter(quantity.abw_mesh.ravel(),
               quantity.aaw_attribution_mesh.ravel(),
               c =quantity.aaw_attribution_mesh.ravel(),
               cmap="viridis", s= 10)
    cbar_2 = fig_2.colorbar(sc_2, ax=ax_2)
    cbar_2.set_label(r"Attribution of $a_{AW}$", fontsize=18, fontweight="bold")  # Label for colorbar
    fig_2.savefig("5_fig_att_aaw_abw.pdf", format='pdf',bbox_inches='tight')
    
    # create a 2d plot attribution of abw vs aaw
    fig_3 = plt.figure(figsize=[8,6])
    ax_3 = fig_3.add_subplot(111)
    #ax_3.set_title(r"Attribution of $a_{BW}$ wrt $a_{AW}$",fontweight="bold",fontsize=18)
    ax_3.set_xlabel(r"$a_{AW}$",fontweight="bold", fontsize=18, color='black')
    ax_3.set_ylabel(r"Attribution of $a_{BW}$",fontweight="bold", fontsize=18, color='black')
    sc_3 = ax_3.scatter(quantity.aaw_mesh.ravel(),
               quantity.abw_attribution_mesh.ravel(),
               c =quantity.abw_attribution_mesh.ravel(),
               cmap="viridis", s= 15)
    cbar_3 = fig_3.colorbar(sc_3, ax=ax_3)
    cbar_3.set_label(r"Attribution of $a_{BW}$", fontsize=18, fontweight="bold")  # Label for colorbar
    fig_3.savefig("6_fig_att_abw_aaw.pdf", format='pdf',bbox_inches='tight')
    
   
     # create a 2d plot attribution of abw vs abw
    fig_4 = plt.figure(figsize=[8,6])
    ax_4 = fig_4.add_subplot(111)
    #ax_4.set_title(r"Attribution of $a_{BW}$ wrt $a_{BW}$",fontweight="bold",fontsize=18)
    ax_4.set_xlabel(r"$a_{BW}$",fontweight="bold", fontsize=18, color='black')
    ax_4.set_ylabel(r"Attribution of $a_{BW}$",fontweight="bold", fontsize=18, color='black')
    sc_4 = ax_4.scatter(quantity.abw_mesh.ravel(),
               quantity.abw_attribution_mesh.ravel(),
               c =quantity.abw_attribution_mesh.ravel(),
               cmap="viridis", s= 10)
    cbar_4 = fig_4.colorbar(sc_4, ax=ax_4)
    cbar_4.set_label(r"Attribution of $a_{BW}$", fontsize=18, fontweight="bold")  # Label for colorbar
    fig_4.savefig("7_fig_att_abw_abw.pdf", format='pdf',bbox_inches='tight')
    
    
    # create a 3d plot
    fig_1 = plt.figure(figsize=[10,7.5])
    ax_1 = fig_1.add_subplot(111,projection = "3d")
    #ax_1.set_title(r"Partial dependence plot for $a_{AW}$",fontweight="bold",fontsize=18)
    ax_1.set_xlabel(r"$a_{AW}$",fontweight="bold", fontsize=18, color='black')
    ax_1.set_ylabel(r"$a_{BW}$",fontweight="bold", fontsize=18, color='black')
    ax_1.set_zlabel(r"Attribution of $a_{AW}$", fontweight="bold", fontsize=18 ,color='black')
    sc = ax_1.scatter(quantity.aaw_mesh,
               quantity.abw_mesh,
               quantity.aaw_attribution_mesh,
               c =quantity.aaw_attribution_mesh,
               cmap="viridis", s= 20)
    ax_1.view_init(elev=30, azim=220)
    ax_1.set_zlim(min_lim,max_lim)
    cbar1 = fig_1.colorbar(sc, ax=ax_1, shrink=0.6, aspect=10)
    cbar1.set_label(r'Attribution value', fontsize=18)
    cbar1.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax_1.set_box_aspect(None,zoom=0.90)
    fig_1.savefig("8_Partial dependence plot for aaw.pdf", bbox_inches='tight')


    # create a 3d plot
    fig_2 = plt.figure(figsize=[10,7.5])
    ax_2 = fig_2.add_subplot(111,projection = "3d")
    #ax_2.set_title(r"Partial dependence plot for $a_{BW}$",fontweight="bold",fontsize=18)
    ax_2.set_xlabel(r"$a_{AW}$",fontweight="bold", fontsize=18, color='black')
    ax_2.set_ylabel(r"$a_{BW}$",fontweight="bold", fontsize=18, color='black')
    ax_2.set_zlabel(r"Attribution of $a_{BW}$",fontweight="bold", fontsize=18, color='black')
    sc = ax_2.scatter(quantity.aaw_mesh,
               quantity.abw_mesh,
               quantity.abw_attribution_mesh,
               c =quantity.abw_attribution_mesh,
               cmap="viridis", s= 20)
    ax_2.set_zlim(min_lim,max_lim)
    ax_2.view_init(elev=30, azim=220)
    ax_2.set_zlim(min_lim,max_lim)
    cbar2 = fig_2.colorbar(sc, ax=ax_2, shrink=0.6, aspect=10)
    cbar2.set_label(r'Attribution value', fontsize=18)
    cbar2.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    ax_2.set_box_aspect(None,zoom=0.90)
    fig_2.savefig("9_Partial dependence plot for abw.pdf", bbox_inches='tight')
    

