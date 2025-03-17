import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process.kernels import WhiteKernel
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({"text.usetex":True,"font.family":"Helvetica"})
from matplotlib import rcParams

plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.formatter.limits']=[-3,4]
plt.rcParams['ps.usedistiller']='xpdf'
plt.rcParams['figure.autolayout']=True



# Load the training data 
split1 = 33

data = np.loadtxt('DatasetL64_Rg.txt', delimiter='\t')
num_input_parameters = 1
X_data = data[:,4:6]
y_data = data[:,9]

print(X_data,y_data)

# Pre-processing training data

#Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_data_scaled = scaler.fit_transform(X_data)
print(X_data_scaled ,y_data)

#Shuffle 
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(split1), split1, replace=False)
X_shuffled, y_shuffled = X_data_scaled[training_indices], y_data[training_indices]
print(X_shuffled ,y_shuffled)

# Select training data  
X_train = X_shuffled
y_train = y_shuffled

print("X_train:", X_train)
print("y_train:", y_train)

# Select validation data  
X_test = X_data_scaled[split1:,:]
y_test = y_data[split1:]

print("X_test:", X_test)
print("y_test:", y_test)




# Create Gaussian Process model
Ls=[1,1]
kernel = C(1, (1e-5, 1e-3)) * RBF(Ls, (0.0001, 20000))
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100,random_state=15,normalize_y=True)
gp.max_iter_predict = 1000000000000000
gp.fit(X_train, y_train)

# Evaluate the GPR performance in training points

y_train_pred,sigma_train = gp.predict(X_train, return_std=True)

# Calculate mean squared error, mean absolute error, and R-squared
absolute_error = np.abs(y_train - y_train_pred)
squared_error = np.square(y_train - y_train_pred)
    
mse_training = mean_squared_error(y_train, y_train_pred)  # Mean Squared Error
mae_training = mean_absolute_error(y_train, y_train_pred)  # Mean Absolute Error
r2_training = r2_score(y_train, y_train_pred)  # R-squared (Coefficient of Determination)


with open("training_metrics_with_errors.txt", "w") as file:
    # Write headers
    file.write("y_train, y_train_pred, absolute_error, squared_error\n")
    
    # Write data in columns
    for i in range(len(y_train)):
        file.write(f"{y_train[i]}, {y_train_pred[i]}, {absolute_error[i]}, {squared_error[i]}\n")



    # Write the metrics to the file
    file.write(f"Mean Squared Error (MSE): {mse_training}\n")
    file.write(f"Mean Absolute Error (MAE): {mae_training}\n")
    file.write(f"R-squared (R2): {r2_training}\n")

print("Data saved to training_metrics_with_errors.txt")


# Access the kernel_ attribute
fitted_kernel = gp.kernel_

# Now you can use fitted_kernel for further analysis or plotting
print(fitted_kernel)

print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gp.kernel_} \n"
    f"Log-likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}"
)


# GPR predictions in new data points
x1 = np.linspace(0, 1, 32)
x2 = np.linspace(0, 1, 32)
x1_grid, x2_grid = np.meshgrid(x1, x2)
matrix = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T  # Prediction points (1024 points)
sorted_matrix=np.sort(matrix, axis=0)

# Fix a certain value for all other columns

sorted_matrix[:, 0] = 0.31818182
print(matrix[:,1])

X_pred = sorted_matrix
y_pred, sigma = gp.predict(X_pred, return_std=True)
print(y_pred)

# GPR predictions in validation data

y_test_pred,sigma_test = gp.predict(X_test, return_std=True)

print("Predicted Rg:",y_test_pred)
print("Actual Rg:",y_test)

# Evaluate GPR performance in validation points

mse_validation = mean_squared_error(y_test, y_test_pred)
mae_validation = mean_absolute_error(y_test, y_test_pred)
r2_validation = r2_score(y_test, y_test_pred)

absolute_error = np.abs(y_test - y_test_pred)
squared_error = np.square(y_test - y_test_pred)


with open("validation_metrics_with_errors.txt", "w") as file:
    # Write headers
    file.write("y_test, y_test_pred, absolute_error, squared_error\n")
    
    # Write data in columns
    for i in range(len(y_test)):
        file.write(f"{y_test[i]}, {y_test_pred[i]}, {absolute_error[i]}, {squared_error[i]}\n")


    # Write the metrics to the file
    file.write(f"Mean Squared Error (MSE): {mse_validation}\n")
    file.write(f"Mean Absolute Error (MAE): {mae_validation}\n")
    file.write(f"R-squared (R2): {r2_validation}\n")

print("Data saved to training_metrics_with_errors.txt")





import joblib

# After training the models
joblib.dump(gp, 'gp1_model.pkl')


# Plot results
# 2d plots

print(y_train)
print(X_train[:,0])

# Define a consistent style
plt.style.use('seaborn-whitegrid')

# Define a consistent colormap
cmap = 'spring'

# Set color for ticks and axis labels
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
# Set tick parameters globally
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.major.size'] = 5  # Major tick size in points
plt.rcParams['xtick.minor.size'] = 3  # Minor tick size in points
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 3


# First Plot
plt.figure(1)
plt.figure(figsize=(9, 6))
sc2 = plt.scatter(X_train[:, 1], y_train, color='red', s=50, edgecolor='r', label=r'Observations')
plt.plot(X_pred[:, 1], y_pred, color='black', linewidth=2, label=r'Prediction')
plt.fill_between(X_pred[:, 1],
                 y_pred - 1.9600 * sigma,
                 y_pred + 1.9600 * sigma,
                 alpha=0.5, color='g', label=r'$95\%$ confidence interval')
plt.xlabel(r'$a_{BW}$', fontsize=18, color='black')
plt.ylabel(r'$R_g$', fontsize=18, color='black')
plt.ylim(-10, 20)
#plt.title(f'Prediction with {gp.kernel_}', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.grid(True)
plt.savefig('aBW_Rg.png', dpi=300)
plt.show()

plt.figure(2)
plt.figure(figsize=(9, 6))
sc1 = plt.scatter(X_train[:, 0], y_train, color='red', s=50, edgecolor='r', label=r'Observations')
plt.plot(X_pred[:, 0], y_pred, color='black', linewidth=2, label=r'Prediction')
plt.fill_between(X_pred[:, 0],
                 y_pred - 1.9600 * sigma,
                 y_pred + 1.9600 * sigma,
                 alpha=0.5, color='g', label=r'$95\%$ confidence interval')
plt.xlabel(r'$a_{AW}$', fontsize=18, color='black')
plt.ylabel(r'$R_g$', fontsize=18, color='black')
plt.ylim(-10, 20)
#plt.title(f'Prediction with {gp.kernel_}', fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.grid(True)
plt.savefig('aAW_Rg.png', dpi=300)
plt.show()

plt.figure(3)
# Generate grid data
x1_data = np.linspace(0, 1, 100)
x2_data = np.linspace(0, 1, 100)
x1_data, x2_data = np.meshgrid(x1_data, x2_data)

x1_data_flat = x1_data.flatten()
x2_data_flat = x2_data.flatten()

x_data = np.vstack((x1_data_flat, x2_data_flat)).T

z_pred, sigma = gp.predict(x_data, return_std=True)
z_data = z_pred.reshape(x1_data.shape)

fig3 = plt.figure(figsize=(13, 9))
ax1 = fig3.add_subplot(111, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, c='red', s=100, marker="o", edgecolors='r', linewidths=1.5, label='Training points')
ax1.scatter(X_test[:, 0], X_test[:, 1], y_test,  c='blue', s=100, marker="^", edgecolors='b', linewidths=1.5, label='Validation points')
surf1 = ax1.plot_surface(x1_data, x2_data, z_data, cmap=cmap, edgecolor='none', alpha=0.8)
ax1.set_xlabel("$\mathbf{a_{AW}}$", fontweight="bold", fontsize=12, color='black')
ax1.set_ylabel("$\mathbf{a_{BW}}$", fontweight="bold", fontsize=12, color='black')
ax1.set_zlabel("$\mathbf{Rg}$", fontweight="bold", fontsize=12, color='black')
ax1.view_init(elev=30, azim=220)
cbar1 = fig3.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10)
cbar1.set_label('Predicted Rg', fontsize=18)
cbar1.set_ticks(np.linspace(np.min(z_data), np.max(z_data), 5))  # Adjust ticks as needed
cbar1.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax1.legend(['Training points', 'Validation points'], loc='upper right', fontsize=16)
plt.savefig('GPR_Rg_bivariate_subplot1.png', dpi=300)
plt.show()

plt.figure(4)
fig4 = plt.figure(figsize=(10, 7.5))
ax2 = fig4.add_subplot(111, projection='3d')
ax2.scatter(X_train[:, 0], X_train[:, 1], y_train, c='red', s=100, marker="o", edgecolors='r', linewidths=1.5, label=r'Training points')
ax2.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', s=100, marker="^", edgecolors='b', linewidths=1.5, label=r'Validation points')
surf2 = ax2.plot_surface(x1_data, x2_data, z_data, cmap=cmap, edgecolor='none', alpha=0.4)
ax2.scatter(x_data[:,0],x_data[:,1],z_data ,c="k", marker = "o", s = 0.1)
ax2.set_xlabel(r'$a_{AW}$', fontweight="bold", fontsize=30, color='black',labelpad=20)
ax2.set_ylabel(r'$a_{BW}$', fontweight="bold", fontsize=30, color='black',labelpad=10)
ax2.set_zlabel(r'$R_g$', fontweight="bold", fontsize=30, color='black',labelpad=16)
ax2.tick_params(axis='x', labelsize=24)
ax2.tick_params(axis='y', labelsize=24)
ax2.tick_params(axis='z', labelsize=24)
ax2.view_init(elev=30, azim=220)
cbar2 = fig4.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20)
cbar2.ax.tick_params(labelsize=24)
cbar2.set_label(r'Predicted Rg', fontsize=30)
cbar2.set_ticks(np.linspace(np.min(z_data), np.max(z_data), 8))  
cbar2.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax2.legend([r'Training points', r'Validation points'], loc='upper right', fontsize=28)
fig4.subplots_adjust(left=0.10, right=0.95, top=0.99, bottom=0.)
plt.savefig('GPR_Rg_bivariate_subplot2.png', dpi=300)
plt.show()

plt.figure(5)
fig5=plt.figure(figsize=(9, 6))
plt.scatter(y_test, y_test_pred, color='blue', s=50, marker="^", edgecolors='b', label='Predicted')
plt.plot(y_test, y_test, color='red', label='Ideal line')
#plt.title(r'$R_g$ actual values')
plt.ylabel(r'$R_g$ predicted values', color='black', fontsize=30)
plt.xlabel(r'$R_g$ actual values', color='black',fontsize=30)
plt.text(0.1, 0.6, f'R-squared: {r2_validation:.2f}', transform=plt.gca().transAxes, color='green', fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tick_params(axis='both', which='minor', labelsize=24)
plt.legend(fontsize=26)
plt.grid(True)
plt.savefig('Validation_performance.png', dpi=300)
plt.show()




with open("Prediction.txt", "w") as file:
    file.write("X_train[:,1] y_train X_pred[:,1] y_pred\n")
    for x_train, y_train, x_pred, y_pred in zip(X_train[:, 1], y_train, X_pred[:, 1], y_pred):
        file.write(f"{x_train} {y_train} {x_pred} {y_pred}\n")



