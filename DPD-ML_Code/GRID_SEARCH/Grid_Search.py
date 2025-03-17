import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

# Update plot style settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "axes.formatter.limits": [-3, 4],
    'ps.usedistiller': 'xpdf',
    'figure.autolayout': True
})
rcParams['font.family'] = 'sans-serif'

# Load the training data
data = np.loadtxt('DatasetL64.txt', delimiter='\t')
x1_data = data[:, 4].reshape(-1,1)
x2_data = data[:, 5].reshape(-1,1)
y1_data = data[:, 10].reshape(-1,1)
y2_data = data[:, 12].reshape(-1,1)
y3_data = data[:, 13].reshape(-1,1)

# Scalers
scaler_x1 = MinMaxScaler(feature_range=(0, 1))
x1_data_scaled = scaler_x1.fit_transform(x1_data)
print(x1_data)
print(x1_data_scaled)

scaler_x2 = MinMaxScaler(feature_range=(0, 1))
x2_data_scaled = scaler_x2.fit_transform(x2_data)
scaler_y1 = MinMaxScaler(feature_range=(0, 1))
y1_data_scaled = scaler_y1.fit_transform(y1_data)
scaler_y2 = MinMaxScaler(feature_range=(0, 1))
y2_data_scaled = scaler_y2.fit_transform(y2_data)
scaler_y3 = MinMaxScaler(feature_range=(0, 1))
y3_data_scaled = scaler_y3.fit_transform(y3_data)

# Define target values
y1_T = np.array([[1.16]])
y2_T = np.array([[10]])
y3_T = np.array([[0.025]])

# Scale target values
y1_T_scaled = scaler_y1.transform(y1_T)
y2_T_scaled = scaler_y2.transform(y2_T)
y3_T_scaled = scaler_y3.transform(y3_T)



# First scaling for x1_min and x1_max
x1_min = 25.9
x1_min_reshaped = np.array([[x1_min]])
x1_min_scaled = scaler_x1.transform(x1_min_reshaped)
print(x1_min_scaled)
x1_min_scalar = x1_min_scaled[0][0]
print(f"x1_min_scaled: {x1_min_scalar:.6f}")

x1_max = 29
x1_max_reshaped = np.array([[x1_max]])
x1_max_scaled = scaler_x1.transform(x1_max_reshaped)
print(x1_max_scaled)
x1_max_scalar = x1_max_scaled[0][0]
print(f"x1_max_scaled: {x1_max_scalar:.6f}")

# Second scaling for x2_min and x2_max
x2_min = 32
x2_min_reshaped = np.array([[x2_min]])
x2_min_scaled = scaler_x2.transform(x2_min_reshaped)

x2_min_scalar = x2_min_scaled[0][0]
print(f"x2_min_scaled: {x2_min_scalar:.6f}")

x2_max = 36
x2_max_reshaped = np.array([[x2_max]])
x2_max_scaled = scaler_x2.transform(x2_max_reshaped)

x2_max_scalar = x2_max_scaled[0][0]
print(f"x2_max_scaled: {x2_max_scalar:.6f}")

# Load the pre-trained Gaussian Process models
gp1 = joblib.load('gp1_model.pkl')
gp2 = joblib.load('gp2_model.pkl')
gp3 = joblib.load('gp3_model.pkl')

# Define the objective function
def objective_function(y1_T, y1, y2_T, y2, y3_T, y3):
    return ((abs((y1_T - y1)/y1_T)**2)*10**(-3) + (abs(y2_T - y2)/y2_T)**2 + (abs(y3_T - y3)/y3_T)**2)**(1/2)

# Initial coarse grid parameters
#initial_step = 0.2
#initial_step = 0.025
#initial_step = 0.05
initial_step = 0.1
#par1_range = np.linspace(np.min(x1_data_scaled), np.max(x1_data_scaled), 10)
#par2_range = np.linspace(np.min(x2_data_scaled), np.max(x2_data_scaled), 10)

par1_range = np.linspace(x1_min_scalar, x1_max_scalar, 10)
par2_range = np.linspace(x2_min_scalar, x2_max_scalar, 10)


# Ensure no negative values in the parameter space
par1_range = np.maximum(par1_range, 0)  # Set lower bound to 0
par2_range = np.maximum(par2_range, 0)  # Set lower bound to 0

# Adaptive grid search parameters
refinement_tolerance = 1e-3  # Stop when the minimum value change is below this threshold
max_refinements = 10  # Limit the number of refinements
min_value = float('inf')  # Initialize the minimum value
min_coords = (None, None)
y1_min_coords = None
y2_min_coords = None
y3_min_coords = None

# Start the grid search with the initial coarse grid
start_time = time.time()
for refinement in range(max_refinements):
    print(f"Refinement {refinement + 1} with step size {initial_step}")
    
    # Create a meshgrid for the current refinement
    par1_grid, par2_grid = np.meshgrid(par1_range, par2_range)
    values = np.zeros(par1_grid.shape)

    # Perform the grid search over the current grid
    for i, par1 in enumerate(par1_range):
        for j, par2 in enumerate(par2_range):
            X_pred = np.array([[par1, par2]])
            y1_pred, _ = gp1.predict(X_pred, return_std=True)
            y2_pred, _ = gp2.predict(X_pred, return_std=True)
            y3_pred, _ = gp3.predict(X_pred, return_std=True)
            
            # Scale the predictions
            y1_pred_scaled = scaler_y1.transform(y1_pred.reshape(-1, 1))
            y2_pred_scaled = scaler_y2.transform(y2_pred.reshape(-1, 1))
            y3_pred_scaled = scaler_y3.transform(y3_pred.reshape(-1, 1))

            # Calculate the objective function value
            value = objective_function(y1_T_scaled, y1_pred_scaled, y2_T_scaled, y2_pred_scaled, y3_T_scaled, y3_pred_scaled)
            values[j, i] = value

            # Update the minimum value and coordinates
            if value < min_value:
                min_value = value
                min_coords = (par1, par2)
                y1_min_coords = y1_pred_scaled
                y2_min_coords = y2_pred_scaled
                y3_min_coords = y3_pred_scaled

    # Check if we need to stop based on the change in the minimum value
    if refinement > 0 and abs(prev_min_value - min_value) < refinement_tolerance:
        print(f"Converged to minimum value within tolerance. Stopping refinement.")
        break

    prev_min_value = min_value

    # Refinement step: adjust the grid around the minimum found
    par1_range = np.linspace(min_coords[0] - initial_step / 2, min_coords[0] + initial_step / 2, 10)
    par2_range = np.linspace(min_coords[1] - initial_step / 2, min_coords[1] + initial_step / 2, 10)

    # Ensure no negative values in the parameter space after refinement
    par1_range = np.maximum(par1_range, 0)  # Set lower bound to 0
    par2_range = np.maximum(par2_range, 0)  # Set lower bound to 0

    # Reduce the step size for the next refinement
    initial_step /= 2

    # Save the plot for this step
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(par1_grid, par2_grid, values, cmap='viridis', alpha=0.8)

    # Highlight the minimum point
    ax.scatter(min_coords[0], min_coords[1], min_value, color='red', s=100, label='Minimum')

    # Set labels
    ax.set_xlabel(r'$a_{AW}$', fontsize=22, labelpad=20)
    ax.set_ylabel(r'$a_{BW}$', fontsize=22, labelpad=20)
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    ax.tick_params(axis='z', labelsize=26)

    # Execution time details on the plot
    end_time = time.time()
    total_execution_time = end_time - start_time
    average_execution_time = total_execution_time / (len(par1_range) * len(par2_range))

    # Display execution time on the plot
    plt.figtext(0.05, 0.96, f'Step = {initial_step:.3f}', fontsize=22, color='black')
    plt.figtext(0.05, 0.92, f'Total Execution Time: {total_execution_time:.3f} sec', fontsize=22, color='black')

    # Display optimal parameters
    opt_par_1 = scaler_x1.inverse_transform(np.array([[min_coords[0]]]))
    opt_par_2 = scaler_x2.inverse_transform(np.array([[min_coords[1]]]))
    plt.figtext(0.05, 0.84, fr'$a_{{AW}}^{{opt}} = {opt_par_1[0][0]:.3f}$', fontsize=24, color='red')
    plt.figtext(0.05, 0.77, fr'$a_{{BW}}^{{opt}} = {opt_par_2[0][0]:.3f}$', fontsize=24, color='red')

    # Save the figure for the current step
    plt.savefig(f'adaptive_grid_search_step_{refinement + 1}.png')
    plt.close()

# After finding min_coords, get y1 and y2 corresponding to these coordinates
X_min_coords = np.array([min_coords])
y1_min_coords, _ = gp1.predict(X_min_coords, return_std=True)
y2_min_coords, _ = gp2.predict(X_min_coords, return_std=True)
y3_min_coords, _ = gp3.predict(X_min_coords, return_std=True)

# Output the results
print(f"The minimum value is {min_value} at coordinates {min_coords}")
print(f"Corresponding y1 value: {y1_min_coords}")
print(f"Corresponding y2 value: {y2_min_coords}")
print(f"Corresponding y3 value: {y3_min_coords}")

# Output the execution time details
print(f"The minimum value is {min_value} at coordinates {min_coords}")
print(f"Average execution time per prediction: {average_execution_time:.6f} seconds")
print(f"Total execution time for the grid search: {total_execution_time:.6f} seconds")

# Example of finding optimal parameters and inverse transforming
# Inverse transform min_coords back to the original scale
par1 = np.array([[min_coords[0]]])
par2 = np.array([[min_coords[1]]])

# Inverse transform the parameters to the original scale
opt_par_1 = scaler_x1.inverse_transform(par1)
opt_par_2 = scaler_x2.inverse_transform(par2)

# Output the optimal parameters in the original scale
print(f"\nThe optimal parameter par1 in original scale is: {opt_par_1[0][0]}")
print(f"\nThe optimal parameter par2 in original scale is: {opt_par_2[0][0]}")
# Save the results to a text file
with open("output.txt", "w") as f:
    f.write(f"The minimum value is {min_value} at coordinates {min_coords}\n")
    f.write(f"Corresponding y1 value: {y1_min_coords}\n")
    f.write(f"Corresponding y2 value: {y2_min_coords}\n")
    f.write(f"Corresponding y3 value: {y3_min_coords}\n")
    
    # Output the execution time details
    end_time = time.time()
    total_execution_time = end_time - start_time
    average_execution_time = total_execution_time / (len(par1_range) * len(par2_range))
    
    f.write(f"\nThe minimum value is {min_value} at coordinates {min_coords}\n")
    f.write(f"Average execution time per prediction: {average_execution_time:.6f} seconds\n")
    f.write(f"Total execution time for the grid search: {total_execution_time:.6f} seconds\n")
    
    # Example of finding optimal parameters and inverse transforming
    # Inverse transform min_coords back to the original scale
    par1 = np.array([[min_coords[0]]])
    par2 = np.array([[min_coords[1]]])

    # Inverse transform the parameters to the original scale
    opt_par_1 = scaler_x1.inverse_transform(par1)
    opt_par_2 = scaler_x2.inverse_transform(par2)

    # Output the optimal parameters in the original scale
    f.write(f"\nThe optimal parameter par1 in original scale is: {opt_par_1[0][0]}\n")
    f.write(f"The optimal parameter par2 in original scale is: {opt_par_2[0][0]}\n")
