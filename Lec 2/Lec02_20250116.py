import numpy as np
import pandas as pd
 
# Create a dataframe using pandas from the dataset
df = pd.read_csv('Lec 2/real_estate_dataset.csv')
 
# Get the number of n_sample and n_features from the csv
n_sample, n_n_features = df.shape
 
# Get the columns of the dataframe and store it in a text file
df_columns = df.columns
np.savetxt('Lec 2/columns.txt', df_columns, fmt='%s')
 
# From the dataset use square_feet, Garage_size, Location_Score, Distance_to_center as feature for the model
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']]
 
# Use price as the target variable
y = df['Price']
 
print(f"Shape of X: {X.shape}\n")
print(f"Data Type of X: {X.dtypes}\n")
 
n_sample,n_features = X.shape
 
# Build a linear model to predict price from the four n_features in X
# make an array of coefs of the size of n_features + 1, initialize to 1.
coefs = np.ones(n_features+1)
 
# Predict the price for each sample in X
predictions_initial = X @ coefs[1:] + coefs[0]
 
#Append a column of 1s to X
X = np.hstack((np.ones((n_sample,1)),X))

print(f"Shape of X after appending 1s column: {X.shape}\n")
 
# Predict the price for each sample in X
predictions = X @ coefs
 
# Check if predictions and predictions_initial are the same
print(f"Predictions and predictions_initial are the same: {np.allclose(predictions, predictions_initial)}")

errors = y - predictions
#find relative error
rel_errors = errors / y

#Find mean squared error from error vector
squared_loss = errors.T @ errors
mean_squared_loss = squared_loss / n_sample

#printthe shape and norm of the error
print(f"Shape of errors: {errors.shape}")
print(f"Norm of errors: {np.linalg.norm(errors)}")
print(f"Norm of relative errors: {np.linalg.norm(rel_errors)}")
 
# What is the optimization problem that we are solving here?
# We are trying to minimize the mean squared loss by finding the optimal values of the coefficients
# These type of problems are called least square problems
 
 
# Aside
# IN heat transfer Nu = f(Re, Pr) : Nu = \alpha Re^m Pr^n, we want to find the values of m and n that minimize the error
# between the predicted Nu and the actual Nu. This is a least square problem
# Since it does not satisfy the linear regression criteria we use log on both size and then fit a equation on the data.
 
# Objective function: f(coefs) = 1/n_sample + \sum_{i=1}^{n_sample} (y_i - coefs^T x_i)^2
 
# What is a solution?
# The solution is the values of the coefficients that minimize the objective function
# In this case, the solution is the values of the coefficients that minimize the mean squared loss

# How do I find the solution?
# the solution have to satisfy the first order condition of the objective function i.e the gradient of the objective function must be zero at the solution
 
# get the loss matrix for the given data
loss_matrix = (y - X @ coefs).T @ (y - X @ coefs)/n_sample
 
# Calculate the gradient of the loss function
gradient = -2 * X.T @ (y - X @ coefs) / n_sample
 
# Setting the gradient to zero we get the follwoing equation
# X^T @ (X @ coefs - y) = 0
# X^T @ X @ coefs = X^T @ y
 
coefs = np.linalg.inv(X.T @ X) @ X.T @ y
 
# Save the coefficients to a text file
np.savetxt('Lec 2/coefficients.txt', coefs, fmt='%s')
 
# Predict the price for each sample in X
optimal_solution_predictions = X @ coefs
 
# error model
optimal_solution_errors = y - optimal_solution_predictions
 
# Find relative error
optimal_solution_rel_errors = optimal_solution_errors / y
 
# print the norms of the errors
print(f"Norm of optimal solution errors: {np.linalg.norm(optimal_solution_errors)}")
print(f"Norm of optimal solution relative errors: {np.linalg.norm(optimal_solution_rel_errors)}")
 
 
# Use all the n_features in the dataset to build a linear model to predict price
y = df['Price'].values
X = df.drop('Price', axis=1).values
n_sample, n_features = X.shape

print(f"Shape of X: {X.shape}\n")

# Append a column of 1s to X
X = np.hstack((np.ones((n_sample,1)),X))
print(f"Shape of X after appending 1s column: {X.shape}\n")
 
coefs = np.ones(n_features+1)

print(f"Shape of coefs: {coefs.shape}\n")

# Calculate the coefficients
coefs = np.linalg.inv(X.T @ X) @ X.T @ y
 
# Predict the price for each sample in X
predictions_with_all_n_features = X @ coefs
 
# Calculate the errors
errors_with_all_n_features = y - predictions_with_all_n_features
 
# Find the relative errors
rel_errors_with_all_n_features = errors_with_all_n_features / y
 
# Print the norms of the errors
print(f"Norm of errors with all n_features: {np.linalg.norm(errors_with_all_n_features)}")
print(f"Norm of relative errors with all n_features: {np.linalg.norm(rel_errors_with_all_n_features)}")
 
# Save the coefficients to a text file
np.savetxt('Lec 2/coefficients_all_n_features.txt', coefs, fmt='%s')
 
# Solve the normal equation using Q R decomposition
Q,R = np.linalg.qr(X)
 
print(f"Shape of Q: {Q.shape}")
print(f"Shape of R: {R.shape}")
 
# Write R to a file named R.csv
 
np.savetxt('Lec 2/R.csv', R, delimiter=',')
 
#Calculate Q.T @ Q and save it to a file named Q_TQ.csv
Q_TQ = Q.T @ Q
np.savetxt('Lec 2/Q_TQ.csv', Q_TQ, delimiter=',')
 
# X = QR
# X.T @ X = R.T @ Q.T @ Q @ R = R.T @ R
# X.T @ y = R.T @ Q.T @ y
# R @ coefs = Q.T @ y
 
b = Q.T @ y

# Solve for coefs using back substitution
# Use the loop to solve back substitution problem
coef_QR = np.zeros(n_features+1)
for i in range(n_features,-1,-1):
    coef_QR[i] = b[i]
    for j in range(i+1,n_features+1):
        coef_QR[i] = coef_QR[i] - R[i,j] * coef_QR[j]
    coef_QR[i] = coef_QR[i] / R[i,i]
 
#Save the coefficients to a text file
np.savetxt('Lec 2/coefficients_using_qr_backsub.txt', coef_QR, fmt='%s')

print(f"The coefficients calculated using QR and Normal equation are same: {np.allclose(coefs, coef_QR)}")

# Solve the normal equation using the SVD decomposition
U, S, V_T = np.linalg.svd(X , full_matrices=False)

print(f"Shape of S: {S.shape}")
coef_SVD = V_T.T @ np.linalg.inv(np.diag(S)) @ U.T @ y
np.savetxt('Lec 2/coef_using_svd.txt', coef_SVD, fmt='%s')

print(f"The coefficients calculated using SVD and Normal equation are same: {np.allclose(coefs, coef_SVD)}")

# Calculate the predictions using the coefficients calculated using Eigen decomposition

D, V = np.linalg.eig(X.T @ X)

V_TV = V.T @ V
np.savetxt('Lec 2/V_TV.csv', V_TV, delimiter=',')

# X.T @ X = V @ D @ V.T 
#X.T @ y = X.T @ y
#(X.T @ X)^-1 = V @ D^-1 @ V.T

D_inv = np.diag(1 / D)
coefs_eigen = V @ D_inv @ V.T @ X.T @ y
np.savetxt('Lec 2/coef_using_eigen_decomp.txt', coefs, fmt='%s')

print(f"The coefficients calculated using Eigen decomposition and Normal equation are same: {np.allclose(coefs_eigen, coef_SVD)}")




