#
# %% Creating Covariance Matrix
import numpy as np
import matplotlib.pyplot as plt
A = np.random.randint(-10, 10, (20, 5))

# Creating a augmented matrix

AugA = A - np.mean(A, axis=0)

# Covariance Matrix
AtA = (AugA.T @ AugA) / (AugA.shape[0] - 1)

covA = np.cov(A, bias=False, rowvar=False)

np.allclose(covA, AtA)

# %% Calculating standard deviation of sample

# Calculating standard deviation for each rows
std = np.sqrt(np.diag(AtA))

# With function
std1 = np.std(A, axis=0, ddof=1)

np.allclose(std, std1)

# %% Creating correaltion matrix

# Creating an outer product of std matrix
outer_std = np.outer(std, std1)

# Normalizing the covariance matrix by it's std
corrA = AtA / outer_std

CorrA1 = np.corrcoef(A, rowvar=False)

np.allclose(corrA, CorrA1)
\
# %% Calculating PCA using SVD

# First we need to create a tall matrices

matRandom = np.random.uniform(-10,11, (213, 4))

weight1 = np.random.uniform(-5, 5, (4, 1))
weight2 = np.random.uniform(-5, 5, (4, 1))

lincom1 = matRandom @ weight1
lincom2 = matRandom @ weight2

df = np.concatenate((matRandom, lincom1, lincom2), axis=1)

# Calculate the mean for each feature

df_mean = np.mean(df, axis=0)

# Center the df by substracting with its mean

df_centered = df - df_mean

np.allclose(df[0, 0] - df_mean[0], df_centered[0, 0])

# Decompose the centered matrix using SVD

U, S, V = np.linalg.svd(df_centered)

#%%
# 3. Extract principal components
principal_components = V[:4, :].T  # Transposeto get (features x n_components)
# 4. Calculate explained variance ratio
explained_variance_ratio = (S**2) / np.sum(S**2)
explained_variance_ratio = explained_variance_ratio[:4]

# 5. Project the data
X_reduced = df @ principal_components

# Print the results
print("Reduced data:\n", X_reduced)
print("\nPrincipal components:\n", principal_components)
print("\nExplained variance ratio:\n", explained_variance_ratio)
# %%

# Create the scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()