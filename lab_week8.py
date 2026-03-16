import numpy as np

# Day 36: Vectors
feature = np.array([30.0, 50.0, 10.0])
weights = np.array([0.05, 0.8, -0.1])

v_add = feature + weights
v_scale = feature * 0.5
f_norm = np.linalg.norm(feature)
print(feature.shape, weights.shape)

# Day 37: Dot Product
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.5, 1.0, 1.5])

dot_res = np.dot(a, b)
cos_sim = dot_res / (np.linalg.norm(a) * np.linalg.norm(b))

# Day 38: Matrices
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
W = np.array([[0.1, -0.2], [0.4, 0.0], [-0.3, 0.5]])

Y_mat = X @ W
print(X.shape, W.shape, Y_mat.shape)

# Day 39: Broadcasting
X_data = np.array([[3.0, 4.0], [1.0, 2.0], [0.0, 5.0]])

row_norms = np.linalg.norm(X_data, axis=1, keepdims=True)
X_normed = X_data / row_norms
print(np.linalg.norm(X_normed, axis=1))

# Day 40: Matrix Operations
X_in = np.array([[1.0, 0.5], [2.0, -1.0], [0.0, 3.0]])
W_in = np.array([[0.2, 0.1, 0.5], [0.7, 0.3, -0.2]])
b = np.array([0.1, 0.0, -0.3])

Y_final = (X_in @ W_in) + b
print(Y_final.shape)
