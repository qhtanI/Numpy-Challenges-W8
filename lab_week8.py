import numpy as np

# Day 36: Vectors
[span_0](start_span)feature = np.array([30.0, 50.0, 10.0])[span_0](end_span)
[span_1](start_span)weights = np.array([0.05, 0.8, -0.1])[span_1](end_span)

v_add = feature + weights
v_scale = feature * 0.5
f_norm = np.linalg.norm(feature)
print(feature.shape, weights.shape)

# Day 37: Dot Product
[span_2](start_span)a = np.array([1.0, 2.0, 3.0])[span_2](end_span)
[span_3](start_span)b = np.array([0.5, 1.0, 1.5])[span_3](end_span)

dot_res = np.dot(a, b)
cos_sim = dot_res / (np.linalg.norm(a) * np.linalg.norm(b))

# Day 38: Matrices
[span_4](start_span)X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])[span_4](end_span)
[span_5](start_span)W = np.array([[0.1, -0.2], [0.4, 0.0], [-0.3, 0.5]])[span_5](end_span)

Y_mat = X @ W
[span_6](start_span)print(X.shape, W.shape, Y_mat.shape)[span_6](end_span)

# Day 39: Broadcasting
[span_7](start_span)X_data = np.array([[3.0, 4.0], [1.0, 2.0], [0.0, 5.0]])[span_7](end_span)

row_norms = np.linalg.norm(X_data, axis=1, keepdims=True)
X_normed = X_data / row_norms
[span_8](start_span)print(np.linalg.norm(X_normed, axis=1))[span_8](end_span)

# Day 40: Matrix Operations
[span_9](start_span)X_in = np.array([[1.0, 0.5], [2.0, -1.0], [0.0, 3.0]])[span_9](end_span)
[span_10](start_span)W_in = np.array([[0.2, 0.1, 0.5], [0.7, 0.3, -0.2]])[span_10](end_span)
[span_11](start_span)b = np.array([0.1, 0.0, -0.3])[span_11](end_span)

[span_12](start_span)Y_final = (X_in @ W_in) + b[span_12](end_span)
[span_13](start_span)print(Y_final.shape)[span_13](end_span)
