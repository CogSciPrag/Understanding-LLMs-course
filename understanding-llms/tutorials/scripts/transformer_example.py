import numpy as np
import torch

def np_conv(name, mtx):
    mtx = str(mtx)[1:-1]
    out = f"${name}"" = \\begin{bmatrix}\n"
    for i, char in enumerate(mtx):
        if char == '[':
            out += "\t"
        elif char == ' ' and mtx[i-1].isdigit():
            out += ' & '
        elif char == ']':
            out += "\\\\"
        else:
            out += char
    out += "\n\end{bmatrix}$\\\\\\\\"
    return out



# Embedding matrix
E = np.array([
    [0, 1, 2],
    [6, 7, 1],
    [3, 4, 5],
    [0, 2, 1],
    [1, 3, 0],
    [3, 8, 6],
    [2, 7, 5],
    [6, 2, 1],
    [9, 1, 3],
    [0, 1, 1]])

# Query matrix
Q = np.array([
    [1, 1, 7],
    [2, 5, 1],
    [2, 6, 9]])

# Key matrix
K = np.array([
    [0, 4, 8],
    [1, 6, 9],
    [4, 2, 2]])

# Value Matrix
V = np.array([
    [2, 1, 0],
    [4, 3, 1],
    [6, 5, 1]])

# FFN matrix
W_f = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]])

# FFN bias
b_f = np.array([2, 1, 1]).T

# Output projection matrix
M_out = np.array([
    [0, 2, 1, 1, 3, 1, 0, 0, 4, 1],
    [1, 1, 3, 1, 0, 0, 4, 1, 0, 2],
    [1, 4, 0, 0, 1, 3, 1, 1, 2, 0]])

# one hot vecs for input sentence [BOS] the fox jumped [EOS]
I = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

X = I@E
print(np_conv('X', X))

# Calculate Q, K and V for each input vec
Q_x = Q@X.T
print(np_conv('Q_x', Q_x))

K_x = K@X.T
print(np_conv('K_x', K_x))

V_x = V@X.T
print(np_conv('V_x', V_x))

# compute attention scores
S = (Q@X.T).T@(K@X.T)
print(np_conv('S', S))

A = np.array([
    [0, -np.inf, -np.inf, -np.inf, -np.inf],
    [0, 0, -np.inf, -np.inf, -np.inf],
    [0, 0, 0, -np.inf, -np.inf],
    [0, 0, 0, 0, -np.inf],
    [0, 0, 0, 0, 0]])

S = S + A
print(np_conv('S', S))

# scale S by d_h
S = np.divide(S, np.sqrt(Q.shape[0]))
print(np_conv('S', S))

# normalise with softmax
S = np.round(np.array(torch.softmax(torch.tensor(S), dim=1)), decimals=5)
print(np_conv('S', S))

print(np.exp(520/1.73)/((np.exp(-np.inf/1.73)*3) + np.exp(570/1.73) + np.exp(520/1.73)))

# multiply score matrix with value matrix (i.e. z_0 = a^0_0 x v_0 + a^0_1 x v_1 + ... + a^0_4 x v_4 for all z)
R = S@V_x.T
print(np_conv('R', R))

# residual connection
R = R + X
print(np_conv('R', R))

# layer normalisation
epsilon = 0.00001
gamma = 1
beta = 0
Y = np.divide((np.subtract(R.T, np.mean(R, axis=1))), np.sqrt((np.var(R, axis=1) + epsilon))) * gamma + beta
print(np_conv('Y', np.round(Y, decimals = 3)))

# FFN forward pass
O = (W_f@Y).T + b_f
print(np_conv('O', np.round(O, decimals=3)))

# calculate logits over vocab
L = M_out.T@O.T
print(np_conv('L', np.round(L.T, decimals=3)))

# apply softmax to predict next token
probs = np.round(np.array(torch.softmax(torch.tensor(L), dim=0)), decimals=3)
print(np_conv('probs_T', probs.T))