from mlp_mark1 import Mlp
import numpy as np
from matplotlib.pyplot import show

np.random.seed(12)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
D = np.array([-1,1,1,-1])

nn = Mlp([2], X, D)
nn.epochs_n, nn.batch_sz =600, 4
nn.lr, nn.mmtum, nn.reg = 1, 1., .001

nn.train()

print(nn.predict(np.array([1,1])))
print(nn.predict(np.array([0,1])))

# Wd = {}

# for i in range(len(nn.W)):
	# Wd[i] = nn.W[i]

# np.savez('state', weights=Wd)

# show()

# W = np.load('state.npz')
# print(W['weights'])