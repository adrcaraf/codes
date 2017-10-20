import mlp_mark1
import numpy as np
from matplotlib.pyplot import show

np.random.seed(10)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
D = np.array([-1,1,1,-1])

nn = mlp_mark1.Mlp([2], X, D)
nn.train(epochs_n=600, batch_sz=4, lr=1, mmtum=1., reg=.001)

print(nn.predict(np.array([1,1])))
print(nn.predict(np.array([0,1])))

Wd = {}

for i in range(len(nn.W)):
	Wd[i] = nn.W[i]

np.savez('state', weights=Wd)

show()

W = np.load('state.npz')
print(W['weights'])