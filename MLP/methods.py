import numpy
from matplotlib import pyplot as plt
import sys

def _train(self):
	__learn(self.nnCfg, self.Xtrain, self.Dtrain, self.W, self.lr, 
			self.batch_sz, self.reg, self.epochs_n, self.mmtum, 
			self.actv, self.actvp, self.stopTol)
	return self

def _autoencode(self, n_sz):
	W = []
	
	__learn([self.nnCfg[0], n_sz, self.nnCfg[0]], self.Xtrain, self.Xtrain,
				W, self.lr, self.batch_sz, self.reg, self.epochs_n,
				self.mmtum, self.actv, self.actvp, self.stopTol)
	
	O = numpy.ones([self.Xtrain.shape[0], 1])
	self.Xtrain = self.actv(numpy.dot(numpy.hstack([O, self.Xtrain]), W[0]))
	self.nnCfg[0] = n_sz
	self.Wac = W[0]
	
	return self

#pretrain with shallow autoencoder
def _pretrain(self):
	nW = len(self.nnCfg)-1
	Y = [None]*(nW)
	
	Y[0] = self.Xtrain
	for i in range(nW):
		print('Pretrain %d' % i)
		W = []
		
		__learn([self.nnCfg[i], self.nnCfg[i+1], self.nnCfg[i]], Y[i], Y[i],
				W, self.lr, self.batch_sz, self.reg, self.epochs_n,
				self.mmtum, self.actv, self.actvp, self.stopTol)
		
		if (i+1) != nW:
			O = numpy.ones([Y[i].shape[0], 1])
			Y[i+1] = self.actv(numpy.dot(numpy.hstack([O, Y[i]]), W[0]))
		
		self.W.append(W[0].copy())
	
	return self

def __learn(cfg, _X, _D, W, lr, batch_sz, reg, epochs_n, 
			mmtum, actv, actvp, stopTol=.0):
	#initialization
	print(cfg)
	#plotting
	Ec = numpy.zeros(epochs_n)
	
	#locals 
	tmpf = 0
	
	X = _X.copy()
	D = _D.copy()
	
	#if the ratio is not an integer, than training data will be lost
	K = int(X.shape[0]/batch_sz)
	
	O, Delta0 = numpy.ones([batch_sz, 1]), None
	
	nW = len(cfg)-1
	
	#start
	if not W: #W=[]
		print('a')
		for _ in range(nW):
			W.append(None)
		
		for i in range(nW):
			W[i] = numpy.random.rand(cfg[i]+1, cfg[i+1])-.5
			W[i]*=(12./(cfg[i]+1))**.5
	else:
		print('b')
	
	V = [None]*nW
	Y = [None]*(nW+1)
	G = [None]*nW
	
	for e in range(epochs_n):
		#always shuffle prior to beginning a new epoch
		if not tmpf:
			tmp = numpy.hstack([X,D])
			tmpf = 1

		numpy.random.shuffle(tmp)
		X = tmp[:, 0:X.shape[1]]
		D = tmp[:,-1*D.shape[1]:]
		
		acc_err = 0.

		for k in range(K):
			sys.stdout.write('\r%5d out of %4d, on epoch %5d'% (k,K,e))
			sys.stdout.flush()
			
			Y[0] = X[int(batch_sz*k):int((k+1)*batch_sz),:]
			Dn = D[int(batch_sz*k):int((k+1)*batch_sz),:]

			for i in range(nW):
				V[i] = numpy.dot(numpy.hstack([O, Y[i]]), W[i])
				Y[i+1] = actv(V[i])

			E = Dn - Y[-1]
			G[-1] = numpy.multiply(E, actvp(V[-1]))
			
			acc_err += (.5*E**2).sum()

			for i in reversed(range(nW)):
				if i > 0:
					G[i-1] = numpy.multiply(actvp(V[i-1]), 
							numpy.dot(G[i], W[i][1:,:].T))

				Delta1 = numpy.dot(lr*numpy.hstack([O, Y[i]]).T, G[i])

				W[i] = W[i] + Delta1/batch_sz

				if reg > 0:
					W[i][1:,:] = W[i][1:,:] - reg*W[i][1:,:]/batch_sz

				if mmtum > 0:
					if k>0:
						W[i] = W[i] + mmtum*Delta0
					Delta0 = Delta1.copy()
				
		
		Ec[e] = acc_err/batch_sz/K
		
		if e>0:
			if numpy.abs(Ec[e]-Ec[e-1])/Ec[e] < stopTol:
				Ec = Ec[:e]
				break
	
	sys.stdout.write('\n')
	plt.plot(numpy.arange(1,Ec.shape[0]+1), Ec)
	plt.draw()


def _predict(self, x):
	nW = len(self.nnCfg)-1
	
	if len(x.shape)==1:
		_x = x[:,numpy.newaxis].T.copy()
	else:
		_x = x.copy()
	
	o = numpy.ones((_x.shape[0],1))
	for i in range(nW):
		_x = self.actv(numpy.dot(numpy.hstack([o, _x]), self.W[i]))
	
	return _x

def _accuracy(self, x, d):
	
	if len(self.Wac):
		_x = x.copy()
		O = numpy.ones([_x.shape[0], 1])
		_x = self.actv(numpy.dot(numpy.hstack([O, _x]), self.Wac))
	else:
		_x=x
		
	p=self.predict(_x).argmax(axis=1)
	print(numpy.hstack([p[:10,numpy.newaxis],d[:10,numpy.newaxis]]))
	return numpy.sum(numpy.equal(p,d)*1.)/d.shape[0]