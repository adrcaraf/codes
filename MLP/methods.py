import numpy
from matplotlib import pyplot as plt
import sys

def _train(self):
	__learn(self, self.nnCfg, self.Xtrain, self.Dtrain, self.W, self.lr, 
		self.batch_sz, self.reg, self.epochs_n, self.mmtum)

def __learn(self,cfg, _X, _D, W, lr, batch_sz, reg, epochs_n, mmtum):
	#initialization
	
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
		for _ in range(nW):
			W.append(None)
		
		for i in range(nW):
			W[i] = numpy.random.rand(cfg[i]+1, cfg[i+1])-.5
			W[i]*=(12./(cfg[i]+1))**.5
	
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
				Y[i+1] = self.actv(V[i])

			E = Dn - Y[-1]
			G[-1] = numpy.multiply(E, self.actvp(V[-1]))
			
			acc_err += (.5*E**2).sum()

			for i in reversed(range(nW)):
				if i > 0:
					G[i-1] = numpy.multiply(self.actvp(V[i-1]), 
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
	
	sys.stdout.write('\n')
	plt.plot(numpy.arange(1,epochs_n+1), Ec)
	plt.draw()

def ___train(self):
	#initialization
	
	#aliases	
	cfg = self.nnCfg
	lr, batch_sz, reg = self.lr, self.batch_sz, self.reg
	epochs_n, mmtum = self.epochs_n, self.mmtum
	
	#plotting
	Ec = numpy.zeros(epochs_n)
	
	#locals 
	tmpf = 0

	X = self.Xtrain.copy()
	D = self.Dtrain.copy()

	#if the ratio is not an integer, than training data will be lost
	K = int(X.shape[0]/batch_sz)

	O, Delta0 = numpy.ones([batch_sz, 1]), None

	nW = len(cfg)-1

	#start
	self.W = [None]*nW
	V = [None]*nW
	Y = [None]*(nW+1)
	G = [None]*nW
	
	for i in range(nW):
		self.W[i] = numpy.random.rand(cfg[i]+1, cfg[i+1])-.5
		self.W[i]*=(12./(cfg[i]+1))**.5
	
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
				V[i] = numpy.dot(numpy.hstack([O, Y[i]]), self.W[i])
				Y[i+1] = self.actv(V[i])

			E = Dn - Y[-1]
			G[-1] = numpy.multiply(E, self.actvp(V[-1]))
			
			acc_err += (.5*E**2).sum()

			for i in reversed(range(nW)):
				if i > 0:
					G[i-1] = numpy.multiply(self.actvp(V[i-1]), 
							numpy.dot(G[i], self.W[i][1:,:].T))

				Delta1 = numpy.dot(lr*numpy.hstack([O, Y[i]]).T, G[i])

				self.W[i] = self.W[i] + Delta1/batch_sz

				if reg > 0:
					self.W[i][1:,:] = self.W[i][1:,:] - reg*self.W[i][1:,:]/batch_sz

				if mmtum > 0:
					if k>0:
						self.W[i] = self.W[i] + mmtum*Delta0
					Delta0 = Delta1.copy()
				
		
		Ec[e] = acc_err/batch_sz/K
	
	sys.stdout.write('\n')
	plt.plot(numpy.arange(1,epochs_n+1), Ec)
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
	p=self.predict(x).argmax(axis=1)
	print(numpy.hstack([p[:10,numpy.newaxis],d[:10,numpy.newaxis]]))
	return numpy.sum(numpy.equal(p,d)*1.)/d.shape[0]