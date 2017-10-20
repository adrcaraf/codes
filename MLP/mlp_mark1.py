import numpy
from methods import _train, _predict, _accuracy

class Mlp:
	def __init__(self, hiddenCfg = None, Xtrain=None, Dtrain=None):
		self.actv = numpy.vectorize(lambda x: numpy.tanh(.6667*x))
		self.actvp = numpy.vectorize(lambda x: .6667*(1-numpy.tanh(.6667*x)**2))
		self.lr = .01
		self.batch_sz = 1
		self.epochs_n = 1
		self.mmtum = .0
		self.reg = .0
		self.Xtrain, self.Dtrain = None, None
		self.nnCfg = []
		self.W = []
		
		if Xtrain.any():
			self.Xtrain = Xtrain
		if Dtrain.any():
			self.Dtrain = Dtrain
		
		if len(self.Xtrain.shape) == 1:
			self.Xtrain = self.Xtrain[:,numpy.newaxis]
		if len(self.Dtrain.shape) == 1:
			self.Dtrain = self.Dtrain[:,numpy.newaxis]
		
		self.nnCfg.append(self.Xtrain.shape[1])
		for s in hiddenCfg:
			self.nnCfg.append(s)
		self.nnCfg.append(self.Dtrain.shape[1])

	train =	_train
	predict = _predict
	accuracy = _accuracy