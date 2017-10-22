classdef MLP < handle
    properties (SetAccess = private)
        nW%number of weight matrices
        nnCfg %neural network configuration
        W %weight matrices
        G %local gradients
        Y %function signals
        V %induced fields
        Xtrain %train data
        Dtrain %train desired output
        Xtest %test data
        Dtest %test desired output
        Delta0 %delta update of current iteration
        Delta1 %delta update of last iteration  
    end
    
    properties (SetAccess = public)
        iEpochs = 1
        iBatchSize = 1
        regParam = 0
        learnParam = 1e-4
        momentum = 0        
        actv = @(x) tanh(.6667*x)
        actv_p = @(x) .6667*(sech(.6667*x)).^2
        stopChangeTol = .1
        stopTol = .005
        
        trainAccuracy = 0
        testAccuracy = 0
        lastPrediction
        
        plotTest = 0
    end
    
    methods (Access = public)
        function obj = MLP(Xtrain_, Dtrain_, hiddenCfg_, Xtest_, Dtest_)
            if nargin < 3 || nargin > 5
                error('Incorrect number of arguments');
            end
            
            if size(hiddenCfg_,1) > 1
                error('hiddenCfg_ shall be a line vector')
            end
            
            if size(Xtrain_,1) ~= size(Dtrain_,1)
                error('Number of samples in X and D shall be the same')
            end
            
            obj.Xtrain = Xtrain_;
            obj.Dtrain = Dtrain_;
            obj.nnCfg = [size(obj.Xtrain,2) hiddenCfg_ size(obj.Dtrain,2)];
            
            if nargin > 3
                obj.Xtest = Xtest_;
                obj.Dtest = Dtest_;
            end
        end    
        
        obj = train(obj)
        obj = accuracy(obj)
        obj = predict(obj, X)
    end
    
    methods (Access = private)
        obj = initialize_(obj)
    end   
    
end