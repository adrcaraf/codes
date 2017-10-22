classdef MlpNdekf < handle
    properties (SetAccess = private)
        %MLP properties
        
        nW      %number of weight matrices
        nnCfg   %neural network configuration
        W       %weight matrices
        G       %local gradients
        Y       %function signals
        V       %induced fields
        Xtrain  %train data
        Dtrain  %train desired output
        Xtest   %test data
        Dtest   %test desired output
        Wbest
        iBatchSize = 1;
        
        %GEKF properties
        
        P       %approximate error covariance matrix (n)
        Q       %process noise covariance matrix
        R       %measurement noise covariance matrix
        dW      %matrix of partial derivatives of each output
        nNodes
        west
        C
    end
    
    properties (SetAccess = public)
        %MLP properties
        iEpochs = 1
        actv = @(x) tanh(.6667*x)
        actv_p = @(x)  .6667*(sech(.6667*x)).^2
        keepBest = 0;
        stopChangeTol = .1
        stopTol = .005
        
        %GEKF properties
        eErrCov = .001 
        eLearnRate = .1
        eProcNoiseCov = 1e-4
        
        %Results
        trainAccuracy = 0
        testAccuracy = 0
        lastPrediction
        
    end
    
    methods (Access = public)
        function obj = MlpNdekf(Xtrain_, Dtrain_, hiddenCfg_, Xtest_, Dtest_)
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