function obj = initialize_(obj)
    %MLP

    nL = length(obj.nnCfg); %total number of layers
    obj.nW = nL - 1;
    
    obj.W = cell(1,obj.nW);
    obj.dW = cell(1,obj.nW);
    obj.Y = cell(1,nL);
    obj.G = cell(1,obj.nW);
    obj.V = cell(1,obj.nW);
    
    %first function signal is, in reality, the input pattern
    obj.Y{1} = zeros(1, obj.nnCfg(1));
        
    for i=1:obj.nW
        obj.W{i} = weights__(obj.nnCfg(i)+1,obj.nnCfg(i+1));
        obj.dW{i} = zeros(obj.nnCfg(i)+1,obj.nnCfg(i+1));
        obj.Y{i+1} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
        obj.G{i} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
        obj.V{i} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
    end
    
    %initialize measurement noise cov. matrix
    obj.R = eye(obj.nnCfg(end))/obj.eLearnRate;
    
    nNodes = sum(obj.nnCfg(2:end));
    obj.nNodes = nNodes;
    obj.P = cell(1,nNodes);
    obj.Q = cell(1,nNodes);
    obj.west = cell(1,nNodes);
    obj.C = cell(1,nNodes);
    
    %calculate the total # of weights
    k=1;
    for i=1:obj.nW
        for j=1:obj.nnCfg(i+1)
            %Initialize the approximate error cov. matrix linked to each output
            obj.P{k} = eye(obj.nnCfg(i)+1)./obj.eErrCov;
            
            %Initialize the process noise cov. matrix
            obj.Q{k} = eye(obj.nnCfg(i)+1).*obj.eProcNoiseCov;
            
            %Initialize k-th group weight estimate
            obj.west{k} = zeros(1, obj.nnCfg(i)+1);
            
            %Initialize k-th group partial derivative
            obj.C{k} = zeros(obj.nnCfg(end), obj.nnCfg(i)+1);
            
            k = k + 1;
        end
    end        
end

function w = weights__(n0, n1)
    %Initialize weights with variance proportinal to the inverse of the
    %number of weights connected to each neuron. 
    %See Simon Haykin - Neural networks a comprehensive foundation (2008),
    %section 4.6, subsection "Initialization" 
    w = rand(n0, n1) - .5;
    w = w.*sqrt(12/n0);
end