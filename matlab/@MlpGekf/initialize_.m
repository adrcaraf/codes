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
        obj.Y{i+1} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
        obj.G{i} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
        obj.V{i} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
        
        if i==obj.nW
            obj.dW{i} = zeros(obj.nnCfg(i)+1, 1);
        else
            obj.dW{i} = zeros(obj.nnCfg(i)+1,obj.nnCfg(i+1));
        end
    end
    
    %GEKF
    
    %initialize measurement noise cov. matrix
    obj.R = eye(obj.nnCfg(end))/obj.eLearnRate;
    
    %calculate the # of weights linked to each output
    obj.nWo = 0;
    for i=1:obj.nW
        obj.nWo = obj.nWo + numel(obj.W{i});
    end
    
    %obj.nWo = obj.nWo + size(obj.W{end},1);
    
    %Initialize the approximate error cov. matrix linked to each output
    obj.P = eye(obj.nWo)/obj.eErrCov;
    
    %Initialize the process noise cov. matrix
    obj.Q = eye(obj.nWo)*obj.eProcNoiseCov;
    
end

function w = weights__(n0, n1)
    %Initialize weights with variance proportinal to the inverse of the
    %number of weights connected to each neuron. 
    %See Simon Haykin - Neural networks a comprehensive foundation (2008),
    %section 4.6, subsection "Initialization" 
    w = rand(n0, n1) - .5;
    w = w.*sqrt(12/n0);
end