function obj = initialize_(obj)
    nL = length(obj.nnCfg); %total number of layers
    obj.nW = nL - 1;
    
    obj.W = cell(1,obj.nW);
    obj.Y = cell(1,nL);
    obj.G = cell(1,obj.nW);
    obj.V = cell(1,obj.nW);
    
    %first function signal is, in reality, the input pattern
    obj.Y{1} = zeros(obj.iBatchSize, obj.nnCfg(1));
        
    for i=1:obj.nW
        obj.W{i} = weights__(obj.nnCfg(i)+1,obj.nnCfg(i+1));
        obj.Y{i+1} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
        obj.G{i} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
        obj.V{i} = zeros(obj.iBatchSize, obj.nnCfg(i+1));
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