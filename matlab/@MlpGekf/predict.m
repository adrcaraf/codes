function [obj] = predict(obj, X)
    obj.Y{1} = X;
    Ones = ones(size(X,1), 1);
        
    for i=1:obj.nW
        obj.Y{i+1} = obj.actv([Ones obj.Y{i}]*obj.W{i});
    end
    
    obj.lastPrediction = obj.Y{i+1};
end