function [obj] = accuracy(obj)
    obj.trainAccuracy = acc_(obj.Xtrain, obj.Dtrain, obj);
    if ~isempty(obj.Xtest)
        obj.testAccuracy = acc_(obj.Xtest, obj.Dtest, obj);
    end
end

function [acc] = acc_(X,D,obj)
    obj.Y{1} = X;
    Ones = ones(size(X,1), 1);
        
    for i=1:obj.nW
        obj.Y{i+1} = obj.actv([Ones obj.Y{i}]*obj.W{i});
    end
    
    if size(D,2)>1
        [v, P] = max(obj.Y{end}, [], 2);
        [v, D] = max(D, [], 2);

        acc =  mean(P==D);
    elseif size(D,2)==1        
    end
end
