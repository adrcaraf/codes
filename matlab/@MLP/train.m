function [obj] = train(obj)
    obj.initialize_();
    
    K = floor(size(obj.Xtrain,1)/obj.iBatchSize);
    Tmp = [obj.Xtrain obj.Dtrain];
    Ec = zeros(1,obj.iEpochs);
    
    Ones = ones(obj.iBatchSize, 1);
    
    if obj.plotTest
        Ones_test = ones(size(obj.Xtest,1),1);
        Ec_test = zeros(1,obj.iEpochs);
    end
    
for epoch = 1:obj.iEpochs

    Tmp = Tmp(randperm(size(Tmp,1)),:);
    D = Tmp(:,size(obj.Xtrain,2)+1:end);
    X = Tmp(:,1:size(obj.Xtrain,2));

    tmpE = zeros(1,K);
        
    for n = 1:K
        obj.Y{1} = X((n-1)*obj.iBatchSize+1:n*obj.iBatchSize,:);
        Dn = D((n-1)*obj.iBatchSize+1:n*obj.iBatchSize,:);
                
        %Forward Pass
        for i=1:obj.nW
            obj.V{i} = [Ones obj.Y{i}]*obj.W{i};
            obj.Y{i+1} = obj.actv(obj.V{i});
        end

        %Prior to Backward Pass
        E = e_(obj.Y{end}, Dn, 1);
        obj.G{obj.nW} = E .* obj.actv_p(obj.V{end});

        tmpE(n) = mean(.5*sum(E.^2,2));
        
        for i=obj.nW:-1:1
            if i>1
                obj.G{i-1} = obj.actv_p(obj.V{i-1}) .* (obj.G{i}*obj.W{i}(2:end,:)');
            end

            obj.Delta1{i} = obj.learnParam.*[Ones obj.Y{i}]'*obj.G{i};
            
            obj.W{i} = obj.W{i} + obj.Delta1{i}/obj.iBatchSize;
            
            if ~~obj.regParam
                obj.W{i} = obj.W{i}  - ...
                    obj.regParam*[zeros(1,size(obj.W{i},2)); ...
                    ones(size(obj.W{i},1)-1, size(obj.W{i},2))].*obj.W{i}...
                    /obj.iBatchSize;
            end
            
            if  ~~obj.momentum
                if n ~= 1
                    obj.W{i} = obj.W{i} + obj.momentum*obj.Delta0{i};
                end
                
                obj.Delta0{i} = obj.Delta1{i};
            end
            
        end
        
        clc;
        [epoch n]
    end
    
    if obj.plotTest
        y0 = obj.Xtest;
        
        for i=1:obj.nW
            y1 = [Ones_test y0]*obj.W{i};
            y0 = obj.actv(y1);
        end
        
        Ec_test(epoch) = mean(.5*sum((obj.Dtest-y0).^2,2));
    end
        
    Ec(epoch) = mean(tmpE(n));
    
    if epoch>1
        if abs(Ec(epoch)-Ec(epoch-1))/Ec(epoch-1) < obj.stopChangeTol ...
                || Ec(epoch) < obj.stopTol
            disp('tol stop');
            break;
        end
    end
    
end

figure;
plot(Ec); 
if obj.plotTest
    hold on;
    plot(Ec_test, 'k');
    legend('Train', 'Test')
end
xlabel('epoch'); ylabel('MSE');

end


function y = e_(x,d,s)
    switch s
        case 1
            y = d-x;
        case 2
            y = (1-d)./(1-x)/2 - (1+d)./(1+x)/2;
    end
end
