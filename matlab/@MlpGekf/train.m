function [obj] = train(obj)
    obj.initialize_();
    
    Kn = floor(size(obj.Xtrain,1)/obj.iBatchSize);
    Tmp = [obj.Xtrain obj.Dtrain];
    Ec = zeros(1,obj.iEpochs);
    
    C = zeros(obj.nnCfg(end), obj.nWo);
    cW = zeros(1, obj.nWo);
    
    Ones = ones(obj.iBatchSize, 1);
    
    lastError = 1e10;
    
for epoch = 1:obj.iEpochs

    Tmp = Tmp(randperm(size(Tmp,1)),:);
    D = Tmp(:,size(obj.Xtrain,2)+1:end);
    X = Tmp(:,1:size(obj.Xtrain,2));
        
    tmpE = zeros(1,Kn);
    
    for n = 1:Kn
        obj.Y{1} = X((n-1)*obj.iBatchSize+1:n*obj.iBatchSize,:);
        Dn = D((n-1)*obj.iBatchSize+1:n*obj.iBatchSize,:);
                
        %Forward Pass
        for i=1:obj.nW
            obj.V{i} = [Ones obj.Y{i}]*obj.W{i};
            obj.Y{i+1} = obj.actv(obj.V{i});
        end

        %Prior to Backward Pass
        E = e_(obj.Y{end}, Dn, 1);
        
        tmpE(n) = mean(.5*sum(E.^2,2));
        
        if ~~obj.keepBest
            if tmpE(n) < lastError
                obj.Wbest = obj.W;
                lastError = tmpE(n);
            end
        end
        
        %all ones if no non-linear function is applied to the output
        %obj.G{obj.nW} = ones(size(obj.V{end}));
        
        %this case covers the case where non-linear functions are applied
        %to the output
        obj.G{obj.nW} = obj.actv_p(obj.V{end});
        
        cWdone = 0;
        for k=1:obj.nnCfg(end)
            Go = obj.G{end}(k);
            for i=obj.nW:-1:1
                if i>1    
                    if i==obj.nW
                        Wm = obj.W{i}(2:end,k);
                    else
                        Wm = obj.W{i}(2:end,:);
                    end
                    
                    obj.G{i-1} = obj.actv_p(obj.V{i-1}) .* (Go*Wm');
                end
                
                obj.dW{i} = [Ones obj.Y{i}]'*Go;
                
                if i>1
                    Go = obj.G{i-1};
                end
            end
            
            i0 = 1; i1 = 1;
            for i=1:obj.nW
                stp0 = numel(obj.W{i});
                if i~=obj.nW
                    C(k,i0:i0+stp0-1) = obj.dW{i}(:)';
                    i0=i0+stp0;
                else
                    tmp0 = zeros(size(obj.W{i}));
                    tmp0(:,k) = obj.dW{i};
                    C(k,i0:i0+stp0-1) = tmp0(:)';
                end
                
                if ~cWdone
                    cW(1, i1:i1+stp0-1) = obj.W{i}(:)';
                    i1=i1+stp0;
                end
            end
            
            cWdone = 1;
        end
        
        A = pinv(obj.R + C*obj.P*C');
        K = obj.P*C'*A;
        cW = cW + E*K';
        obj.P = obj.P - K*C*obj.P + obj.Q;
        
        %reconstruct temporary weight vector and update weight matrices
        b=1;
        for i=1:obj.nW
            tmp = numel(obj.W{i});
            obj.W{i} = reshape(cW(1, b:b+tmp-1), size(obj.W{i}));
            b = tmp + 1;
        end
        
        clc;
        [epoch n]
    end   
    
    Ec(epoch) = mean((tmpE(n)));
    
    if epoch>1
        if abs(Ec(epoch)-Ec(epoch-1))/Ec(epoch-1) < obj.stopChangeTol ...
                || Ec(epoch) < obj.stopTol
            disp('tol stop');
            break;
        end
    end
    
end

if ~~obj.keepBest
    obj.W = obj.Wbest;
end

figure;
plot(Ec); 
end


function y = e_(x,d,s)
    switch s
        case 1
            y = d-x;
        case 2
            y = (1-d)./(1-x)/2 - (1+d)./(1+x)/2;
    end
end
