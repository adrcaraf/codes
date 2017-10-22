function [obj] = train(obj)
    obj.initialize_();
    
    Kn = floor(size(obj.Xtrain,1)/obj.iBatchSize);
    Tmp = [obj.Xtrain obj.Dtrain];
    Ec = zeros(1,obj.iEpochs);
    
%     C = zeros(obj.nnCfg(end), obj.nWo);
%     cW = zeros(1, obj.nWo);
    
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
            
            %up to this point we have all partial derivatives of the k-th
            %output.
            
            node=1;
            for i=1:obj.nW
                for j=1:obj.nnCfg(i+1)
                    if i~=obj.nW
                        obj.C{node}(k,:) = obj.dW{i}(:,j)';
                    else
                        obj.C{node}(k,:) = obj.dW{i}';
                    end
                    
                    if ~cWdone
                        obj.west{node}(1,:) = obj.W{i}(:,j)';
                    end
                    
                    node = node + 1;
                end                
            end
            
            cWdone = 1;
        end
        
        A = 0;
        for i=1:obj.nNodes
            A = A + obj.C{i}*obj.P{i}*obj.C{i}';
        end
        
        A = pinv(obj.R + A);
        
        nW = 1; nC = obj.nnCfg(2:end); nP = 1;
        for i=1:obj.nNodes
            K = obj.P{i}*obj.C{i}'*A;
            obj.west{i} = obj.west{i} + E*K';
            obj.P{i} = obj.P{i} - K*obj.C{i}*obj.P{i} + obj.Q{i};
            
            %weigth matrix update
            obj.W{nW}(:,nP) = obj.west{i}';
            
            nP = nP + 1;
            nC(nW) = nC(nW)-1;
            if nC(nW)==0
                nW = nW + 1;
                nP = 1;
            end
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
