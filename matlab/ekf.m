clear nnEkf; clc;
rng(10);

fid = fopen('mg30.dat', 'r');
data = textscan(fid, '%f'); data = data{1};
fclose(fid);
clear fid;

%remove mean
data = data - mean(data);
data = data/max(abs(data));

% figure;
% plot(data, 'k', 'linewidth', 1.2);
% ylabel('y'); title('Mackey-Glass Series');
% axis([0 1500 .2 1.4])
 
%assemble training data
stepAhead = 1;
backSupp  = 8;

M = size(data,1);
K = (M - stepAhead - backSupp) + 1;

xtrain = zeros(K , backSupp);
ytrain = zeros(K, stepAhead);

for k=1:K
    xtrain(k, :) = data(k:k+backSupp-1)';
    ytrain(k, :) = data(k+backSupp:k+backSupp+stepAhead-1)';
end
%% Global EKF
tic;
nnEkf = MlpGekf(xtrain, ytrain, [10]); %last argument is the hidden layer 
                                       %configuration: [l2 l3 .. ln]

nnEkf.iEpochs = 5;
nnEkf.eLearnRate = .1; %eta
nnEkf.eProcNoiseCov = .001; %q
nnEkf.eErrCov = .001; %epsilon
nnEkf.stopChangeTol = 0; nnEkf.stopTol = 0;

nnEkf.train();
toc
%%
close
%clc
ypred = zeros(size(data,1),1);
ypred(1:backSupp) = data(1:backSupp);
figure;
subplot(211);
plot(data, '--.k');hold on;

for k=1:stepAhead:K
    v1 = k+backSupp:k+backSupp+stepAhead-1;
    v2 = k:k+backSupp-1;
    tmp = nnEkf.predict(data(v2)').lastPrediction;
    ypred(v1) = tmp;
    %subplot(211);
    %plot(v1, tmp, '--.r');
end
subplot(211);
plot(ypred, '--.r');
str = sprintf('%f', sum((data-ypred).^2)/sum((data-mean(data)).^2));
%title(['FKEG p = 8, s = 4, l_2 = 25, l_3 = 25, EQMN = ' str ' e 5 épocas em 708[s]'])

subplot(212);
v = 100:200;
plot(v, data(v), '--.k');hold on;
plot(v, ypred(v), '--.r');
legend('real', 'pred.')
%% Global EKF XOR problem
% clear nn;
% rng(10);
% 
% X = [0 0; 1 0; 0 1; 1 1];
% D = [-1; 1; 1; -1];
% 
% nnEkf = MlpGekf(X, D, [4]);
% nnEkf.iBatchSize = 1;
% nnEkf.iEpochs = 100;
% nnEkf.eLearnRate = 1;
% nnEkf.eProcNoiseCov = 1e-4;
% nnEkf.eErrCov = .1;
% 
% nnEkf.train();


%% MLP with gradient descent
rng(10);
clear nn;

nn = MLP(xtrain, ytrain, [24]);
nn.learnParam = .1; nn.regParam = 0;
nn.momentum = 0;
nn.iBatchSize = 1; nn.iEpochs = 50;
nn.plotTest = 0;
nn.stopChangeTol = .001; nn.stopTol = 1e-6;
nn.train();

%%
close
ypred = zeros(size(data,1),1);
ypred(1:backSupp) = data(1:backSupp);
figure;
subplot(211);
plot(data, '--.k');hold on;

for k=1:stepAhead:K
    v1 = k+backSupp:k+backSupp+stepAhead-1;
    v2 = k:k+backSupp-1;
    tmp = nn.predict(data(v2)').lastPrediction;
    ypred(v1) = tmp;
    %subplot(211);
    %plot(v1, tmp, '--.r');
end
subplot(211);
plot(ypred, '--.r');
 str = sprintf('%f', sum((data-ypred).^2)/sum((data-mean(data)).^2));
title(str)

subplot(212);
v = 100:200;
plot(v, data(v), '--.k');hold on;
plot(v, ypred(v), '--.b');
legend('real', 'pred.')
%% Decoupled EFK
tic;
nnEkf = MlpNdekf(xtrain, ytrain, [10]);

nnEkf.iEpochs = 5;
nnEkf.eLearnRate = .1;
nnEkf.eProcNoiseCov = .001;
nnEkf.eErrCov = .001;
nnEkf.stopChangeTol = 0; nnEkf.stopTol = 0;

nnEkf.train();
toc
%%
close
%clc
ypred = zeros(size(data,1),1);
ypred(1:backSupp) = data(1:backSupp);
figure;
subplot(211);
plot(data, '--.k');hold on;

for k=1:stepAhead:K
    v1 = k+backSupp:k+backSupp+stepAhead-1;
    v2 = k:k+backSupp-1;
    tmp = nnEkf.predict(data(v2)').lastPrediction;
    ypred(v1) = tmp;
    %subplot(211);
    %plot(v1, tmp, '--.r');
end
subplot(211);
plot(ypred, '--.r');
str = sprintf('%.4f', sum((data-ypred).^2)/sum((data-mean(data)).^2));
title(['FKEDN p = 8, s = 4, l_2 = 25, l_3 = 25, EQMN = ' str ' e 5 épocas em 52[s]'])

subplot(212);
v = 100:200;
plot(v, data(v), '--.k');hold on;
plot(v, ypred(v), '--.r');
legend('real', 'pred.')