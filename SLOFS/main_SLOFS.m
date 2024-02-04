clc;
clear;
addpath(genpath('.'));
starttime = datestr(now,0);
load('data/Scene_data.mat')

[num_train, num_label] = size(Y_train); [num_test, num_feature] = size(X_test);
pca_remained = round(num_feature*0.95);

% Performing PCA
all = [X_train; X_test];
ave = mean(all);
all = (all'-concur(ave', num_train + num_test))';
covar = cov(all); covar = full(covar);
[u,s,v] = svd(covar);
t_matrix = u(:, 1:pca_remained)';
all = (t_matrix * all')';
X_train = all(1:num_train,:); X_test = all((num_train + 1):(num_train + num_test),:);

fea=X_train; gnd=Y_train;
% Parameter
alpha=0.1; beta=0.9; lamda1=1; lamda2=0.1; delta=0.01; gamma=0.9;
nClass1=length(unique(gnd));

tic
[W,obj] =SLOFS(fea,gnd,nClass1,alpha,beta,lamda1,lamda2,delta);
tt=toc;
score= sqrt(sum(W.*W,2));
[res, idx] = sort(score,'descend');

t_feature=round(num_feature*0.2);
feature_idx = idx(1:t_feature);
Num = 10;Smooth = 1;
for i = 1:t_feature
    fprintf('Running the program with the selected features - %d/%d \n',i,t_feature);
    f=feature_idx(1:i);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(X_train(:,f),Y_train',Num,Smooth);
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,macrof1,microf1,SubsetAccuracy,Outputs,Pre_Labels]=...
        MLKNN_test(X_train(:,f),Y_train',X_test(:,f),Y_test',Num,Prior,PriorN,Cond,CondN);
    HL_MDFS(i)=HammingLoss;
    RL_MDFS(i)=RankingLoss;
    OE_MDFS(i)=OneError;
    CV_MDFS(i)=Coverage;
    AP_MDFS(i)=Average_Precision;
    MA_MDFS(i)=macrof1;
    MI_MDFS(i)=microf1;
    AC_MDFS(i)=SubsetAccuracy;
end
Mean_slofs=[mean(HL_MDFS),mean(RL_MDFS),mean(OE_MDFS),mean(CV_MDFS),mean(AP_MDFS),mean(MA_MDFS),mean(MI_MDFS),mean(AC_MDFS)]

