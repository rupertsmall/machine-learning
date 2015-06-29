% Create K synthetic data clusters, N points per cluster
 
function [clustersX, clustersY] = create_k_cluster(K,N)
 
clustersX=zeros(N,K);   % x-vals for clusters
clustersY=zeros(N,K);   % y-vals for clusters
var_upper=1;            % upper limit of cluster variance
 
for i=1:K               % initialise clusters
    meanX=random('unif',0,15*K);
    meanY=random('unif',0,15*K);
    variance=random('unif',0.1*K,var_upper*K);
    clustersX(:,i)=normrnd(meanX, variance, N,1);
    clustersY(:,i)=normrnd(meanY, variance, N,1);
end
 
end
