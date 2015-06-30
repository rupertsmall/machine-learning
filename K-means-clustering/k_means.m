% find seed candidates for implementation of the K-means algorithm
% X x-vals, Y y-vals, 1 <= labels <= K for each (X,Y) pair 
 
function [] = k_means()
 
clear all
close all
 
K = 5;
points = 50;
N = 2;
Kp = K*points;
error = 5;
epsilon = .0000001;
 
[X Y] = create_k_cluster(K,points);
 
figure
plot(X,Y,'.')
hold on
 
saveX=X;
saveY=Y;
 
X = reshape(X, Kp, 1);
Y = reshape(Y, Kp, 1);
[Xk Yk success] = seed_k_means(X,Y,K,N);
dimension = success + K;
 
plot(Xk,Yk,'ok','MarkerEdgeColor','k', 'MarkerFaceColor','k','MarkerSize',5)
grid
 
expandClusters = ones(dimension,1);
bigClustersX = kron(X,expandClusters);
bigClustersY = kron(Y,expandClusters);
bigClustersX = reshape(bigClustersX,dimension,Kp);
bigClustersY = reshape(bigClustersY,dimension,Kp);
 
while (error > epsilon)
    bigXk = kron(Xk, ones(1,Kp));
    bigYk = kron(Yk, ones(1,Kp));
    bigDiffX = (bigClustersX - bigXk).^2;
    bigDiffY = (bigClustersY - bigYk).^2;
    bigDiff = bigDiffX + bigDiffY;
    [ignore index] = min(bigDiff);
    
    for i=1:dimension
       fetch = logical(index == i);
       elmnts = sum(fetch);
       errX = Xk(i);
       errY = Yk(i);
       Xk(i) = sum(X(fetch))/elmnts;
       Yk(i) = sum(Y(fetch))/elmnts;
       errX = (errX - Xk(i)).^2;
       errY = (errY - Yk(i)).^2;
       error = errX + errY;
    end
end
 
figure
plot(saveX,saveY,'.')
hold on
plot(Xk,Yk,'pk','MarkerEdgeColor','k', 'MarkerFaceColor','y','MarkerSize',12)
grid
size(Xk)
end
