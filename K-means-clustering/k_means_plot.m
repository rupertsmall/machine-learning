% plot synthetic data and seed points
 
close all
clear all
 
K = 5;
points = 50;
N=10;
 
[X Y] = create_k_cluster(K,points);
figure
plot(X,Y,'.')
grid
 
figure
hold on
plot(X,Y,'.')
 
X = reshape(X, K*points, 1);
Y = reshape(Y, K*points, 1);
[Xk Yk success]=seed_k_means(X,Y,K,N);
 
hold on
plot(Xk, Yk,'ok','MarkerEdgeColor','k', 'MarkerFaceColor','k','MarkerSize',5)
grid
 
success
points*K
