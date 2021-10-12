%% Gradient Descent Algorithm for the Single Neuron
% dataset test1.dat is a toy data set
% alpha = 0.1 used for regularization.

% Load dataset
load test1.dat
X = [ones(8,1) test1(:,1:2)];
t = test1(:,3);

% Plot data
figure(1); clf
plot(X(1:4,2),X(1:4,3),'ks'); hold on
plot(X(5:8,2),X(5:8,3),'k*')
xlim([0 10]); ylim([0 10]); axis square
xlabel('x1'); ylabel('x2')

% Randomly initialize weights and bias
W = rand(1,3);

% Loop T times
T = 50000;
eta = 0.01;
alpha = 0.1;
y = @(W) sigmf(W*X', [1 0]);
for i = 1:T
    grad = -(t' - y(W))*X +alpha*W;
    W = W - eta*grad;
end

% Plot learned function
figure(1); hold on
learned_y = @(X) sigmf(W*X', [1 0]);
x1 = linspace(0,10);
x2=x1;
[x1 x2] = meshgrid(x1, x2);
learned_y_cont = reshape(learned_y([ones(10000,1) x1(:), x2(:)]), 100, 100);
contour(x1, x2, learned_y_cont, [0.27 0.73], '--k'); hold on
contour(x1, x2, learned_y_cont, [0.5 0.5], 'k')

% Seems overconfident away from decision boundary; that is, I don't believe
% the decision boundary is very reliable away from the data.
