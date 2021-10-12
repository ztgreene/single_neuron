%% Hamiltonian Monte Carlo for Bayesian treatment of Single Neuron

% import the data
load test1.dat
X = [ones(8,1) test1(:,1:2)];
t = test1(:,3);

% Variables to tune
lag = 100
burn_in = 2000
L = burn_in + 30*lag;              % Number of loops
Tau = 200;                         % # of leapfrog steps
epsilon = 0.055                     
alpha = 0.01                       % Regularization


W = [0 0 0];                       % Initialize weights
W_stored = zeros(L, 3);

% Defining posterior functions
y = @(W) sigmf(W*X', [1 0]);
G = @(W) -(t'*log(y(W)') + (1-t')*log(1-y(W))') + alpha*sum(W.^2, 2)'/2; % THIS IS OBJECTIVE FUNCTION
P = @(W) exp(-G(W));    % This is the likelihood function

gradM = @(W) -(t' - y(W))*X +alpha*W;         % Gradient of objective function

% Plot data
figure(1); clf
plot(X(1:4,2),X(1:4,3),'ks'); hold on
plot(X(5:8,2),X(5:8,3),'k*')
xlim([0 10]); ylim([0 10]); axis square
xlabel('x1'); ylabel('x2')


%% Hamiltonian Monte Carlo
gM = gradM(W);                   % Set gradient using initial weights
E = G(W);                   % Set objective function
for l = 1: L-1                    % loop L times
    p = randn(size(W));         % initial momentum is Normal(0,1)
    H = p'*p/2 + E;              % Evalutate H(x,p)
    wnew = W; gnew = gM;
    for tau = 1:Tau             % Make Tau 'leapfrog' steps
        p = p - epsilon*gnew/2;  % Make half-step in p
        wnew = wnew + epsilon*p; % Make step in x
        gnew = gradM(wnew);     % Find new gradient
        p = p - epsilon*gnew/2;  % Make half-step in p
    end
    Enew = G(wnew);         % Find new value of H
    Hnew = p'*p/2 + Enew;
    dH = Hnew - H;              % Decide whether to accept
    if (dH < 0)
        accept = 1;
    elseif (rand() < exp(-dH))
        accept = 1;
    else
        accept = 0;
    end
    if (accept)
        gM = gnew; x = wnew; E = Enew;
    end
        W_stored(l+1,:) = wnew;
end


% Sum sampled output functions to find average neuron output
W_indep = W_stored(burn_in+lag:lag:L,:);
learned_y = @(x) zeros(1, length(x));
for i = 1:length(W_indep)
    W = W_indep(i,:);
    learned_y = @(x) [learned_y(x); sigmf(W*x', [1 0])];
end
learned_y = @(x) sum(learned_y(x))/length(W_indep);

% Plots 
figure(1); clf
% Sample autocorrelation
subplot(1,2,1)
acf(W_stored(:,2), lag);
% Predictive distribution
subplot(1,2,2)
plot(X(1:4,2), X(1:4,3), 'ks'); hold on
plot(X(5:8,2), X(5:8,3), 'k*')
xlim([0 10]); ylim([0 10]); axis square
title('Predictive Distribution'); xlabel('x1'); ylabel('x2')
hold on
x1 = linspace(0,10);
x2 = x1;
[x1 x2] = meshgrid(x1, x2);
learned_y_cont = reshape(learned_y([ones(10000,1) x1(:) x2(:)]), 100, 100);
contour(x1, x2, learned_y_cont, [0.12 0.27 0.73 0.88], '--k'); hold on
contour(x1, x2, learned_y_cont, [0.5 0.5], 'k')


% The decision boundary seems to fit the data better than the simple
% gradient descent using a single neuron; it is much less certain away from the
% data. The autocorrelation function shows very low correlation between
% samples, with even very low levels of lag (less than 100), and no 
% appreciable trend over varying levels of lag In fact, varying lag seems 
% to have relatively little effect on the results of the predictive
% distribution.
