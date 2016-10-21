function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

%groundTruth = full(sparse(labels, 1:numCases, 1));
groundTruth = sparse(labels, 1:numCases, 1);
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

h = theta * data;
h = exp(bsxfun(@minus, h, max(h, [], 1)));
h = bsxfun(@rdivide, h, sum(h));

%cost = -mean(diag(h'*groundTruth)) + lambda*sum(sum(theta.^2), 2)/2;
%for i = 1:numCases
%    cost = cost + log_h(:, i)'*groundTruth(:, i);
%end;
%cost = -cost/numCases + lambda*sum(sum(theta.^2), 2)/2;
cost = -mean(sum(bsxfun(@times, log(h), groundTruth)), 2) + lambda*sum(sum(theta.^2), 2)/2;

gap = groundTruth - h;
for j=1:numClasses
    thetagrad(j, :) = -mean(bsxfun(@times, data, gap(j, :)), 2)' + lambda*theta(j, :);
end

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

