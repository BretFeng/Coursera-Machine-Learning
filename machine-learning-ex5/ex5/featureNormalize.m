function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);  % calculate the mean of every column. mu is a 1*p matrix
X_norm = bsxfun(@minus, X, mu);   % every row of matrix X will minus matrix mu. X is a m*p matrix, mu is a 1*p matrix

sigma = std(X_norm); % calculate the standard deviation of every column. sigma is a 1*p matrix
X_norm = bsxfun(@rdivide, X_norm, sigma); % every row of matrix X_norm will point divide matrix sigma. X_norm is a m*p matrix, sigma is a 1*p matrix


% ============================================================

end
