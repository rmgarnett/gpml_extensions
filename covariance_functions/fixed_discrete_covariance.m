% FIXED_DISCRETE_COVARIANCE for discrete points with precomputed covariance.
%
% This provides a GPML-compatible covariance function for a function
% defined on a set of n discrete points using a precomputed (n x n)
% covariance matrix K.
%
% This implementation assumes that the training and test inputs are
% given as integers between 1 and n, which simply index the provided
% matrix.
%
% Note that this function does not allow you to learn K; rather, K is
% assumed to be absolutely fixed. Should you wish to learn K, see
% instead covDiscrete.m.
%
% There are no hyperparameters requried.
%
% See also COVDISCRETE, FIXED_DISCRETE_MEAN, COVFUNCTIONS.

% Copyright (c) 2012--2015 Roman Garnett.

function result = fixed_discrete_covariance(K, ~, train_ind, test_ind, i, ~)

  % check for covariance matrix
  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'K must be specified!');
  end

  % number of hyperparameters
  if (nargin <= 2)
    result = '0';

  % training covariance
  elseif ((nargin == 3) || ((nargin == 4) && isempty(test_ind)))
    result = K(train_ind, train_ind);

  % diagonal training variance
  elseif ((nargin == 4) && strcmp(test_ind, 'diag'))
    result = K(sub2ind(size(K), train_ind, train_ind));

  % test covariance
  elseif (nargin == 4)
    result = K(train_ind, test_ind);

  % training derivatives
  elseif (isempty(test_ind))
    result = zeros(numel(train_ind));

  % diagonal training derivatives
  elseif (strcmp(i, 'diag'))
    result = zeros(numel(train_ind), 1);

  % test derivatives
  else
    result = zeros(numel(train_ind), numel(test_ind));
  end

end
