% DISCRETE_COVARIANCE for discrete points.
%
% This provides a GPML-compatible covariance function for a function
% defined on a set of n discrete points using a discrete (n x n)
% covariance matrix K.
%
% This implementation assumes that the training and test inputs are
% given as integers between 1 and n, which simply index the provided
% matrix.
%
% The hyperparameters specify the lower triangluar part of the
% covariance matrix:
%
%   hyperparameters = [ K_11 ...
%                       K_21 ...
%                       ....
%                       K_n1 ...
%                       K_22 ...
%                       ....
%                       K_n2 ...
%                       ....
%                       K_nn ].
%
% This can be generated, e.g., with
%
%   hyperparameters = K(tril(true(n)));
%
% See also FIXED_DISCRETE_COVARIANCE, COVFUNCTIONS.

% Copyright (c) 2014 Roman Garnett.

function result = discrete_covariance(n, hyperparameters, train_ind, test_ind, i, ~)

  % check for covariance matrix
  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'n must be specified!');
  end

  % number of hyperparameters
  if (nargin <= 2)
    result = num2str(n * (n + 1) / 2);
    return;
  end

  % build K
  K = zeros(n);
  K(tril(true(n))) = hyperparameters(:);
  K = K + tril(K, -1)';

  % training covariance
  if ((nargin == 3) || ((nargin == 4) && isempty(test_ind)))
    result = K(train_ind, train_ind);
    return;

  % diagonal training variance
  elseif ((nargin == 4) && strcmp(test_ind, 'diag'))
    diagonal = diag(K);
    result = diagonal(train_ind);
    return;

  % test covariance
  elseif (nargin == 4)
    result = K(train_ind, test_ind);
    return;
  end

  % derivatives

  % build lookup matrix
  A = zeros(n);
  A(tril(true(n))) = (1:numel(hyperparameters(:)));

  [row, column] = find(A == i);

  % training derivatives
  if (isempty(test_ind))
    result = zeros(numel(train_ind));
    result(train_ind == row, train_ind == column) = 1;
    result(train_ind == column, train_ind == row) = 1;

  % diagonal training derivatives
  elseif (strcmp(test_ind, 'diag'))
    [row, column] = find(A == i);
    result = zeros(numel(train_ind), 1);
    if (row == column)
      result(train_ind == row) = 1;
    end

  % test derivatives
  else
    result = zeros(numel(train_ind), numel(test_ind));
    result(train_ind == row,    test_ind == column) = 1;
    result(train_ind == column, test_ind == row) = 1;
  end

end