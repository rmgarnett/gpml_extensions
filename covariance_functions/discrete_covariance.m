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
% The hyperparameters specify the upper-triangular part of the
% Cholesky factor of K, where the (positive) diagonal elements are
% specified by their logarithm.  If L = chol(K), so K = L' L, then:
%
%   hyperparameters = [ log(L_11) ...
%                           L_21  ...
%                       log(L_22) ...
%                           L_31  ...
%                           L_32  ...
%                           ....
%                       log(L_nn) ].
%
% This can be generated, e.g., with
%
%   L = chol(K);
%   L(1:(n + 1):end) = log(diag(L));
%   hyperparameters = L(triu(true(n)));
%
% See also FIXED_DISCRETE_COVARIANCE, COVFUNCTIONS.

% Copyright (c) 2014 Roman Garnett.

function result = discrete_covariance(n, hyperparameters, train_ind, test_ind, i)

  % check for dimension
  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'n must be specified!');
  end

  % number of hyperparameters
  if (nargin <= 2)
    result = num2str(n * (n + 1) / 2);
    return;
  end

  % build Cholesky factor of K
  L = zeros(n);
  L(triu(true(n))) = hyperparameters(:);
  L(1:(n + 1):end) = exp(diag(L));

  % covariance mode
  if (nargin <= 4)
    K = L' * L;

    if (nargin == 3)
      test_ind = [];
    end

    result = fixed_discrete_covariance(K, [], train_ind, test_ind);

  % derivatives mode
  else

    % build lookup matrix
    A = zeros(n);
    A(triu(true(n))) = (1:numel(hyperparameters(:)));

    [row, column] = find(A == i);

    % derivative of Cholesky factor
    dL = zeros(n);
    if (row == column)
      % diagonal entries have exp() transformation applied
      dL(row, column) = L(row, column);
    else
      dL(row, column) = 1;
    end

    % derivative of covariance matrix
    dK = dL' * L;
    dK = dK + dK';

    result = fixed_discrete_covariance(dK, [], train_ind, test_ind);
  end

end