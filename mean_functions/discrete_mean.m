% DISCRETE_MEAN for discrete points.
%
% This provides a GPML-compatible mean function for a function defined
% on a set of n discrete points using a discrete (N x 1) mean vector
% mu.
%
% This implementation assumes that the training and test inputs are
% given as integers between 1 and N, which simply index the provided
% vector.
%
% Note that this function does not allow you to learn mu; rather, mu
% is assumed to be absolutely fixed. Should you wish to learn mu, see
% instead discrete_mean.m.
%
% There are no hyperparameters requried.
%
% See also DISCRETE_MEAN, FIXED_DISCRETE_COVARIANCE, MEANFUNCTIONS.

% Copyright (c) 2014 Roman Garnett.

function result = discrete_mean(n, hyperparameters, x, i)

  % check for dimension
  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'n must be specified!');
  end

  % report number of hyperparameters
  if (nargin <= 2)
    result = num2str(n);

  % evaluate prior mean
  elseif (nargin == 3)
    result = mu(x);

  % evaluate derivatives with respect to hyperparameters
  else
    result = zeros(numel(x), 1);
    result(x == i) = 1;
  end

end
