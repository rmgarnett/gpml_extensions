% FIXED_DISCRETE_MEAN for discrete points with precomputed mean.
%
% This provides a GPML-compatible mean function for a function defined
% on a set of n discrete points using a precomputed (N x 1) mean
% vector mu.
%
% This implementation assumes that the training inputs are given as
% integers between 1 and N, which simply index the provided vector.
%
% There are no hyperparameters requried.
%
% See also FIXED_DISCRETE_COVARIANCE, MEANFUNCTIONS.

% Copyright (c) Roman Garnett, 2012--2014

function result = fixed_discrete_mean(mu, ~, x, i)

  % check for mean vector
  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'mu must be specified!');
  end

  % report number of hyperparameters
  if (nargin <= 2)
    result = '0';

  % evaluate prior mean
  elseif (nargin == 3)
    result = mu(x);

  % evaluate derivatives with respect to hyperparameters
  else
    result = zeros(numel(x), 1);
  end

end
