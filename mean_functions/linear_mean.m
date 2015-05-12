% LINEAR_MEAN linear mean function.
%
% This provides a GPML-compatible mean function implementing a
% linear mean function:
%
%   \mu(x) = a' x.
%
% This can be used as a drop-in replacement for meanLinear.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of \mu with respect to any pair of
% hyperparameters. The syntax is:
%
%   dm2_didj = linear_mean(theta, x, i, j)
%
% where dm2_didj is \partial^2 \mu / \partial \theta_i \partial \theta_j,
% and the Hessian is evaluated at x.
%
% These Hessians can be used to ultimately compute the Hessian of the
% GP training likelihood (see, for example, exact_inference.m).
%
% The hyperparameters are the same as for meanLinear.
%
% See also MEANLINEAR, MEANFUNCTIONS.

% Copyright (c) 2014--2015 Roman Garnett.

function result = linear_mean(hyperparameters, x, i, ~)

  % report number of hyperparameters
  if (nargin <= 1)
    result = 'D';
    return;
  end

  num_points = size(x, 1);

  % evaluate prior mean
  if (nargin == 2)
    result = x * hyperparameters(:);

  % evaluate derivative with respect to hyperparameter
  elseif (nargin == 3)
    result = x(:, i);

  % evaluate second derivative with respect to hyperparameter
  else
    result = zeros(num_points, 1);
  end

end