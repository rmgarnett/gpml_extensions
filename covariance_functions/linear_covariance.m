% LINEAR_COVARIANCE covLIN replacement with Hessian support
%
% This provides a GPML-compatible covariance function implementing the
% linear covariance. This can be used as a drop-in replacement for
% covLIN.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of K with respect to any pair of
% hyperparameters. The syntax is:
%
%   dK2_didj = linear_covariance(theta, x, z, i, j)
%
% where dK2_didj is \partial^2 K / \partial \theta_i \partial \theta_j,
% and the Hessian is evalauted at K(x, z). As in the derivative API,
% if z is empty, then the Hessian is evaluated at K(x, x). Note that
% the option of setting z = 'diag' for Hessian computations is not
% supported due to no obvious need.
%
% These Hessians can be used to ultimately compute the Hessian of the
% GP training likelihood (see, for example, exact_inference.m).
%
% The hyperparameters are the same as for covLIN.
%
% See also COVLIN, COVFUNCTIONS.

% Copyright (c) 2015 Roman Garnett.

function result = linear_covariance(theta, x, z, i, ~)

  % call covLIN for everything but Hessian calculation
  if (nargin <= 1)
    result = covLIN;
  elseif (nargin == 2)
    result = covLIN(theta, x);
  elseif (nargin == 3)
    result = covLIN(theta, x, z);
  elseif (nargin == 4)
    result = covLIN(theta, x, z, i);

  % Hessian with respect to \theta_i \theta_j
  else

    % there are no hyperparameters, return GPML error
    result = covLIN(theta, x, z, i);
  end

end