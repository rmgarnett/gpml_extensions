% MASK_COVARIANCE covMask replacement with Hessian support
%
% This provides a GPML-compatible meta covariance function
% implementing the a masked covariance (where a covariance is only
% evaluatd on a subset of features). This can be used as a drop-in
% replacement for covMask.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of K with respect to any pair of
% hyperparameters. The syntax is:
%
%   dK2_didj = mask_covariance(K, theta, x, z, i, j)
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
% The hyperparameters are the same as for covMask.
%
% See also COVMASK, COVFUNCTIONS.

% Copyright (c) 2015 Roman Garnett.

function result = mask_covariance(K, theta, x, z, i, j)

  % call covMask for everything but Hessian calculation
  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'covariance input K is required!');
  elseif (nargin <= 2)
    result = covMask(K);
  elseif (nargin == 3)
    result = covMask(K, theta, x);
  elseif (nargin == 4)
    result = covMask(K, theta, x, z);
  elseif (nargin == 5)
    result = covMask(K, theta, x, z, i);

  % Hessian with respect to \theta_i \theta_j
  else

    mask = fix(K{1}(:));
    K = K(2);
    % expand cell
    if (iscell(K{:}))
      K = K{:};
    end

    if (isempty(z))
      result = feval(K{:}, theta, x(:, mask), [],         i, j);
    else
      result = feval(K{:}, theta, x(:, mask), z(:, mask), i, j);
    end
  end

end