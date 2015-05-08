% ARD_SQDEXP_COVARIANCE squared exponential covariance with ARD.
%
% This provides a GPML-compatible covariance function implementing the
% squared exponential covariance with automatic relevance
% determination (ARD). This can be used as a drop-in replacement for
% covSEard.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of K with respect to any pair of
% hyperparameters. The syntax is:
%
%   dK2_didj = ard_sqdexp_covariance(theta, x, z, i, j)
%
% where dK2_didj is \partial^2 K / \partial \theta_i \partial \theta_j,
% and the Hessian is evalauted at K(x, z). As in the derivative API,
% if z is empty, then the Hessian is evaluated at K(x, x).  Note that
% the option of setting z = 'diag' for Hessian computations is not
% supported due to no obvious need.
%
% These Hessians can be used to ultimately compute the Hessian of the
% GP training likelihood (see, for example, exact_inference.m).
%
% The hyperparameters are the same as for covSEard.
%
% See also COVSEARD, COVFUNCTIONS.

% Copyright (c) 2013--2015 Roman Garnett.

function result = ard_sqdexp_covariance(theta, x, z, i, j)

  % used during gradient and Hessian calculations to avoid constant recomputation
  persistent K;

  % call covSEard for everything but Hessian calculation
  if (nargin <= 1)
    result = covSEard;
  elseif (nargin == 2)
    result = covSEard(theta, x);
  elseif (nargin == 3)
    result = covSEard(theta, x, z);
  elseif (nargin == 4)
    result = covSEard(theta, x, z, i);

  % Hessian with respect to \theta_i \theta_j
  else

    % ensure i <= j by exploiting symmetry
    if (i > j)
      result = ard_sqdexp_covariance(theta, x, z, j, i);
      return;
    end

    % Hessians involving the log output scale
    if (j == numel(theta))
      result = 2 * covSEard(theta, x, z, i);
      return;
    end

    % precompute and store K for repeated reuse when the first
    % Hessian is requested
    if ((i == 1) && (j == 1))
      K = covSEard(theta, x, z);
    end

    % avoid if (isempty(z)) checks
    if (isempty(z))
      z = x;
    end

    ell_i = exp(-theta(i));
    ell_j = exp(-theta(j));

    first_factor = ...
        bsxfun(@minus, x(:, i) * ell_i, (z(:, i) * ell_i)').^2;
    second_factor = ...
        bsxfun(@minus, x(:, j) * ell_j, (z(:, j) * ell_j)').^2 - 2 * (i == j);

    result = first_factor .* second_factor .* K;
  end

end