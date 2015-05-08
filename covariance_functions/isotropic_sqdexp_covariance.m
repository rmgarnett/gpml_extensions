% ISOTROPIC_SQDEXP_COVARIANCE isotropic squared exponential covariance.
%
% This provides a GPML-compatible covariance function implementing the
% isotropic squared exponential covariance. This can be used as a
% drop-in replacement for covSEiso.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of K with respect to any pair of
% hyperparameters. The syntax is:
%
%   dK2_didj = isotropic_sqdexp_covariance(theta, x, z, i, j)
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
% The hyperparameters are the same as for covSEiso.
%
% See also COVSEISO, COVFUNCTIONS.

% Copyright (c) 2014--2015 Roman Garnett.

function result = isotropic_sqdexp_covariance(theta, x, z, i, j)

  % call covSEiso for everything but Hessian calculation
  if (nargin <= 1)
    result = covSEiso;
  elseif (nargin == 2)
    result = covSEiso(theta, x);
  elseif (nargin == 3)
    result = covSEiso(theta, x, z);
  elseif (nargin == 4)
    result = covSEiso(theta, x, z, i);

  % Hessian with respect to \theta_i \theta_j
  else

    % ensure i <= j by exploiting symmetry
    if (i > j)
      result = isotropic_sqdexp_covariance(theta, x, z, j, i);
      return;
    end

    % Hessians involving the log output scale
    if (j == 2)
      result = 2 * covSEiso(theta, x, z, i);
      return;
    end

    K = covSEiso(theta, x, z, 1);

    if (isempty(z))
      z = x;
    end

    ell = exp(-theta(1));

    factor = sq_dist(x' * ell, z' * ell);

    result = (factor - 2) .* K;
  end

end