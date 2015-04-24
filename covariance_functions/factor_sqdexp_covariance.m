% FACTOR_SQDEXP_COVARIANCE squared exponential factor analysis covariance.
%
% This provides a GPML-compatible covariance function implementing the
% squared exponential factor analysis covariance. This can be used as
% a drop-in replacement for covSEfact.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of K with respect to any pair of
% hyperparameters. The syntax is:
%
%   dK2_didj = factor_sqdexp_covariance(theta, x, z, i, j)
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
% The hyperparameters are the same as for covSEfact.
%
% See also COVSEFACT, COVFUNCTIONS.

% Copyright (c) 2013--2015 Roman Garnett.

function result = factor_sqdexp_covariance(d, theta, x, z, i, j)

  % used during Hessian calculations to avoid constant recomputation
  persistent K;

  % call covSEfact for everything but Hessian calculation
  if (nargin == 0)
    result = covSEfact();
  elseif (nargin <= 2)
    result = covSEfact(d);
  elseif (nargin == 3)
    result = covSEfact(d, theta, x);
  elseif (nargin == 4)
    result = covSEfact(d, theta, x, z);
  elseif (nargin == 5)
    result = covSEfact(d, theta, x, z, i);

  % Hessian with respect to \theta_i \theta_j
  else

    % ensure i <= j by exploiting symmetry
    if (i > j)
      result = factor_sqdexp_covariance(d, theta, x, z, j, i);
      return;
    end

    % Hessians involving the log output scale
    if (j == numel(theta))
      result = 2 * covSEfact(d, theta, x, z, i);
      return;
    end

    % precompute and store K for repeated reuse when the first
    % Hessian is requested
    if ((i == 1) && (j == 1))
      K = covSEfact(d, theta, x, z);
    end

    D = size(x, 2);

    % extract L
    L = zeros(d, D);
    L_entries = theta(1:(end - 1));
    L(triu(true(d, D))) = L_entries(:);
    if (d == 1)
      L_diag = L(1);
    else
      L_diag = diag(L);
    end
    L(1:(d + 1):d^2) = exp(L_diag);

    % transform inputs
    Lx = x * L';

    % avoid if (isempty(z)) checks
    if (isempty(z))
       z =  x;
      Lz = Lx;
    else
      Lz = z * L';
    end

    % make lookup matrix
    ind = zeros(size(L));
    ind(triu(true(d, D))) = 1:(numel(theta) - 1);

    [i_row, i_col] = ind2sub([d, D], find(ind == i));
    [j_row, j_col] = ind2sub([d, D], find(ind == j));

    exp_factor = 1;
    if (i_row == i_col)
     exp_factor = exp_factor * L(i_row, i_col);
    end
    if (j_row == j_col)
     exp_factor = exp_factor * L(j_row, j_col);
    end

    untransformed_factor = ...
        bsxfun(@minus, x(:, i_col), z(:, i_col)') .* ...
        bsxfun(@minus, x(:, j_col), z(:, j_col)');

    transformed_factor = ...
        bsxfun(@minus, Lx(:, i_row), Lz(:, i_row)') .* ...
        bsxfun(@minus, Lx(:, j_row), Lz(:, j_row)') -  ...
        (i_row == j_row);

    result = untransformed_factor .* transformed_factor .* exp_factor .* K;

    % diagonal entries need correction due to exp() transformation
    if (all([i_row, i_col, j_row, j_col] == i_row))
      result = result + covSEfact(d, theta, x, z, i);
    end

  end

end