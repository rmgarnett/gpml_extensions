% FACTOR_SQDEXP_COVARIANCE squared exponential "factor analysis covariance."
%
% This provides a GPML-compatible covariance function implementing the
% "factor analysis covariance:"
%
%   K(x, y) = \sigma^2 K_SE(xR', yR'),
%
% where x and y are D-dimensional row vectors, R is a d x D linear
% embedding matrix, \sigma is the output scale, and K_SE is the
% squared exponential covariance in R^d with unit length scale.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of K with respect to any pair of
% hyperparameters. The syntax is:
%
%   dK2_didj = ...
%      factor_sqdexp_covariance(hyperparameters, x, z, i, j)
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
% The hyperparameters are:
%
%   hyperparameters = [ R(:)
%                       log(\sigma) ],
%
% where R(:) is the vectorized R matrix (i.e., R in column-major
% order) and \sigma is the output scale.
%
% See also COVFUNCTIONS.

% Copyright (c) 2013--2014 Roman Garnett.

function result = factor_sqdexp_covariance(d, hyperparameters, x, z, i, j)

  % used during gradient and Hessian calculations to avoid constant recomputation
  persistent K;

  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'd muist be specified!');
  end

  % report number of hyperparameters
  if (nargin <= 2)
    result = ['(D * ' num2str(d) ' + 1)'];
    return;
  end

  D = size(x, 2);

  % extract embedding matrix
  R = reshape(hyperparameters(1:(end - 1)), [d, D])';

  % create empty z if it does not exist
  if (nargin == 3)
    z = [];
  end

  % embed inputs
  x_transformed = x * R;
  if (isnumeric(z) && ~isempty(z))
    z_transformed = z * R;
  % don't modify the 'diag' string or empty arrays
  else
    z_transformed = z;
  end

  % covariance, call covSEiso on the embedded points
  if (nargin <= 4)
    result = covSEiso([0; hyperparameters(end)], x_transformed, z_transformed);
    return;
  end

  % avoid silly if (isempty(z)) checks later
  if (isempty(z))
    z = x;
    z_transformed = x_transformed;
  end

  % precompute and store K for repeated reuse when the first
  % gradient or Hessian is requested
  if (((nargin == 5) && (i == 1)) || ...
      ((nargin == 6) && (i == 1) && (j == 1)))
    K = covSEiso([0; hyperparameters(end)], x_transformed, z_transformed);
  end

  % derivative with respect to \theta_i
  if (nargin == 5)

    % derivative wrt entry of R?
    embedding_derivative = (i < numel(hyperparameters));

    % diagonal derivatives
    if (~isnumeric(z))
      % diagonal deriviatives wrt R are all zero
      if (embedding_derivative)
        result = zeros(size(x, 1), 1);
      % log output scale
      else
        result = 2 * covSEiso([0; hyperparameters(end)], x_transformed, 'diag');
      end
      return;
    end

    if (embedding_derivative)
      row    = 1 + mod(i - 1, d);
      column = 1 + floor((i - 1) / d);

      factor = bsxfun(@minus, x(:, column), z(:, column)') .* ...
               bsxfun(@minus, x_transformed(:, row), z_transformed( :, row)');
      result = -factor .* K;
    % log output scale
    else
      result = K + K;
    end

  % Hessian with respect to \theta_i \theta_j
  else

    % ensure i <= j by exploiting symmetry
    if (i > j)
      result = factor_sqdexp_covariance(d, hyperparameters, x, z, j, i);
      return;
    end

    % Hessians involving the log output scale
    if (j == numel(hyperparameters))
      result = 2 * factor_sqdexp_covariance(d, hyperparameters, x, z, i);
      return;
    end

    first_row     = 1 + mod(i - 1, d);
    first_column  = 1 + floor((i - 1) / d);
    second_row    = 1 + mod(j - 1, d);
    second_column = 1 + floor((j - 1) / d);

    % Hessians involving only the entries of R
    untransformed_factor = ...
        bsxfun(@minus, x(:, first_column),  z(:, first_column)') .* ...
        bsxfun(@minus, x(:, second_column), z(:, second_column)');
    transformed_factor = ...
        bsxfun(@minus, x_transformed(:, first_row),  z_transformed(:, first_row)') .* ...
        bsxfun(@minus, x_transformed(:, second_row), z_transformed(:, second_row)') - ...
        (first_row == second_row);

    result = untransformed_factor .* transformed_factor .* K;
  end

end