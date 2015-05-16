% COVARIANCE_SUM drop-in replacement for covSum with Hessian support
%
% This provides a GPML-compatible meta covariance function implementing
% a sum of covariance functions:
%
%   K(x, x') = \sum_i K_i(x, x').
%
% This can be used as a drop-in replacement for covSum.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of K with respect to any pair of
% hyperparameters. The syntax is:
%
%   dK2_didj = covariance_sum(Ks, theta, x, z, i, j)
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
% The hyperparameters are the same as for covSum.
%
% See also COVSUM, COVFUNCTIONS.

% Copyright (c) 2015 Roman Garnett.

function result = covariance_sum(Ks, theta, x, z, i, j)

  % call covSum for everything but Hessian calculation
  if (nargin == 0)
    result = covSum;
  elseif (nargin <= 2)
    result = covSum(Ks);
  elseif (nargin == 3)
    result = covSum(Ks, theta, x);
  elseif (nargin == 4)
    result = covSum(Ks, theta, x, z);
  elseif (nargin == 5)
    result = covSum(Ks, theta, x, z, i);

  % Hessian with respect to \theta_i \theta_j
  else

    % ensure i <= j by exploiting symmetry
    if (i > j)
      result = covariance_sum(Ks, theta, x, z, j, i);
      return;
    end

    % needed to compute number of hyperparameters
    D = size(x, 2);

    ind = zeros(eval(covSum(Ks)), 1);

    % build mapping from hyperparameter index to covariance index
    offset = 0;
    for k = 1:numel(Ks)
      K = Ks(k);
      % expand cell
      if (iscell(K{:}))
        K = K{:};
      end

      this_num_hyperparameters = eval(feval(K{:}));

      ind((offset + 1):(offset + this_num_hyperparameters)) = k;
      offset = offset + this_num_hyperparameters;
    end

    % if pair (i, j) correspond to the same covariance, we compute the
    % appropriate Hessian and return
    if (ind(i) == ind(j))
      K = Ks(ind(i));
      % expand cell
      if (iscell(K{:}))
        K = K{:};
      end

      offset = nnz(ind < i);
      result = feval(K{:}, theta(ind == ind(i)), x, z, i - offset, j - offset);
    else
      % entries corresponding to pairs in different covariances are zero
      if (isempty(z))
        result = zeros(size(x, 1));
      else
        result = zeros(size(x, 1), size(z, 1));
      end
    end
  end

end