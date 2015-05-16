% COVARIANCE_PRODUCT drop-in replacement for covProd with Hessian support
%
% This provides a GPML-compatible meta covariance function implementing
% a product of covariance functions:
%
%   K(x, x') = \prod_i K_i(x, x').
%
% This can be used as a drop-in replacement for covProd.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of K with respect to any pair of
% hyperparameters. The syntax is:
%
%   dK2_didj = covariance_product(Ks, theta, x, z, i, j)
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
% The hyperparameters are the same as for covProd.
%
% See also COVPROD, COVFUNCTIONS.

% Copyright (c) 2015 Roman Garnett.

function result = covariance_product(Ks, theta, x, z, i, j)

  % call covProd for everything but Hessian calculation
  if (nargin == 0)
    result = covProd;
  elseif (nargin <= 2)
    result = covProd(Ks);
  elseif (nargin == 3)
    result = covProd(Ks, theta, x);
  elseif (nargin == 4)
    result = covProd(Ks, theta, x, z);
  elseif (nargin == 5)
    result = covProd(Ks, theta, x, z, i);

  % Hessian with respect to \theta_i \theta_j
  else

    % ensure i <= j by exploiting symmetry
    if (i > j)
      result = covariance_product(Ks, theta, x, z, j, i);
      return;
    end

    % needed to compute number of hyperparameters
    D = size(x, 2);

    ind = zeros(eval(covProd(Ks)), 1);

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

    if (isempty(z))
      result = ones(size(x, 1));
    else
      result = ones(size(x, 1), size(z, 1));
    end

    % accumulate product containing Hessian
    for k = 1:numel(ks)
      K = Ks(k);
      % expand cell
      if (iscell(K{:}))
        K = K{:};
      end

      offset = (nnz(ind < k));

      % both parameters are in this covariance;
      % contribution is d^2 K_k(x, z) / d\theta_i / d\theta_j
      if ((k == ind(i)) && (ind(i) == ind(j)))
        result = result .* ...
                 feval(K{:}, theta(ind == k), x, z, i - offset, j - offset);

      % exactly one parameter (i) is in this covariance;
      % contribution is dK_k(x, z) / d\theta_i
      elseif (k == ind(i))
        result = result .* feval(K{:}, theta(ind == k), x, z, i - offset);

      % exactly one parameter (j) is in this covariance;
      % contribution is dK_k(x, z) / d\theta_j
      elseif (k == ind(j))
        result = result .* feval(K{:}, theta(ind == k), x, z, j - offset);

      % no parameters occur in this covariance;
      % contribution is K_k(x, z)
      else
        result = result .* feval(K{:}, theta(ind == k), x, z);
      end
    end

  end

end