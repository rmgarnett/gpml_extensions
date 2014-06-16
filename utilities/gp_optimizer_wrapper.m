% GP_OPTIMIZER_WRAPPER standard interface for optimizing log likelihood.
%
% This is a trivial wrapper that, given a GP prior on a function f:
%
%   p(f | \theta) = GP(f; \mu(x; \theta), K(x, x'; \theta)),
%
% with hyperparameters \theta, along with an observation model
%
%   p(y | f, \theta),
%
% and observations D = (X, y), returns the negative log likelihood
%
%   L(\theta) = -log p(y | X, \theta),
%
% its gradient, and its Hessian using an interface common to many
% generic MATLAB optimization routines.
%
% This allows the use of alternative optimizers when learning
% hyperparameters, such as those in the MATLAB Optimization Toolbox
% (fminunc, fmincon) or Mark Schmidt's minFunc function:
%
%   http://www.di.ens.fr/~mschmidt/Software/minFunc.html
%
% Note
% ----
%
% Access to the third output, HnlZ, the Hessian of the negative log
% likelihood with respect to \theta, requires that the inference
% method support an extended GPML API; see hessians.m for more
% information.
%
% Usage
% -----
%
%   [nlZ, dnlZ, HnlZ] = gp_optimizer_wrapper(hyperparameter_values, ...
%           prototype_hyperparameters, inference_method, mean_function, ...
%           covariance_function, likelihood, x, y)
%
% Inputs:
%
%       hyperparameter_values: a (row or column) vector specifying all
%                              hyperparameters, in the order returned by
%                              unwrap(prototype_hyperparameters)
%   prototype_hyperparameters: any GPML hyperparameter struct
%                              compatible specified with the GP model;
%                              that is, the lengths of the vectors
%
%                                prototype_hyperparameters.mean
%                                prototype_hyperparameters.cov
%                                prototype_hyperparameters.lik
%
%                              need to be correct (the actual values
%                              will not be used).
%            inference_method: a GPML inference method
%               mean_function: a GPML mean function
%         covariance_function: a GPML covariance function
%                  likelihood: a GPML likelihood
%                           x: training observation locations (N x D)
%                           y: training observation values (N x 1)
%
% Outputs:
%
%    nlZ: the negative log likelihood, -log p(y | X, \theta)
%   dnlZ: the gradient of the negative log likelihood with respect to
%         \theta, returned as a (#\theta x 1) vector in the order
%         specified by unwrap(prototype_hyperparameters).
%   HnlZ: the Hessian of the negative log likelihood with respect to
%         \theta, returned as a (#\theta x #\theta) matrix in the
%         order specified by unwrap(prototype_hyperparameters).
%
% See also HESSIANS.

% Copyright (c) 2014 Roman Garnett.

function [nlZ, dnlZ, HnlZ] = gp_optimizer_wrapper(hyperparameter_values, ...
          prototype_hyperparameters, inference_method, mean_function, ...
          covariance_function, likelihood, x, y)

  hyperparameters = rewrap(prototype_hyperparameters, ...
                           hyperparameter_values(:));

  if (nargout <= 1)
    [~, nlZ] = inference_method(hyperparameters, mean_function, ...
            covariance_function, likelihood, x, y);

    return;

  elseif (nargout == 2)
    [~, nlZ, dnlZ] = inference_method(hyperparameters, mean_function, ...
            covariance_function, likelihood, x, y);

  elseif (nargout > 2)
    [~, nlZ, dnlZ, HnlZ] = inference_method(hyperparameters, mean_function, ...
            covariance_function, likelihood, x, y);

    HnlZ = HnlZ.H;
  end

  dnlZ = unwrap(dnlZ);

end