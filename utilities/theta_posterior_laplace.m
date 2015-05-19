% THETA_POSTERIOR_LAPLACE Laplace approximation to hyperparameter posterior
%
% This function makes a Laplace approximation to the hyperparameter
% posterior p(\theta | D):
%
%   p(\theta | D) ~ N(\theta; \hat{\theta}, \Sigma)
%
% as well as a Laplace approximation to the log model evidence
%
%   log p(y | X) = log \int p(y | X, \theta) p(\theta | D) d\theta
%
% The Laplace approximation works by taking a second-order Taylor
% expansion to the (unnormalized) log posterior
%
%   log p(\theta | D) = log p(y | X, \theta) + log p(\theta)
%
% around the mode
%
%   \hat{\theta} = \argmax_\theta log(\theta | D).
%
% Note that this function does not perform the maximization over
% \theta but rather assumes that the given hyperparameters represent
% the MLE/MAP point.
%
% Usage
% -----
%
%   [mu, Sigma_inv, log_evidence] = ...
%        theta_laplace_approximation(map_theta, inference_method, ...
%            mean_function, covariance_function, likelihood, x, y)
%
% Inputs
% ------
%
%             map_theta: a GPML hyperparameter struct containing a
%                        local optimum of the log posterior p(\theta | D)
%      inference_method: a GPML inference method
%         mean_function: a GPML mean function
%   covariance_function: a GPML covariance function
%            likelihood: a GPML likelihood
%                     x: training observation locations (n x D)
%                     y: training observation values (n x 1) or
%                        GPML posterior struct
%
% Outputs
% -------
%
%             mu: the approximate posterior mean E[\theta | D] (under
%                 laplace approxmation, this is equal to \hat{\theta}
%      Sigma_inv: the approximate posterior precision matrix,
%                 inv[cov[\theta | D]]
%   log_evidence: the approximate log evidence, \log p(y | X)

% Copyright (c) 2015 Roman Garnett.

function [mu, Sigma_inv, log_evidence] = ...
      theta_posterior_laplace(map_theta, inference_method, ...
          mean_function, covariance_function, likelihood, x, y)

  % log(2\pi) / 2
  half_log_2pi = 0.918938533204673;

  % perform initial argument checks/transformations
  [map_theta, inference_method, mean_function, covariance_function, ...
   likelihood] = check_arguments(map_theta, inference_method, mean_function, ...
          covariance_function, likelihood, x);

  [~, nlZ, ~, ~, ~, HnlZ] = feval(inference_method{:}, map_theta, ...
          mean_function, covariance_function, likelihood, x, y);

  mu        = map_theta;
  Sigma_inv = HnlZ;

  d = size(Sigma_inv.value, 1);
  [L, p] = chol(HnlZ.value);
  if (p ~= 0)
    error('gpml_extensions:theta_laplace_approximation', ...
          'Hessian is not positive definite!');
  end

  % Laplace approximation to model evidence is
  %
  %   log Z ~ L(\hat{\theta}) + (d / 2) log(2\pi) - (1 / 2) log det H,
  %
  % where d is the dimension of \theta and H is the negative Hessian
  % of L evaluated at \hat{\theta}

  log_evidence = -nlZ + d * half_log_2pi - sum(log(diag(L)));

end