% INFERENCE_WITH_PRIOR inference with hyperparameter prior.
%
% This implements a "meta"-inference method that allows the user to
% use an arbitrary hyperparameter prior p(\theta) with a given
% inference method. This allows MAP rather than MLE inference during
% hyperparameter learning.
%
% Given a training set (X, y), and a GPML inference method and
% observation likelihood, the inference method would normally
% return:
%
%   posterior: a GPML posterior struct containing the approximate
%              GP posterior
%         nlZ: the negative log marginal likelihood:
%
%                -log p(y | X, \theta)
%
%        dnlZ: the partial derivatives of the negative log marginal
%              likelihood with respect to each hyperparameter
%              \theta_i:
%
%                -d log p(y | X, \theta) / d \theta_i
%
% Given a hyperparameter prior p(\theta), this function allows you to
% combine it with an inference method and likelihood to replace the
% negative log marginal likelihood p(y | X, \theta) with the
% negative (unnormalized) log posterior
%
%   -[ log p(y | X, \theta) + log p(\theta) ].
%
% The nlZ and dnlZ values returned reflect the additional
% hyperparameter prior term. This function also supports an extended
% GPML interface allowing the Hessian of the negative unnormalized log
% posterior to also be returned, if desired.
%
% The API of this function is not directly compatible with GPML prior
% to version 3.4. GPML prior to version 3.5 did not support
% constructing a meta inference method in the same way as you would a
% covariance or mean function; that is, you could not use, e.g.,
%
%   inference_method = {@inference_with_prior, inference_method, prior}.
%
% This is now possible. For GPML prior to version 3.5, you may instead
% use the provided add_prior_to_inference_method function to return a
% function handle for use in, e.g., gp.m.
%
% Usage
% -----
%
%   [posterior, nlZ, dnlZ, dalpha, dWinv, HnlZ] = ...
%           inference_with_prior(inference_method, prior, ...
%           hyperparameters, mean_function, covariance_function, ...
%           likelihood, x, y)
%
% Inputs:
%
%      inference_method: a GPML inference method
%                 prior: a function handle to a hyperparameter
%                        prior (see priors.m)
%       hyperparameters: a GPML hyperparameter struct
%         mean_function: a GPML mean function
%   covariance_function: a GPML covariance function
%            likelihood: a GPML likelihood
%                     x: the observation locations (N x D)
%                     y: the observation values (N x 1)
%
% Outputs:
%
%   posterior: a GPML posterior struct
%         nlZ: the negative unnormalized log posterior
%        dnlZ: the gradient of the negative unnormalized log posterior
%      dalpha: the gradient of alpha
%       dWinv: the gradient of diag(W^{-1})
%        HnlZ: the Hessian of the negative unnormalized log posterior
%
% See also PRIORS, INDEPENDENT_PRIOR, EXACT_INFERENCE, HESSIANS.

% Copyright (c) 2014 Roman Garnett.

function [posterior, nlZ, dnlZ, dalpha, dWinv, HnlZ] = ...
      inference_with_prior(inference_method, prior, theta, mean_function, ...
          covariance_function, likelihood, x, y)

  if (nargout <= 1)
    posterior = ...
        inference_method(theta, mean_function, covariance_function, ...
                         likelihood, x, y);
    return;

  elseif (nargout == 2)
    [posterior, nlZ] = ...
        inference_method(theta, mean_function, covariance_function, ...
                         likelihood, x, y);
    nlZ = nlZ - prior(theta);
    return;

  elseif (nargout == 3)
    [prior_lp, prior_dlp] = prior(theta);
    [posterior, nlZ, dnlZ] = ...
        inference_method(theta, mean_function, covariance_function, ...
                         likelihood, x, y);

  elseif (nargout == 4)
    [prior_lp, prior_dlp] = prior(theta);
    [posterior, nlZ, dnlZ, dalpha] = ...
        inference_method(theta, mean_function, covariance_function, ...
                         likelihood, x, y);

  elseif (nargout == 5)
    [prior_lp, prior_dlp] = prior(theta);
    [posterior, nlZ, dnlZ, dalpha, dWinv] = ...
        inference_method(theta, mean_function, covariance_function, ...
                         likelihood, x, y);

  elseif (nargout == 6)
    [prior_lp, prior_dlp, prior_Hlp] = prior(theta);
    [posterior, nlZ, dnlZ, dalpha, dWinv, HnlZ] = ...
        inference_method(theta, mean_function, covariance_function, ...
                         likelihood, x, y);
  end

  nlZ = nlZ - prior_lp;

  % merge gradient structs
  fields = {'cov', 'lik', 'mean'};
  for i = 1:numel(fields)
    field = fields{i};

    if (~isfield(dnlZ, field))
      continue;
    end

    dnlZ.(field) = dnlZ.(field) - prior_dlp.(field);
  end

  % merge Hessian if required
  if (nargout == 6)
    HnlZ.value = HnlZ.value - prior_Hlp.value;
  end

end