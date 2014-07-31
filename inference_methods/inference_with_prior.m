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
% The API of this function is not directly compatible with GPML. GPML
% does not support constructing a meta inference method in the same
% way as you would a covariance or mean function; that is, you cannot
% use, e.g.,
%
%   inference_method = {@inference_with_prior, inference_method, prior}.
%
% Instead, you may use the provided add_prior_to_inference_method
% function to return a function handle for use in, e.g., gp.m.
%
% Usage
% -----
%
%   [posterior, nlZ, dnlZ, HnlZ] = ...
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
%        dnlZ: the gradient negative of the unnormalized log posterior
%        HnlZ: the Hessian negative of the unnormalized log posterior
%
% Notes on Hessian
% ----------------
%
% To evaluate the Hessian, the supplied inference method must support
% an extended GPML syntax:
%
%   [posterior, nlZ, dnlZ, HnlZ] = ...
%       inference_method(hyperparameters, mean_function, ...
%                        covariance_function, likelihood, x, y);
%
% where the fourth output is the Hessian of the negative log
% likelihood. See hessians.m for more information regarding the
% Hessian struct HnlZ.
%
% See exact_inference.m for an example inference method supporting
% this extended API. exact_inference may be used as a drop-in
% replacement for infExact.
%
% See also PRIORS, INDEPENDENT_PRIOR, EXACT_INFERENCE, HESSIANS.

% Copyright (c) 2014 Roman Garnett.

function [posterior, nlZ, dnlZ, HnlZ, varargout] = ...
      inference_with_prior(inference_method, prior, hyperparameters, ...
                           mean_function, covariance_function, ...
                           likelihood, x, y)

  % only posterior requested
  if (nargout <= 1)
    posterior = ...
        inference_method(hyperparameters, mean_function, covariance_function, ...
                         likelihood, x, y);
    return;

  % posterior and nlZ requested
  elseif (nargout == 2)
    [posterior, nlZ] = ...
        inference_method(hyperparameters, mean_function, covariance_function, ...
                         likelihood, x, y);
    nlZ = nlZ + prior(hyperparameters);
    return;

  % posterior, nlZ, and dnlZ requested
  elseif (nargout == 3)
    [prior_nlZ, prior_dnlZ] = prior(hyperparameters);
    [posterior, nlZ, dnlZ] = ...
        inference_method(hyperparameters, mean_function, covariance_function, ...
                         likelihood, x, y);

  % posterior, nlZ, dnlZ, and HnlZ requested
  elseif (nargout >= 4)
    [prior_nlZ, prior_dnlZ, prior_HnlZ] = prior(hyperparameters);
    [posterior, nlZ, dnlZ, HnlZ, varargout{1:(nargout - 4)}] = ...
        inference_method(hyperparameters, mean_function, covariance_function, ...
                         likelihood, x, y);
  end

  nlZ = nlZ + prior_nlZ;

  % merge gradient structs
  fields = fieldnames(hyperparameters);
  for i = 1:numel(fields)
    field = fields{i};

    dnlZ.(field) = dnlZ.(field) + prior_dnlZ.(field);
  end

  % merge Hessian if required
  if (nargout == 4)
    HnlZ.H = HnlZ.H + prior_HnlZ.H;
  end

end