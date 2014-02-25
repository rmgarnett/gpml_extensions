% GP_POSTERIOR (unnormalized) negative log training posterior of a GP.
%
% Given a training set (X, y), this file calculates the unnormalized
% negative log training posterior of a GP:
%
%   -(\log p(y | X, \theta) + \log p(\theta)),
%
% as well as its gradient and Hessian, if desired. This is made to be
% compatible with GPML and allows MAP rather than MLE inference during
% the hyperparameter learning. It supports a similar API to gp.m in
% training mode.
%
% Usage
% -----
%
%   [nlZ, dnlZ, HnlZ] = gp_posterior(hyperparameters, prior, inference_method, ...
%           mean_function, covariance_function, likelihood, x, y)
%
% Inputs:
%
%       hyperparameters: a GPML hyperparameter struct
%                 prior: a function handle to a hyperparameter
%                        prior (see below)
%      inference_method: a GPML inference method
%         mean_function: a GPML mean function
%   covariance_function: a GPML covariance function
%            likelihood: a GPML likelihood
%                     x: the observation locations (N x D)
%                     y: the associated observations (N x 1)
%
% Outputs:
%
%         nlZ: the negative unormalized log posterior
%        dnlZ: the gradient of the negative unnormalized log posterior
%        HnlZ: the Hessian of the negative unnormalized log posterior
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
% Hyperprior specification
% ------------------------
%
% The hyperparameter, prior, must be a function handle supporting the
% interface:
%
%   [nlZ, dnlZ, HnlZ] = prior(hyperparameters)
%
% where:
%
%   hyperparameters: a GPML-compatible hyperparameters struct
%               nlZ: the negative log prior
%              dnlZ: the gradient of the negative log prior
%              HnlZ: the Hessian of the negative log prior
%
% The gradient dnlZ should have the same format as in GPML, namely it
% contains the derivatives of the negative log prior with respect to
% the hyperparameters in a struct of the same layout as the
% hyperparameters. If Hessians are desired, HnlZ should have the
% format as described in hessians.m.
%
% See also PRIORS, INDEPENDENT_PRIOR, EXACT_INFERENCE, HESSIANS.

% Copyright (c) 2013--2014 Roman Garnett

function [nlZ, dnlZ, HnlZ] = gp_posterior(hyperparameters, prior, ...
          inference_method, mean_function, covariance_function, ...
          likelihood, x, y)

  % only nlZ requested
  if (nargout <= 1)
    [~, nlZ] = ...
        inference_method(hyperparameters, mean_function, covariance_function, ...
                         likelihood, x, y);
    nlZ = nlZ + prior(hyperparameters);
    return;
  end

  if (nargout == 2)
    [prior_nlZ, prior_dnlZ] = prior(hyperparameters);
    [~, nlZ, dnlZ] = ...
        inference_method(hyperparameters, mean_function, covariance_function, ...
                         likelihood, x, y);
  elseif (nargout == 3)
    [prior_nlZ, prior_dnlZ, prior_HnlZ] = prior(hyperparameters);
    [~, nlZ, dnlZ, HnlZ] = ...
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
  if (nargout == 3)
    HnlZ.H = HnlZ.H + prior_HnlZ.H;
  end

end