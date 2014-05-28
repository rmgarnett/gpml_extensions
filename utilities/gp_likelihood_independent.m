% GP_LIKELIHOOD_INDEPENDENT joint likelihood of independent GP samples
%
% This function computes the negative log likelihood and its
% derivatives given sets of observations that are supposed to have
% been drawn independently from the same Gaussian process prior.
%
% Namely, assume we have N datasets
%
%   {(X_i, y_i)} (1 \leq i \leq N)
%
% we assume these sets of observations are generated indpendently
% from a common GP prior:
%
%   p(y_i | f_i, X_i, \theta) = g(f_i(X_i));
%   p(f_i | \theta)           = GP(f_i; \mu(\theta), K(\theta)),
%
% where g is the observation likelihood and the hyperparameters
% \theta are shared across the f_i.
%
% The interface of this function is almost identical to that of gp.m
% in training mode, except that the sets of observations are provided
% as cell arrays.
%
% Usage:
%
% [nlZ, dnlZ] = gp_likelihood_independent(hyperparameters, inference_method, ...
%         mean_function, covariance_function, likelihood, xs, ys)
%
% Inputs:
%
%       hyperparameters: a GPML hyperparameter struct
%      inference_method: a GPML inference method
%         mean_function: a GPML mean function
%   covariance_function: a GPML covariance function
%            likelihood: a GPML likelihood
%                    xs: a cell array of the observation locations
%                        (each is N_i x D)
%                    ys: a cell array of the observation values
%                        (each is N_i x 1)
%
% Outputs:
%
%    nlZ: the negative log likelihood of the given data
%   dnlZ: the gradient of the negative log likelihood with
%         respect to the hyperparmeters
%
% See also GP.

% Copyright (c) 2013--2014 Roman Garnett.

function [nlZ, dnlZ] = gp_likelihood_independent(hyperparameters, ...
          inference_method, mean_function, covariance_function, ...
          likelihood, xs, ys)

  num_samples = numel(xs);

  % initialize nlZ and, optionally, dnlZ struct
  nlZ = 0;
  if (nargout == 2)
    dnlZ = rewrap(hyperparameters, 0 * unwrap(hyperparameters));
  end

  for i = 1:num_samples
    if (nargout == 1)
      this_nlZ = ...
          gp(hyperparameters, inference_method, mean_function, ...
             covariance_function, likelihood, xs{i}, ys{i});
    else
      [this_nlZ, this_dnlZ] = ...
          gp(hyperparameters, inference_method, mean_function, ...
             covariance_function, likelihood, xs{i}, ys{i});
    end

    % accumulate likelihoods and derivatives
    nlZ = nlZ + this_nlZ;
    if (nargout == 2)
      dnlZ = rewrap(dnlZ, unwrap(dnlZ) + unwrap(this_dnlZ));
    end
  end

end