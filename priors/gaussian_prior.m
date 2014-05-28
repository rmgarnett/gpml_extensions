% GAUSSIAN_PRIOR Gaussian hyperprior with given mean and variance.
%
% This file implements a Gaussian prior for a hyperparameter with a
% given mean \mu and variance \sigma^2:
%
%   p(\theta) = N(\theta; \mu, \sigma^2)
%
% This file supports both calculating the negative log prior and its
% derivatives as well as drawing a sample from the prior. The latter
% is accomplished by not passing in a value for the hyperparameter.
% See priors.m for more information.
%
% Usage (prior mode)
% ------------------
%
%   [nlZ, dnlZ, d2nlZ] = gaussian_prior(mean, variance, theta)
%
% Inputs:
%
%       mean: the prior mean
%   variance: the prior variance
%      theta: the value of the hyperparameter
%
% Outputs:
%
%     nlZ: the negative log prior evaluated at theta
%    dnlZ: the derivative of the negative log prior evaluated at theta
%   d2nlZ: the second derivative of the negative log prior
%          evaluated at theta
%
% Usage (sample mode)
% -------------------
%
%   sample = gaussian_prior(mean, variance)
%
% Inputs:
%
%       mean: the prior mean
%   variance: the prior variance
%
% Output:
%
%   sample: a sample drawn from the prior
%
% See also: PRIORS.

% Copyright (c) 2014 Roman Garnett.

function [result, dnlZ, d2nlZ] = gaussian_prior(mean, variance, theta)

  % draw sample
  if (nargin < 3)
    result = mean + randn * sqrt(variance);
    return;
  end

  log_2pi = 1.837877066409345;

  % negative log likelihood
  result = 0.5 * ((theta - mean)^2 / variance + log(variance) + log_2pi);

  % first derivative of negative log likelihood
  if (nargout >= 2)
    dnlZ = (theta - mean) / variance;
  end

  % second derivative of negative log likelihood
  if (nargout >= 3)
    d2nlZ = 1 / variance;
  end

end
