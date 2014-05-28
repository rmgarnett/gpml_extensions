% LAPLACE_PRIOR Laplace hyperprior given mean and diversity.
%
% This file implements a Laplace prior for a hyperparameter given a
% mean \mu and diversity b:
%
%   p(\theta) = Laplace(\theta; \mu, b).
%
% This file supports both calculating the negative log prior and its
% derivatives as well as drawing a sample from the prior. The latter
% is accomplished by not passing in a value for the hyperparameter.
% See priors.m for more information.
%
% Usage (prior mode)
% ------------------
%
%   [nlZ, dnlZ, d2nlZ] = laplace_prior(mean, diversity, theta)
%
% Inputs:
%
%        mean: the prior mean
%   diversity: the prior "diversity" parameter b
%       theta: the value of the hyperparameter
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
%   sample = laplace_prior(mean, diversity)
%
% Inputs:
%
%        mean: the prior mean
%   diversity: the prior "diversity" parameter b
%
% Output:
%
%   sample: a sample drawn from the prior
%
% See also: PRIORS.

% Copyright (c) 2014 Roman Garnett.

function [result, dnlZ, d2nlZ] = laplace_prior(mean, diversity, theta)

  % draw sample
  if (nargin < 3)
    u = rand - 0.5;
    result = mean - diversity * sign(u) * log(1 - 2 * abs(u));
    return;
  end

  log_2 = 0.693147180559945;

  % negative log likelihood
  result = log_2 + log(diversity) + abs(theta - mean) / diversity;

  % first derivative of negative log likelihood
  if (nargout >= 2)
    if (theta == mean)
      dnlZ = nan;
    else
      dnlZ = sign(theta - mean) / diversity;
    end
  end

  % second derivative of negative log likelihood
  if (nargout >= 3)
    if (theta == mean)
      d2nlZ = nan;
    else
      d2nlZ = 0;
    end
  end

end
