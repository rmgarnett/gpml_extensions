% UNIFORM_PRIOR uniform hyperprior on given interval.
%
% This file implements a uniform prior for a hyperparameter on a given
% interval [l, u]:
%
%   p(\theta) = U(\theta; l, u).
%
% This file supports both calculating the negative log prior and its
% derivatives as well as drawing a sample from the prior. The latter
% is accomplished by not passing in a value for the hyperparameter.
% See priors.m for more information.
%
% Usage (prior mode)
% ------------------
%
%   [nlZ, dnlZ, d2nlZ] = uniform_prior(lower_bound, upper_bound, theta)
%
% Inputs:
%
%   lower_bound: the lower bound of the interval
%   upper_bound: the upper bound of the interval
%         theta: the value of the hyperparameter
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
%   sample = uniform_prior(lower_bound, upper_bound)
%
% Inputs:
%
%   lower_bound: the lower bound of the interval
%   upper_bound: the upper bound of the interval
%
% Output:
%
%   sample: a sample drawn from the prior
%
% See also: PRIORS.

% Copyright (c) 2014 Roman Garnett.

function [result, dnlZ, d2nlZ] = uniform_prior(lower_bound, upper_bound, theta)

  % draw sample
  if (nargin < 3)
    result = lower_bound + rand * (upper_bound - lower_bound);
    return;
  end

  % negative log likelihood
  if ((lower_bound < theta) && (theta < upper_bound))
    result = log(upper_bound - lower_bound);
  else
    result = -inf;
  end

  % first derivative of negative log likelihood
  if (nargout >= 2)
    dnlZ = 0;
  end

  % second derivative of negative log likelihood
  if (nargout >= 3)
    d2nlZ = 0;
  end

end
