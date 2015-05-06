% CONSTANT_PRIOR constant (improper) hyperparameter prior.
%
% This file implements an (improper) constant prior for a
% hyperparameter:
%
%   p(\theta) = 1
%
% This file supports both calculating the log prior and its
% derivatives as well as drawing a sample from the prior. The latter
% is accomplished by not passing in a value for the hyperparameter.
% See priors.m for more information. Notice that this particular prior
% cannot be sampled from; therefore, a default value must be provided,
% which is always returned when a sample is requested.
%
% Usage (prior mode)
% ------------------
%
%   [lp, dlp, d2lp] = constant_prior(default_value, theta)
%
% Inputs:
%
%   default_value: (ignored)
%           theta: the value of the hyperparameter (ignored)
%
% Outputs:
%
%     lp: the log prior evaluated at theta (log(1) = 0)
%    dlp: the derivative of the log prior evaluated at theta (0)
%   d2lp: the second derivative of the log prior evaluated at theta (0)
%
% Usage (sample mode)
% -------------------
%
%   sample = constant_prior(default_value)
%
% Inputs:
%
%   default_value: the value to return
%
% Output:
%
%   sample: a sample "drawn" from the prior (equal to default_value)
%
% See also: PRIORDISTRIBUTIONS.

% Copyright (c) 2014 Roman Garnett.

function [result, dlp, d2lp] = constant_prior(default_value, ~)

  % draw "sample"
  if (nargin < 2)
    result = default_value;
    return;
  end

  % log likelihood
  result = 0;

  % first derivative of log likelihood
  if (nargout >= 2)
    dlp = 0;
  end

  % second derivative of log likelihood
  if (nargout >= 3)
    d2lp = 0;
  end

end
