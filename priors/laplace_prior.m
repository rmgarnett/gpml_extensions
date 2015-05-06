% LAPLACE_PRIOR priorLaplace replacement computing second derivative.
%
% This provides a GPML-compatible prior function implementing a
% Laplace prior for a hyperparameter with a given mean \mu and
% variance \sigma^2:
%
%   p(\theta) = Laplce(\theta; \mu, \sigma^2)
%
% This can be used as a drop-in replacement for priorLaplace.
%
% This implementation supports an extended GPML syntax that allows
% computing the second derivative of the log prior evalauted at
% theta. The syntax is:
%
%   [lp, dlp, d2lp] = laplace_prior(mean, variance, theta),
%
% where d2lp = d^2 \log(p(\theta)) / d \theta^2.
%
% See also: PRIORLAPLACE, PRIORDISTRIBUTIONS, INDEPENDENT_PRIOR.

% Copyright (c) 2014--2015 Roman Garnett.

function [result, dlp, d2lp] = laplace_prior(mean, variance, theta)

  % call priorLaplace for everything but second derivative
  if (nargin < 2)
    result = priorLaplace();
  elseif (nargin == 2)
    result = priorLaplace(mean, variance);
  else
    if (nargout <= 1)
       result       = priorLaplace(mean, variance, theta);
    else
      [result, dlp] = priorLaplace(mean, variance, theta);
    end
  end

  % if second derivative not requested, we are done.
  if (nargout <= 2)
    return;
  end

  % second derivative of log likelihood
  d2lp = zeros(size(result));

end
