% GAUSSIAN_PRIOR priorGauss replacement computing second derivative.
%
% This provides a GPML-compatible prior function implementing a
% Gaussian prior for a hyperparameter with a given mean \mu and
% variance \sigma^2:
%
%   p(\theta) = N(\theta; \mu, \sigma^2)
%
% This can be used as a drop-in replacement for priorGauss.
%
% This implementation supports an extended GPML syntax that allows
% computing the second derivative of the log prior evalauted at
% theta. The syntax is:
%
%   [lp, dlp, d2lp] = gaussian_prior(mean, variance, theta),
%
% where d2lp = d^2 \log(p(\theta)) / d \theta^2.
%
% See also: PRIORGAUSS, PRIORDISTRIBUTIONS, INDEPENDENT_PRIOR.

% Copyright (c) 2014--2015 Roman Garnett.

function [result, dlp, d2lp] = gaussian_prior(mean, variance, theta)

  % call priorGauss for everything but second derivative
  if (nargin < 2)
    result = priorGauss();
  elseif (nargin == 2)
    result = priorGauss(mean, variance);
  else
    if (nargout <= 1)
       result       = priorGauss(mean, variance, theta);
    else
      [result, dlp] = priorGauss(mean, variance, theta);
    end
  end

  % if second derivative not requested, we are done.
  if (nargout <= 2)
    return;
  end

  % second derivative of log likelihood
  d2lp = -ones(size(result)) / variance;

end
