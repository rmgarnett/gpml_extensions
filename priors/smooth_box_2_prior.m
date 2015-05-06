% GAUSSIAN_PRIOR priorGauss replacement computing second derivative.
%
% This provides a GPML-compatible prior function implementing a
% "smoothed uniform prior" with quadraticly decaying tails.  This can
% be used as a drop-in replacement for priorSmoothBox2.
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

function [result, dlp, d2lp] = smooth_box_2_prior(a, b, eta, theta)

  % call priorSmoothBox2 for everything but second derivative
  if (nargin < 3)
    result = priorSmoothBox2();
  elseif (nargin == 3)
    result = priorSmoothBox2(a, b, eta);
  else
    if (nargout <= 1)
       result       = priorSmoothBox2(a, b, eta, theta);
    else
      [result, dlp] = priorSmoothBox2(a, b, eta, theta);
    end
  end

  % if second derivative not requested, we are done.
  if (nargout <= 2)
    return;
  end

  % second derivative of log likelihood
  sqrt_2pi = 2.506628274631000;

  sab = abs(b - a) / (eta * sqrt_2pi);

  d2lp = zeros(size(result));

  ind = (theta < a) | (theta > b);
  d2lp(ind) = -1 / sab^2;

end