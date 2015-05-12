% ZERO_MEAN zero mean function.
%
% This provides a GPML-compatible mean function implementing a
% zero mean function:
%
%   \mu(x) = 0.
%
% This can be used as a drop-in replacement for meanZero.
%
% This implementation supports an extended GPML syntax that allows
% calculating the Hessian of \mu with respect to any pair of
% hyperparameters. The syntax is:
%
%   dm2_didj = zero_mean(theta, x, i, j)
%
% where dm2_didj is \partial^2 \mu / \partial \theta_i \partial \theta_j,
% and the Hessian is evaluated at x.
%
% These Hessians can be used to ultimately compute the Hessian of the
% GP training likelihood (see, for example, exact_inference.m).
%
% No hyperparameters are required.
%
% See also MEANZERO, MEANFUNCTIONS.

% Copyright (c) 2014--2015 Roman Garnett.

function result = zero_mean(~, x, ~, ~)

  % report number of hyperparameters
  if (nargin <= 1)
    result = '0';
    return;
  end

  % any other mode always returns the all-zeros matrix.
  result = zeros(size(x, 1), 1);

end