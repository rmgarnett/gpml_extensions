function A = meanDrift(mean, hyp, x, i)

% Mean function that is 0 except on a specified interval, where it
% takes a specified shape
%
% (x < begin_time or x > begin_time + width)
% ->  m(x) = 0
%
% (otherwise)
% ->  m(x) = f((x - begin_time) / (width))
%
% where f(x) is defined by a GPML mean function. The last dimension of
% x is the dimension along which the "times" are defined.
%
% The hyperparameterd str:
%
% hyp = [ begin_time
%         log(width) ]
%
% Based on code from:
%
% GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 3.1
%    for GNU Octave 3.2.x and Matlab 7.x
% Copyright (c) 2005-2010 Carl Edward Rasmussen & Hannes Nickisch. All
% rights reserved.
%
% Copyright (c) 2011 Roman Garnett. All rights reserved.
%
% See also MEANFUNCTIONS.M.

% report number of hyperparameters
if (nargin < 3)
  A = ['2+' feval(mean{:})];
  return;
end

central_time = hyp(1);
width        = exp(hyp(2));
others       = hyp(3:end);

begin_time = central_time - width / 2;
end_time   = central_time + width / 2;

A = zeros(size(x, 1), 1);

ind = (x(:, end) >= begin_time) & (x(:, end) <= end_time);
transformed_x = (x(ind, end) - begin_time) / (end_time - begin_time);

if (nargin == 3)
  % evaluate mean
  A(ind) = feval(mean{:}, others, transformed_x);
else
  % evaluate derivatives
  A(ind) = feval(mean{:}, others, transformed_x, i);
end
