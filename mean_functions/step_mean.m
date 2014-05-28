% STEP_MEAN simple step-change mean function.
%
%   \mu(x) = { c_1  (x(:, changepoint_dimension) < t)
%            { c_2  (x(:, changepoint_dimension) > t)
%
% The hyperparameters are:
%
%   hyperparameters = [ c_1
%                       c_2
%                       t   ],
%
% where t is the changepoint time.
%
% The time hyperparamter t is associated with an identified dimension
% of the input, changepoint_dimension.
%
% See also MEANFUNCTIONS.

% Copyright (c) 2012--2014 Roman Garnett.

function result = step_mean(changepoint_dimension, hyperparameters, x, i)

  % check for changepoint dimension
  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'changepoint_dimension must be specified!');
  end

  % report number of hyperparameters
  if (nargin <= 2)
    result = '3';
    return;
  end

  c_1 = hyperparameters(1);
  c_2 = hyperparameters(2);
  t   = hyperparameters(3);

  before_ind = (x(:, changepoint_dimension) < t);
  after_ind  = ~before_ind;

  result = zeros(size(x, 1), 1);

  % evaluate prior mean
  if (nargin == 3)
    result(before_ind) = c_1;
    result(after_ind)  = c_2;

  % evaluate derivatives with respect to hyperparameters
  else
    % before constant
    if (i == 1)
      result(before_ind) = 1;
    % after constant
    elseif (i == 2)
      result(after_ind) = 1;
    end
  end

end