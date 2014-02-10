% SMOOTH_STEP_MEAN smooth step-change mean function.
%
% \mu(x) = { c_1  (x(:, changepoint_dimension) < t - w / 2)
%          { c_1 + (c_2 - c_1) * \int_-oo^x exp(-1 / (1 - (2(y - t) / w)^2) dy
%          {      (t - w / 2) < x(:, changepoint_dimension) < (t + w / 2)
%          { c_2  (x(:, changepoint_dimension) > t + w / 2)
%
% The hyperparameters are:
%
% hyperparameters = [ c_1
%                     c_2
%                     t
%                     w   ],
%
% where t is the midpoint of the changepoint period and w is the width
% of the changepoint.
%
% The time hyperparamter t is associated with an identified dimension
% of the input, changepoint_dimension.
%
% See also MEANFUNCTIONS.

% Copyright (c) Roman Garnett, 2012--2014

function result = smooth_step_mean(changepoint_dimension, hyperparameters, ...
          x, i)

  % check for changepoint dimension
  if (nargin == 0)
    error('gpml_extensions:missing_argument', ...
          'changepoint_dimension must be specified!');
  end

  % report number of hyperparameters
  if (nargin <= 2)
    result = '4';
    return;
  end

  c_1 = hyperparameters(1);
  c_2 = hyperparameters(2);
  t   = hyperparameters(3);
  w   = hyperparameters(4);

  before_ind = (x(:, changepoint_dimension) <  t - w / 2);
  after_ind  = (x(:, changepoint_dimension) >= t + w / 2);
  during_ind = find(~(before_ind | after_ind));

  result = zeros(size(x, 1), 1);

  f = @(x) exp(-1 ./ (1 - (2 * (x - t) / w).^2));
  area = quadgk(f, t - w / 2, t + w / 2);

  % evaluate prior mean
  if (nargin == 3)
    result(before_ind) = c_1;
    result(after_ind)  = c_2;

    for j = 1:numel(during_ind)
      result(during_ind(j)) = ...
          c_1 + (c_2 - c_1) * quadgk(f, t - w / 2, x(during_ind(j))) / area;
    end

  % evaluate derivatives with respect to hyperparameters
  else
    % before constant
    if (i == 1)
      result(before_ind) = 1;
      for j = 1:numel(during_ind)
        result(during_ind(j)) = 1 - quadgk(f, t - w / 2, x(during_ind(j))) / area;
      end

    % after constant
    elseif (i == 2)
      result(after_ind) = 1;
      for j = 1:numel(during_ind)
        result(during_ind(j)) = quadgk(f, t - w / 2, x(during_ind(j))) / area;
      end

    % changepoint center
    elseif (i == 3)
      g = @(x) ...
          8 * (x - t) .* f(x) ./ ...
          (w^2 * (1 - (2 * (x - t) / w).^2));

      for j = 1:numel(during_ind)
        result(during_ind(j)) = ...
            (c_2 - c_1) * quadgk(g, t - w / 2, x(during_ind(j))) / area;
      end

    % width
    elseif (i == 4)
      g = @(x) ...
          8 * w * (t - x).^2 .* f(x) ./ ...
          (w^2 - 4 * (t - x).^2).^2;

      for j = 1:numel(during_ind)
        result(during_ind(j)) = ...
            (c_2 - c_1) * quadgk(g, t - w / 2, x(during_ind(j))) / area;
      end
    end
  end
end