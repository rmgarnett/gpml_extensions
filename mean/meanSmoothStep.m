% smooth step-change mean function:
%
% \mu(x) = { c_1  (x(:, changepoint_dimension) < t - w / 2)
%          { c_1 + (c_2 - c_1) * \int_-oo^x exp(-1 / (1 - (2(y - t) / w)^2) dy
%          {      (t - w / 2) < x(:, changepoint_dimension) < (t + w / 2)
%          { c_2  (x(:, changepoint_dimension) > t + w / 2)
%
% the hyperparameters are:
%
% hyperparameters = [ c_1
%                     c_2
%                     t
%                     w ]
%
% the time hyperparamter t is associated with an identified
% dimension of the input, changepoint_dimension.
% 
% See also meanFunctions.m in the GPML toolkit.

function output = meanSmoothStep(changepoint_dimension, hyperparameters, ...
        x, hyperparameter_ind)

  % report number of hyperparameters 
  if (~exist('x', 'var'))
    output = '4';
    return;
  end
  
  c_1 = hyperparameters(1);
  c_2 = hyperparameters(2);
  t   = hyperparameters(3);
  w   = hyperparameters(4);
  
  before_ind = (x(:, changepoint_dimension) <  t - w / 2);
  after_ind  = (x(:, changepoint_dimension) >= t + w / 2);
  during_ind = find(~(before_ind | after_ind));

  output = zeros(size(x, 1), 1);
  
  f = @(x) exp(-1 ./ (1 - (2 * (x - t) / w).^2));
  area = quadgk(f, t - w / 2, t + w / 2);

  % evaluate prior mean
  if (~exist('hyperparameter_ind', 'var'))
    output(before_ind) = c_1;
    output(after_ind)  = c_2;
    
    for i = 1:numel(during_ind)
      output(during_ind(i)) = ...
          c_1 + (c_2 - c_1) * quadgk(f, t - w / 2, x(during_ind(i))) / area;
    end

  % evaluate derivatives with respect to hyperparameters
  else
    % before constant
    if (hyperparameter_ind == 1)
      output(before_ind) = 1;
      for i = 1:numel(during_ind)
        output(during_ind(i)) = 1 - quadgk(f, t - w / 2, x(during_ind(i))) / area;
      end

    % after constant
    elseif (hyperparameter_ind == 2)
      output(after_ind) = 1;
      for i = 1:numel(during_ind)
        output(during_ind(i)) = quadgk(f, t - w / 2, x(during_ind(i))) / area;
      end

    % changepoint center
    elseif (hyperparameter_ind == 3)
      g = @(x) ...
          8 * (x - t) .* f(x) ./ ...
          (w^2 * (1 - (2 * (x - t) / w).^2));

      for i = 1:numel(during_ind)
        output(during_ind(i)) = ...
            (c_2 - c_1) * quadgk(g, t - w / 2, x(during_ind(i))) / area;
      end

    % width
    elseif (hyperparameter_ind == 4)
      g = @(x) ...
          8 * w * (t - x).^2 .* f(x) ./ ...
          (w^2 - 4 * (t - x).^2).^2;

      for i = 1:numel(during_ind)
        output(during_ind(i)) = ...
            (c_2 - c_1) * quadgk(g, t - w / 2, x(during_ind(i))) / area;
      end
    end
  end
end