% step-change mean function:
%
% \mu(x) = { c_1  (x(:, changepoint_dimension) < t)
%          { c_2  (x(:, changepoint_dimension) > t)
%
% the hyperparameters are:
%
% hyperparameters = [ c_1
%                     c_2
%                     t   ]
%
% the time hyperparamter t is associated with an identified
% dimension of the input, changepoint_dimension.
% 
% See also meanFunctions.m in the GPML toolkit.

function output = meanStep(changepoint_dimension, hyperparameters, ...
                           x, hyperparameter_ind)

  % report number of hyperparameters 
  if (nargin < 3)
    output = '3';
    return;
  end

  % check arguments
  if (numel(hyperparameters) ~= 3)
    error('exactly three hyperparameters needed.')
  end
  
  c_1 = hyperparameters(1);
  c_2 = hyperparameters(2);
  t   = hyperparameters(3);
  
  before_ind = (x(:, changepoint_dimension) < t);
  after_ind  = ~(before_ind);

  output = zeros(size(x, 1), 1);

  % evaluate prior mean
  if (nargin == 3)
    output(before_ind) = c_1;
    output(after_ind)  = c_2;

  % evaluate derivatives with respect to hyperparameters
  else
    % before constant
    if (hyperparameter_ind == 1)
      output(before_ind) = 1;
    % after constant
    elseif (hyperparameter_ind == 2)
      output(after_ind) = 1;
    end
  end
end