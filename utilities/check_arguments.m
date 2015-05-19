% performs argument checks/transformations similar to those found in
% gp.m from GPML, but Hessian friendly

% Copyright (c) 2014--2015 Roman Garnett.

function [theta, inference_method, mean_function, covariance_function, ...
          likelihood] = ...
  check_arguments(theta, inference_method, mean_function, covariance_function, ...
                  likelihood, x)

  % default to exact inference
  if (isempty(inference_method))
    inference_method = {@exact_inference};
  end

  % default to zero mean function
  if (isempty(mean_function))
    mean_function = {@zero_mean};
  end

  % no default covariance function
  if (isempty(covariance_function))
    error('gpml_extensions:missing_argument', ...
          'covariance function must be defined!');
  end

  % default to Gaussian likelihood
  if (isempty(likelihood))
    likelihood = {@likGauss};
  end

  % allow string/function handle input; convert to cell arrays if
  % necessary
  if (ischar(inference_method) || ...
      isa(inference_method, 'function_handle'))
    inference_method = {inference_method};
  end

  if (ischar(mean_function) || ...
      isa(mean_function, 'function_handle'))
    mean_function = {mean_function};
  end

  if (ischar(covariance_function) || ...
      isa(covariance_function, 'function_handle'))
    covariance_function = {covariance_function};
  end

  if (ischar(likelihood) || ...
      isa(likelihood, 'function_handle'))
    likelihood = {likelihood};
  end

  % ensure all hyperparameter fields exist
  for field = {'cov', 'lik', 'mean'}
    if (~isfield(theta, field{:}))
      theta.(field{:}) = [];
    end
  end

  % check dimension of hyperparameter fields
  D = size(x, 2);

  expression = feval(mean_function{:});
  if (numel(theta.mean) ~= eval(expression))
    error('gpml_extensions:incorrect_specification', ...
          'wrong number of mean hyperparameters! (%i given, %s expected)', ...
          numel(theta.mean), ...
          expression);
  end

  expression = feval(covariance_function{:});
  if (numel(theta.cov) ~= eval(expression))
    error('gpml_extensions:incorrect_specification', ...
          'wrong number of covariance hyperparameters! (%i given, %s expected)', ...
          numel(theta.cov), ...
          expression);
  end

  expression = feval(likelihood{:});
  if (numel(theta.lik) ~= eval(expression))
    error('gpml_extensions:incorrect_specification', ...
          'wrong number of likelihood hyperparameters! (%i given, %i expected)', ...
          numel(theta.lik), ...
          expression);
  end

end