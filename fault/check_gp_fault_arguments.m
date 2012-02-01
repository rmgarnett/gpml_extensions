% Check hyperparameters, inference method, mean function, covariance
% function, likelihood function, and training data compatibility with
% the Gaussian processes for use with
%
% GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 3.1
%   for GNU Octave 3.2.x and Matlab 7.x
%
% Copyright (c) 2005-2010 Carl Edward Rasmussen & Hannes Nickisch. All
% rights reserved.
%
% This code is taken almost verbatim from the gp.m function included
% in that package.  It allows error checking to be performed only once
% before further inference.  In some cases, a modified hyperparameter
% structure, inference method, mean function, covariance function, or
% likelihood may be returned, for compatibility with later code.
%
% Usage:
%
% function [hyperparameters, inference_method, mean_function, ...
%           covariance_function, likelihood, a_function, b_function] ...
%       = check_gp_fault_arguments(hyperparameters, inference_method, ...
%           mean_function, covariance_function, likelihood, a_function, ...
%           b_function, x)
%
% where:
%
%      inference_method: function specifying the inference method
%   covariance_function: prior covariance function (see below)
%         mean_function: prior mean function
%            likelihood: likelihood function
%            a_function: the a(x) function in a(x)y + b(x). this
%                        should be a valid GPML mean function.
%            b_function: the a(x) function in a(x)y + b(x). this
%                        should be a valid GPML mean function.
%               train_x: training x points
%
% Copyright (c) 2011 Roman Garnett.  All rights reserved.

function [hyperparameters, inference_method, mean_function, ...
          covariance_function, likelihood, a_function, b_function] ...
      = check_gp_fault_arguments(hyperparameters, inference_method, ...
          mean_function, covariance_function, likelihood, a_function, ...
          b_function, fault_covariance_function, train_x)

% classification likelihoods not allowed
if (strcmp(func2str(likelihood), 'likErf') || ...
    strcmp(func2str(likelihood), 'likLogistic'))
  error('gp_extensions:fault_classification_unsupported', ...
        'classification not supported!');
end

if (isempty(inference_method))
  % set default inference method
  inference_method = @infExactFault;
else
  if (iscell(inference_method))
    % cell input is allowed
    inference_method = inference_method{1};
  end
  if (ischar(inference_method))
    % convert into function handle
    inference_method = str2func(inference_method);
  end
end

if (isempty(mean_function))
  % set default mean
  mean_function = {@meanZero};
end
if (ischar(mean_function) || isa(mean_function, 'function_handle'))
  % make cell
  mean_function = {mean_function};
end

if (isempty(covariance_function))
  % no default
  error('gp_extensions:covariance_function_unspecified', ...
        'covariance function cannot be empty');
end
if (ischar(covariance_function) || isa(covariance_function, 'function_handle'))
  % make cell
  covariance_function = {covariance_function};
end
first_covariance_function = covariance_function{1};
if (isa(first_covariance_function, 'function_handle'))
  first_covariance_function = func2str(first_covariance_function);
end
if (strcmp(first_covariance_function, 'covFITC'));
  % only one possible inference method
  inference_method = @infFITC;
end

if (isempty(fault_covariance_function))
    fault_covariance_function = covNoise;
end
if (ischar(fault_covariance_function) || isa(fault_covariance_function, 'function_handle'))
  % make cell
  fault_covariance_function = {fault_covariance_function};
end

if (isempty(likelihood))
  % set default likelihood
  likelihood = @likGauss;
else
  if (iscell(likelihood))
    % cell input is allowed
    likelihood = likelihood{1};
  end
  if (ischar(likelihood))
    % convert into function handle
    likelihood = str2func(likelihood);
  end
end

if (isempty(a_function))
  % set default a function
  a_function = {@meanOne};
end
if (ischar(a_function) || isa(a_function, 'function_handle'))
  % make cell
  a_function = {a_function};
end

if (isempty(b_function))
  % set default b function
  b_function = {@meanZero};
end
if (ischar(b_function) || isa(b_function, 'function_handle'))
  % make cell
  b_function = {b_function};
end

D = size(train_x, 2);
% check the hyperparameter specification
if (~isfield(hyperparameters, 'mean'))
  hyperparameters.mean = [];
end
if (eval(feval(mean_function{:})) ~= numel(hyperparameters.mean))
    error('gpml_extensions:mean_function_hyperparameters_size', ...
        'Number of mean function hyperparameters disagrees with mean function');
end

if (~isfield(hyperparameters, 'cov'))
  hyperparameters.cov = [];
end
if (eval(feval(covariance_function{:})) ~= numel(hyperparameters.cov))
  error('gpml_extensions:covariance_hyperparameters_size', ...
        'Number of covariance function hyperparameters disagrees with covariance function');
end

if (~isfield(hyperparameters, 'lik'))
  hyperparameters.lik = [];
end
if (eval(feval(likelihood)) ~= numel(hyperparameters.lik))
  error('gpml_extensions:likelihood_hyperparameters_size', ...
        'Number of likelihood function hyperparameters disagree with likelihood function');
end

if (~isfield(hyperparameters, 'a'))
  hyperparameters.a = [];
end
if (eval(feval(a_function{:})) ~= numel(hyperparameters.a))
  error('gpml_extensions:a_function_hyperparameters_size', ...
        'Number of a function hyperparameters disagree with a function');
end

if (~isfield(hyperparameters, 'b'))
  hyperparameters.b = [];
end
if (eval(feval(b_function{:})) ~= numel(hyperparameters.b))
  error('gpml_extensions:b_function_hyperparameters_size', ...
        'Number of b function hyperparameters disagree with b function');
end

if (~isfield(hyperparameters, 'fault_covariance_function'))
  hyperparameters.fault_covariance_function = [];
end
if (eval(feval(fault_covariance_function{:})) ~= numel(hyperparameters.fault_covariance_function))
  error('gpml_extensions:covariance_hyperparameters_size', ...
        'Number of fault covariance function hyperparameters disagrees with fault covariance function');
end
