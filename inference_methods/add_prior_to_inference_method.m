% ADD_PRIOR_TO_INFERENCE_METHOD adds hyperprior to GPML inference method.
%
% This function may be used to generate a GPML compatible inference
% method by combining a hyperparameter prior p(\theta) with an
% existing inference method. See inference_with_prior.m for more
% information.
%
% This function is required because GPML does not allow
% constructing "meta" inference methods in the same way as, e.g.,
% mean and covariance functions.
%
% Usage
% -----
%
%   inference_method = ...
%       add_prior_to_inference_method(inference_method, prior)
%
% Inputs:
%
%   inference_method: a GPML compatible inference method
%              prior: a hyperparameter prior (see priors.m)
%
% Outputs:
%
%   inference_method: a GPML compatible inference method
%                     incorporating the given hyperparameter prior
%
% See also INFERENCE_WITH_PRIOR, PRIORS.

% Copyright (c) 2014 Roman Garnett.

function inference_method = add_prior_to_inference_method(inference_method, ...
          prior)

  inference_method = @(hyperparameters, mean_function, covariance_function, ...
                       likelihood, x, y) ...
      inference_with_prior(inference_method, prior, hyperparameters, ...
                           mean_function, covariance_function, ...
                           likelihood, x, y);

end