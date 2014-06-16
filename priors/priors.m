% Hyperparameter priors
% ---------------------
%
% Hyperparameter priors are supported by, e.g., inference_with_prior
% to allow MAP rather than MLE inference when training a
% GP.
%
% We support two modes for hyperpriors: calculating the negative log
% prior, its gradient, and its Hessian, as well as drawing a
% sample. The API for both modes is documented below.
%
% Usage (prior mode)
% ------------------
%
%   [nlZ, dnlZ, HnlZ] = prior(hyperparameters)
%
% Inputs:
%
%   hyperparameters: a GPML hyperparameter struct
%
% Outputs:
%
%     nlZ: the negative log prior evaluated at the hyperparameters
%    dnlZ: the gradient of the negative log prior evaluated at the
%          hyperparameters
%    HnlZ: a struct containing the Hessian of the negative log prior
%          evaluated at theta (see hessians.m)
%
% Usage (sample mode)
% -------------------
%
%   sample = prior()
%
% Output:
%
%   sample: a GPML hyperparameter struct sampled from the prior
%
% Implementations
% ---------------
%
% The primary implementation provided here is in independent_prior.m,
% which provides a simple elementwise independent prior for each
% hyperparameter. See that file for more information.
%
% See also INDEPENDENT_PRIOR.

% Copyright (c) 2014 Roman Garnett.
