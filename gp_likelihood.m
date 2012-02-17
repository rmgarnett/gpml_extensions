function [varargout] = gp_likelihood(hyp, inf, mean, cov, lik, x, y)
% Calcuate the negative log marginal likelihood of a Gaussian process
% and its partial derivatives with respect to the hyperparameters
% given training data.
%
% Please call check_gp_arguments before calling this function.
%
% Usage:
%   [nlZ dnlZ post] = gp_likelihood(hyp, inf, mean, cov, lik, x, y);
%
% where:
%
%   hyp      column vector of hyperparameters
%   inf      function specifying the inference method 
%   cov      prior covariance function (see below)
%   mean     prior mean function
%   lik      likelihood function
%   x        n by D matrix of training inputs
%   y        column vector of length n of training targets
%
%   nlZ      returned value of the negative log marginal likelihood
%   dnlZ     column vector of partial derivatives of the negative
%   post     struct representation of the (approximate) posterior
%            3rd output in training mode and 6th output in prediction mode
% 
% See also covFunctions.m, infMethods.m, likFunctions.m, meanFunctions.m.
%
% Based on code from:
%
% GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 3.1
%    for GNU Octave 3.2.x and Matlab 7.x
% Copyright (c) 2005-2010 Carl Edward Rasmussen & Hannes Nickisch. All
% rights reserved.
%
% Copyright (c) 2011 Roman Garnett. All rights reserved.

try
  % report -log marg lik, derivatives and post
  if (nargout == 1)
    [post nlZ] = inf(hyp, mean, cov, lik, x, y); 
    varargout = {nlZ, {}, post};
    return;
  else
    [post nlZ dnlZ] = inf(hyp, mean, cov, lik, x, y);
    varargout = {nlZ, dnlZ, post};
    return;
  end
 catch msgstr
  warning('gp_likelihood:inference_fail', ...
          ['Inference method failed [' msgstr.message ']' ...
           ', attempting to continue']);
  dnlZ = struct('cov', 0 * hyp.cov, ...
                'mean', 0 * hyp.mean, ...
                'lik', 0 * hyp.lik);
  % continue with a warning
  varargout = {NaN, dnlZ, post};
  return;
end
