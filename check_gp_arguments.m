function [hyp, inf, mean, cov, lik] = ...
    check_gp_arguments(hyp, inf, mean, cov, lik, x, y)
% Check hyperparameters, inference method, mean function, covariance
% function, likelihood function, and training data compatibility with the
% Gaussian processes for use with
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
% [hyp, inf, mean, cov, lik] = ...
%    check_gp_parameters(hyp, inf, mean, cov, lik, x, y)
%
% where:
%
%   hyp      column vector of hyperparameters
%   inf      function specifying the inference method 
%   cov      prior covariance function (see below)
%   mean     prior mean function
%   lik      likelihood function
%   x        n by D matrix of training inputs
%
% Copyright (c) 2011 Roman Garnett.  All rights reserved.

if (isempty(inf))
  % set default inf
  inf = @infExact;
else                        
  if (iscell(inf))
    % cell input is allowed
    inf = inf{1}; 
  end
  if (ischar(inf))
    % convert into function handle
    inf = str2func(inf); 
  end
end

if (isempty(mean))
  % set default mean
  mean = {@meanZero}; 
end
if ( ischar(mean) || isa(mean, 'function_handle') )
  % make cell
  mean = {mean}; 
end

if (isempty(cov))
  % no default
  error('Covariance function cannot be empty'); 
end
if (ischar(cov) || isa(cov, 'function_handle'))
  % make cell
  cov = {cov};  
end
cov1 = cov{1}; 
if (isa(cov1, 'function_handle'))
  cov1 = func2str(cov1); 
end
if (strcmp(cov1,'covFITC'));
  % only one possible inf alg
  inf = @infFITC; 
end

if (isempty(lik))
  % set default lik
  lik = @likGauss;
else
  if (iscell(lik)) 
    % cell input is allowed
    lik = lik{1}; 
  end
  if (ischar(lik))
    % convert into function handle
    lik = str2func(lik);
  end
end

D = size(x,2);
% check the hyp specification
if (~isfield(hyp,'mean'))
  hyp.mean = []; 
end
if (eval(feval(mean{:})) ~= numel(hyp.mean))
  error('Number of mean function hyperparameters disagree with mean function');
end
if (~isfield(hyp,'cov'))
  hyp.cov = []; 
end
if (eval(feval(cov{:})) ~= numel(hyp.cov))
  error('Number of cov function hyperparameters disagree with cov function');
end
if (~isfield(hyp,'lik'))
  hyp.lik = []; 
end
if (eval(feval(lik)) ~= numel(hyp.lik))
  error('Number of lik function hyperparameters disagree with lik function');
end

if ( strcmp(func2str(lik), 'likErf') || strcmp(func2str(lik), 'likLogistic') )
  uy = unique(y);
  if any( (uy ~= +1) & (uy ~= -1) )
    warning('You attempt classification using labels different from {+1, -1}\n');
  end
end
