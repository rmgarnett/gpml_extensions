% GET_PRIOR create function handle to hyperprior.
%
% This is a convenience function for easily creating a function handle
% to a hyperparameter prior. Given a handle to a prior and its
% additional arguments (if any), returns a function handle for use in,
% e.g., independent_prior.m
%
% Example:
%
%   prior = get_prior(@gaussian_prior, mean, variance);
%
% returns the following function handle:
%
%   @(theta) gaussian_prior(mean, variance, theta),
%
% where mean and variance are transparantly fixed to the values
% provided.
%
% This is primarily for improving code readability by avoiding
% repeated verbose function handle declarations.
%
% inputs:
%      prior: a function handle to the desired prior
%   varargin: any additional inputs to be bound to the prior beyond
%             those required by the standard interface (theta)
%
% output:
%   prior: a function handle to the desired prior
%
% See also PRIORS.

% Copyright (c) 2014 Roman Garnett.

function prior = get_prior(prior, varargin)

  extra_arguments = varargin;
  prior = @(varargin) prior(extra_arguments{:}, varargin{:});

end