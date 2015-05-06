% INDEPENDENT_PRIOR elementwise independent hyperparameter prior.
%
% This file implements a simple (but common) hyperparameter prior that
% is simply a product of elementwise indepdenent arbitrary priors on
% each hyperparameter.
%
% indepdendent_prior can be passed into, for example,
% infernce_with_prior to acheive MAP (rather than MLE) hyperparameter
% inference. See inference_with_prior.m for more information.
%
% This file supports both calculating the log prior, its gradient, and
% its (diagonal) Hessian matrix, as well as drawing a sample from the
% prior. The latter is accomplished by not passing in a value for the
% hyperparameters. See priors.m for more information.
%
% Usage (prior mode)
% ------------------
%
%   [lp, dlp, Hlp] = independent_prior(priors, theta)
%
% Inputs:
%
%   priors: a struct containing the elementwise priors for
%           each hyperparameter (see below)
%    theta: a GPML hyperparameter struct
%
% Outputs:
%
%    lZ: the log prior evaluated at the hyperparameters
%   dlp: the gradient of the log prior evaluated at \theta
%   Hlp: the a struct containing the Hessian of the log
%          prior evaluated at \theta (see hessians.m)
%
% Usage (sample mode)
% -------------------
%
%   sample = independent_prior(priors)
%
% Input:
%
%   priors: a struct containing the elementwise priors for each
%           hyperparameter (see below for expected format)
%
% Output:
%
%   sample: a GPML hyperparameter struct sampled from the prior
%
% Format of priors struct
% -----------------------
%
% This file requires specifying the elementwise priors for each
% hyperparameter. This is accomplished by creating a struct
% containing the same fields as the hyperparameter struct (e.g.,
% .cov, .lik, .mean). Each field should contain a cell array the
% same length as the corresponding hyperparameters. This cell array
% should contain function handles to one-dimensional priors for the
% corresponding hyperparameter. The one-dimensional priors should
% comply to the following API:
%
% [Prior mode]
%
%   [lp, dlp, d2lp] = prior(theta)
%
% Input:
%
%      theta: the value of the hyperparameter
%
% Outputs:
%
%     lp: the log prior evaluated at \theta
%    dlp: the derivative of the log prior evaluated at \theta
%   d2lp: the second derivative of the log prior evaluated at \theta
%
% [Sample mode]
%
%   sample = prior()
%
% Output:
%
%   sample: a sample drawn from the prior
%
% There are several example implementations provided for Gaussian,
% Laplace, constant (improper), and "smoothed uniform" hyperpriors.
%
% See also: GP_POSTERIOR, PRIORDISTRIBUTIONS, CONSTANT_PRIOR,
% GAUSSIAN_PRIOR, LAPLACE_PRIOR, SMOOTH_BOX_2_PRIOR, HESSIANS.

% Copyright (c) 2014--2015 Roman Garnett.

function [result, dlp, Hlp] = independent_prior(priors, theta)

  fields = {'cov', 'lik', 'mean'};

  % draw sample
  if (nargin < 2)
    for i = 1:numel(fields)
      field = fields{i};

      if (~isfield(priors, field))
        continue;
      end

      num_entries = numel(priors.(field));
      result.(field) = zeros(num_entries, 1);
      for j = 1:num_entries
        result.(field)(j) = priors.(field){j}();
      end
    end
    return;
  end

  % readability
  want_gradient = (nargout >= 2);
  want_hessian  = (nargout >= 3);

  % initialize output
  result = 0;
  if (want_gradient)
    dlp = theta;
  end

  if (want_hessian)
    num_hyperparameters = numel(unwrap(theta));

    if (isfield(theta, 'cov'))
      num_cov = numel(theta.cov);
    else
      num_cov = 0;
    end

    if (isfield(theta, 'lik'))
      num_lik = numel(theta.lik);
    else
      num_lik = 0;
    end

    Hlp.value = zeros(num_hyperparameters);
    Hlp.covariance_ind = 1:num_cov;
    Hlp.likelihood_ind = (num_cov + 1):(num_cov + num_lik);
    Hlp.mean_ind       = (num_cov + num_lik + 1):num_hyperparameters;
  end

  offset = 0;
  for i = 1:numel(fields)
    field = fields{i};

    if (~isfield(theta, field))
      continue;
    end

    num_entries = numel(theta.(field));

    % apply each of the priors to the input in turn
    for j = 1:num_entries
      % call prior with the appropriate number of outputs
      if (~(want_gradient || want_hessian))
        lp = ...
            priors.(field){j}(theta.(field)(j));
      elseif (want_gradient && ~want_hessian)
        [lp, dlp.(field)(j)] = ...
            priors.(field){j}(theta.(field)(j));
      else
        [lp, dlp.(field)(j), Hlp.value(offset + j, offset + j)] = ...
            priors.(field){j}(theta.(field)(j));
      end

      % accumulate log prior
      result = result + lp;
    end

    % update Hessian offset
    offset = offset + num_entries;
  end

end