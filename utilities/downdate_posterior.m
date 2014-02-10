% UPDATE_POStERIOR updates GPML posterior to remvoe an observation.
%
% This function provides a fast update for a GPML posterior structure
% given a single new observation, avoiding O(N^3) retraining time.
% This implementation assumes exact inference (inf = @infExact) and a
% Gaussian likelihood (lik = @likGauss). It will not work correctly
% for other combinations of likelihood and inference!
%
% Usage:
%
%   new_posterior = update_posterior(hyperparameters, mean_function, ...
%           covariance_function, x, posterior, x_star, y_star)
%
% Inputs:
%
%       hyperparameters: a GPML hyperparameter struct
%         mean_function: a GPML mean function
%   covariance_function: a GPML covariance function
%                     x: the previous observations (N x D)
%             posterior: a GPML posterior struct
%                x_star: the new observation location (1 x D)
%                y_star: the new observation value
%
% Output:
%
%   new_posterior: the updated GPML posterior struct for the GP
%                  conditioned on ([x; x_star], [y; y_star]).
%
% See also INFMETHODS

% Copyright (c) Roman Garnett, 2013--2014

function new_posterior = downdate_posterior(hyperparameters, ...
          mean_function, covariance_function, posterior, ind)

  n = numel(posterior.alpha);

  new_ind = [1:(ind - 1), (ind + 1):n];

  noise_variance = exp(2 * hyperparameters.lik);

  % update posterior.L; handle both high- and low-noise
  % parameterizations of posterior
  if (is_chol(posterior.L))
    % high-noise parameterization: posterior.L contains chol(K / sigma^2 + I)

    alpha_update = solve_chol(posterior.L, k_star) * (1 / noise_variance);

    new_L_column = linsolve(posterior.L, k_star, ...
                            struct('UT', true, 'TRANSA', true)) * (1 / noise_variance);
    new_posterior.L = [posterior.L, new_L_column; ...
                       zeros(1, n), sqrt(1 + k / noise_variance - new_L_column' * new_L_column)];
  else
    % low-noise parameterization: posterior.L contains -inv(K + \sigma^2 I)

    u = posterior.L(ind, ind);
    v = posterior.L(new_ind, ind);

    new_posterior.L = posterior.L(new_ind, new_ind) - v * v' / u;
  end


  new_posterior.alpha = ...
      [posterior.alpha; 0] + ...
      (m_star - y_star) / v_star * [alpha_update; -1];

  % noise vector is constant; just add one more entry
  new_posterior.sW = posterior.sW([1; (1:end)']);

end