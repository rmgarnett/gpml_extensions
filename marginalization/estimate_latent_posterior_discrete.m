% binary gaussian process classifier with hyperparameter
% marginalization of hyperparameters via bayesian monte carlo. the
% posterior distribution of the latent function is approximated as
% a gaussian process mixture:
%
%   p(f | D) = \sum_i w_i p(f | D, \theta_i),
%
% where \sum_i w_i = 1 and the set {\theta_i} contains chosen
% hyperparameter samples.  the posterior weights w_i are calculated
% via bayesian monte carlo numerical integration.
%
% this function requires the gpml_extensions project available here
%
% https://github.com/rmgarnett/gpml_extensions
%
% function [latent_means, latent_covariances, hypersample_weights] = ...
%       estimate_latent_posterior_discrete(data, responses, train_ind, ...
%           test_ind, prior_covariances, inference_method, mean_function, ...
%           covariance_function, likelihood, hypersamples, full_covariance)
%
% inputs:
%                  data: an (n x d) matrix of input data
%             responses: an (n x 1) vector of -1 / 1 responses
%             train_ind: a list of indices into data/responses
%                        indicating the training points
%              test_ind: a list of indices into data/responses
%                        indicating the test points
%     prior_covariances: a (num_hypersamples x n x n) matrix
%                        containing the prior covariance matrices
%      inference_method: a gpml inference method
%         mean_function: a gpml mean function
%   covariance_function: a gpml covariance function
%            likelihood: a gpml likelihood
%          hypersamples: a hypersample structure for use with
%                        gpml_extensions
%       full_covariance: a boolean indicating whether the full
%                        posterior covariance over the latent function
%                        values on the test points should be
%                        calculated
%
% outputs:
%           latent_means: a (# hypersamples) x (# test points) matrix
%                         containing the posterior latent mean on the
%                         test points corresponding to each
%                         hyperparameter
%     latent_covariances: a (# hypersamples) x (# test points)
%                         matrix (if full_covariance = false), or a
%                         (# hypersamples) x (# test points) x (# test points)
%                         array (if full_covariance = true)
%                         containing either the pointwise posterior latent
%                         pointwise variance or the full posterior
%                         latent covariance on the test points
%    hypersample_weights: a (# hypersamples) x 1 vector containing
%                         the
%
% copyright (c) roman garnett, 2011--2012

function [latent_means, latent_covariances, hypersample_weights] = ...
      estimate_latent_posterior_discrete(data, responses, train_ind, ...
          test_ind, prior_covariances, inference_method, mean_function, ...
          covariance_function, likelihood, hypersamples, full_covariance)

  % do not calculate full covariance unless asked
  if (nargin < 10)
    full_covariance = false;
  end
  hyperparameters.full_covariance = full_covariance;

  num_hypersamples = size(hypersamples.values, 1);

  train_x  = data(train_ind, :);
  train_y  = responses(train_ind, :);
  test_x   = data(test_ind, :);
  num_test = size(test_x, 1);

  latent_means = zeros(num_hypersamples, num_test);
  if (full_covariance)
    latent_covariances = zeros(num_hypersamples, num_test, num_test);
  else
    latent_covariances = zeros(num_hypersamples, num_test);
  end
  log_likelihoods = zeros(num_hypersamples, 1);

  for i = 1:num_hypersamples
    try
      % fill gpml hyperparameters array from hypersample structure
      hyperparameters.lik  = hypersamples.values(i, hypersamples.likelihood_ind);
      hyperparameters.mean = hypersamples.values(i, hypersamples.mean_ind);
      hyperparameters.cov  = hypersamples.values(i, hypersamples.covariance_ind);

      % fill mean, covariance, and log likelihood arrays by calling gpml
      if (full_covariance)
        [~, ~, latent_means(i, :), latent_covariances(i, :, :), ~, ...
         log_likelihoods(i)] = ...
            gp_test_given_K(hyperparameteres, inference_method, mean_function, ...
                            covariance_function, likelihood, ...
                            prior_covariances(i, :, :), train_ind, test_ind, ...
                            train_x, train_y, test_x);
      else
        [~, ~, latent_means(i, :), latent_covariances(i, :), ~, ...
         log_likelihoods(i)] = ...
            gp_test_given_K(hyperparameteres, inference_method, mean_function, ...
                            covariance_function, likelihood, ...
                            prior_covariances(i, :, :), train_ind, test_ind, ...
                            train_x, train_y, test_x);
      end
    catch (message)
      warning('gpml_extensions:posterior_inference_failure', ...
              ['error calculating the latent posterior for hypersample' ...
               num2str(i)]);
      getReport(message);

      % effectively ignore sample
      log_likelihoods(i) = -Inf;
    end
  end

  % use bmc to calculate the posterior weights
  hypersamples.log_likelihoods = -log_likelihoods;
  hypersample_weights = calculate_hypersample_weights(hypersamples);

end