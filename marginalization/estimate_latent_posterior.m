function [latent_means, latent_covariances, hypersample_weights, log_likelihoods] = ...
      estimate_latent_posterior(data, responses, test, inference_method, ...
                                mean_function, covariance_function, ...
                                likelihood, hypersamples, full_covariance)

  if (nargin < 9)
    full_covariance = false;
  end
  hyperparameters.full_covariance = full_covariance;

  num_hypersamples = size(hypersamples.values, 1);
  num_test = size(test, 1);

  latent_means = zeros(num_hypersamples, num_test);
  if (full_covariance)
    latent_covariances = zeros(num_hypersamples, num_test, num_test);
  else
    latent_covariances = zeros(num_hypersamples, num_test);
  end

  log_likelihoods = zeros(num_hypersamples, 1);

  for i = 1:num_hypersamples
    try
      hyperparameters.lik  = hypersamples.values(i, hypersamples.likelihood_ind);
      hyperparameters.mean = hypersamples.values(i, hypersamples.mean_ind);
      hyperparameters.cov  = hypersamples.values(i, hypersamples.covariance_ind);

      if (full_covariance)
        [~, ~, latent_means(i, :), latent_covariances(i, :, :), ~, ...
         log_likelihoods(i)] = gp_test(hyperparameters, inference_method, ...
                mean_function, covariance_function, likelihood, data, ...
                responses, test);
      else
        [~, ~, latent_means(i, :), latent_covariances(i, :), ~, ...
         log_likelihoods(i)] = gp_test(hyperparameters, inference_method, ...
                mean_function, covariance_function, likelihood, data, ...
                responses, test);
      end
    catch message
      disp('error in estimate_latent_posterior:');
      disp(getReport(message));
      log_likelihoods(i) = Inf;
    end
  end

  log_likelihoods = -log_likelihoods;
  hypersample_weights = exp(log_likelihoods - max(log_likelihoods));

end