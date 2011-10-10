function [latent_means latent_covariances hypersample_weights] = ...
      estimate_latent_posterior(data, responses, test, inference_method, ...
                                mean_function, covariance_function, ...
                                likelihood, hypersamples, full_covariance)
  
  if (nargin < 9)
    full_covariance = false;
  end

  num_hypersamples = size(hypersamples.values, 1);
  num_test = size(test, 1);

  latent_means = zeros(num_test, num_hypersamples);
  if (full_covariance)
    latent_covariances = zeros(num_test, num_test, num_hypersamples);
  else
    latent_covariances = zeros(num_test, num_hypersamples);
  end

  log_likelihoods = zeros(num_hypersamples, 1);

  for i = 1:num_hypersamples
    try
      hyp.lik = hypersamples.values(i, hypersamples.likelihood_ind);
      hyp.mean = hypersamples.values(i, hypersamples.mean_ind);
      hyp.cov = hypersamples.values(i, hypersamples.covariance_ind);
      
      if (full_covariance)
        [~, ~, latent_means(:, i), latent_covariances(:, :, i), ~, ...
         log_likelihoods(i)] = gp_test_full_covariance(hyp, inference_method, ...
                mean_function, covariance_function, likelihood, data, ...
                responses, test);
      else
        [~, ~, latent_means(:, i), latent_covariances(:, i), ~, ...
         log_likelihoods(i)] = gp_test(hyp, inference_method, ...
                mean_function, covariance_function, likelihood, data, ...
                responses, test);
      end
    catch
      log_likelihoods(i) = -Inf;
    end
  end
  
  hypersamples.log_likelihoods = -log_likelihoods;
  hypersample_weights = calculate_hypersample_weights(hypersamples);

end