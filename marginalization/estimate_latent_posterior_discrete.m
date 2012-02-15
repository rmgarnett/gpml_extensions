function [latent_means, latent_covariances, hypersample_weights] = ...
      estimate_latent_posterior_discrete(data, responses, train_ind, ...
          test_ind, prior_covariances, inference_method, mean_function, ...
          covariance_function, likelihood, hypersamples, full_covariance)

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
      hyperparameters.lik  = hypersamples.values(i, hypersamples.likelihood_ind);
      hyperparameters.mean = hypersamples.values(i, hypersamples.mean_ind);
      hyperparameters.cov  = hypersamples.values(i, hypersamples.covariance_ind);

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
    catch message
      disp('error in estimate_latent_posterior:');
      getReport(message);
      log_likelihoods(i) = -Inf;
    end
  end

  hypersamples.log_likelihoods = -log_likelihoods;
  hypersample_weights = calculate_hypersample_weights(hypersamples);

end