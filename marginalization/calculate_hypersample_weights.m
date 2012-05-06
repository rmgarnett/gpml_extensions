function hypersample_weights = calculate_hypersample_weights(hypersamples)

  jitter = 1e-5;
  
  samples = hypersamples.values(:, hypersamples.marginal_ind);
  [num_samples, num_hyperparameters] = size(samples);

  K = ones(num_samples);
  L = ones(num_samples);

  for i = hypersamples.marginal_ind
    [x, y] = meshgrid(samples(:, i), samples(:, i));

    K = K .* normpdf(x - y, 0, hypersamples.length_scales(i));

    mu    = hypersamples.prior_means(i) * ones(1, 2);
    Sigma = hypersamples.prior_variances(i) * ones(2) + ...
            hypersamples.length_scales(i) * eye(2);

    L = L .* reshape(mvnpdf([x(:) y(:)], mu, Sigma), ...
                     num_samples, num_samples);
  end

  K = K + jitter * ones(num_samples);
  
  likelihoods = exp(hypersamples.log_likelihoods - ...
                    max(hypersamples.log_likelihoods));

  hypersample_weights = (K \ L) / K * likelihoods;
  hypersample_weights = hypersample_weights / sum(hypersample_weights);

end