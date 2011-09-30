function hypersample_weights = calculate_hypersample_weights(hypersamples)

  hypersamples.log_likelihoods(isinf(hypersamples.log_likelihoods)) ...
      = min(hypersamples.log_likelihoods(isfinite(hypersamples.log_likelihoods))) - 1e3;

  [quad_noise_sd, quad_input_scales, quad_output_scale] = ...
      hp_heuristics(hypersamples.values(:, hypersamples.marginal_ind), ...
                    hypersamples.log_likelihoods, 100);

  quad_gp.quad_noise_sd = quad_noise_sd;
  quad_gp.quad_input_scales = quad_input_scales;
  quad_gp.quad_output_scale = quad_output_scale;
  
  num_hypersamples = size(hypersamples.values, 1);
  num_hyperparameters = length(hypersamples.marginal_ind);
  
  for i = 1:num_hypersamples
    gp.hypersamples(i).hyperparameters = ...
        hypersamples.values(i, hypersamples.marginal_ind);
  end
  
  for i = 1:num_hyperparameters
    gp.hyperparams(i).priorMean = hypersamples.prior_means(i);
    gp.hyperparams(i).priorSD = sqrt(hypersamples.prior_variances(i));
  end
  
  weights_mat = bq_params(gp, quad_gp);
  
  for i = 1:num_hypersamples
    gp.hypersamples(i).logL = hypersamples.log_likelihoods(i);
  end
  
  hypersample_weights = weights(gp, weights_mat);

end