function samples = find_ccd_points(prior_means, prior_variances)

  dimension = length(prior_means);

  samples = ccdesign(dimension, 'center', 1);
  samples = samples .* repmat(sqrt(prior_variances(:)'), size(samples, 1), 1);
  samples = samples  + repmat(prior_means(:)', size(samples, 1), 1);

end
