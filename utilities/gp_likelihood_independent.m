% Computes the negative log likelihood and its derivatives given sets
% of observations that are supposed to have been drawn independently
% from the same Gaussian process prior.

function [nlZ, dnlZ] = gp_likelihood_independent(hyperparameters, ...
          inference_method, mean_function, covariance_function, ...
          likelihood, xs, ys)

  num_samples = numel(xs);

  % initialize nlZ and, optionally, dnlZ struct
  nlZ = 0;
  if (nargout == 2)
    dnlZ = rewrap(hyperparameters, 0 * unwrap(hyperparameters));
  end

  for i = 1:num_samples
    if (nargout == 1)
      this_nlZ = ...
          gp(hyperparameters, inference_method, mean_function, ...
             covariance_function, likelihood, xs{i}, ys{i});
    else
      [this_nlZ, this_dnlZ] = ...
          gp(hyperparameters, inference_method, mean_function, ...
             covariance_function, likelihood, xs{i}, ys{i});
    end

    % accumulate likelihoods and derivatives
    nlZ = nlZ + this_nlZ;
    if (nargout == 2)
      dnlZ = rewrap(dnlZ, unwrap(dnlZ) + unwrap(this_dnlZ));
    end
  end

end