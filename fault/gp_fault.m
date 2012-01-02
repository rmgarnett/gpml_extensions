% function varargout = gp_fault(hyperparameters, covariance_function, ...
%                               mean_function, likelihood, a_function, ...
%                               b_function, train_x, train_y, test_x, ...
%                               test_y)
%
% inputs:
%
%      inference_method: function specifying the inference method
%   covariance_function: prior covariance function (see below)
%         mean_function: prior mean function
%            likelihood: likelihood function
%            a_function: the a(x) function in a(x)y + b(x). this
%                        should be a valid GPML mean function.
%            b_function: the a(x) function in a(x)y + b(x). this
%                        should be a valid GPML mean function.
%               train_x: training x points
%               train_y: training y points
%                test_x: test x points
%                test_y: test y points
%
% outputs:
%
%              negative_log_marginal_likelihood: negative log
%                                                marginal likelihood
%  negative_log_marginal_likelihood_derivatives: derivatives of negative
%                                                log marginal likelihood
%                                                with respect to
%                                                hyperparameters
%                                   output_mean: predictive output mean
%                               output_variance: predictive output variance
%                                   latent_mean: predictive latent mean
%                               latent_variance: predictive latent variance
%                                    fault_mean: predictive fault mean
%                                fault_variance: predictive fault variance
%                             log_probabilities: log predictive probabilities
%                                     posterior: struct containing
%                                                the approximate posteior
%
% See also covFunctions.m, infMethods.m, likFunctions.m, meanFunctions.m.
%
% Copyright (c) 2011 Roman Garnett.  All rights reserved.

function varargout = gp_fault(hyperparameters, covariance_function, ...
                              mean_function, likelihood, a_function, ...
                              b_function, train_x, train_y, test_x, ...
                              test_y)

% diagonal A transformation matix
A = diag(feval(a_function{:}, hyp.a, x));

try
  % call the inference method and compute marginal likelihood and its
  % derivatives only if needed
  if (nargin > 9)
    posterior = ...
        inference_method(hyperparameters, mean_function, ...
                         covariance_method, likelihood, a_function, ...
                         b_function, x, y);
  else
    if (nargout == 1)
      [posterior, negative_log_marginal_likelihood] = ...
          inference_method(hyperparameters, mean_function, ...
                           covariance_method, likelihood, a_function, ...
                            b_function, x, y);
       marginal_likelihood_derivatives = {};
    else
      [posterior, negative_log_marginal_likelihood, ...
       negative_log_marginal_likelihood_derivatives] = ...
          inference_method(hyperparameters, mean_function, ...
                           covariance_method, likelihood, a_function, ...
                           b_function, x, y);
    end
  end
catch
  msgstr = lasterr;
  if (nargin > 9)
    error('gpml_extensions:inference method failed [%s]', msgstr);
  else
    % continue with a warning
    warning(['gpml_extensions:inference method failed [%s] ... ' ...
             'attempting to continue'], msgstr);
    negative_log_marginal_likelihood_derivatives = ...
        struct('cov',  0 * hyperparameters.cov, ...
               'mean', 0 * hyperparameters.mean, ...
               'lik',  0 * hyperparameters.lik);
    varargout = {NaN, marginal_likelihood_derivatives};
    return;
  end
end

if (nargin == 9)
  % no test cases are provided report negative log marginal
  % likelihood, its derivatives, and the posterior struct
  varargout = {negative_log_marginal_likelihood, ...
               negative_logmarginal_likelihood_derivatives, posterior};
else
  alpha = posterior.alpha;
  L = posterior.L;
  sW = posterior.sW;

  % handle things for sparse representations
  if (issparse(alpha))
    % determine nonzero indices
    nz = (alpha ~= 0);
    % convert L and sW if necessary
    if (issparse(L))
      L = full(L(nz, nz));
    end
    if (issparse(sW))
      sW = full(sW(nz));
    end
  else
    % non-sparse representation
    nz = true(size(alpha));
  end

  % in case L is not provided, we compute it
  if (numel(L) == 0)
    K = feval(covariance_function{:}, hyperparameters.cov, train_x(nz, :));
    L = chol(eye(sum(nz)) + sW * sW' .* K);
  end

  % is L an upper triangular matrix?
  L_tril = all(all(tril(L, -1) == 0));

  % number of data points
  num_points = size(test_x, 1);
  % number of data points per mini batch
  num_per_batch = 5000;
  % number of already processed test data points
  num_processed = 0;

  % allowcate memory
  output_mean       = zeros(num_points, 1);
  output_variance   = zeros(num_points, 1);
  latent_mean       = zeros(num_points, 1);
  latent_variance   = zeros(num_points, 1);
  fault_mean        = zeros(num_points, 1);
  fault_variance    = zeros(num_points, 1);
  log_probabilities = zeros(num_points, 1);

  % process minibatches of test cases to save memory
  while (num_processed < num_points)
    % data points to process
    ind = (n_processed + 1):(min(num_points + num_per_batch, num_points);

    % self variances
    kss = feval(covariance_function{:}, hyperparameters.cov, test_x(nz, :) ...
                test_x(ind, :), 'diag');
    % cross covariances
    Ks = feval(covariance_function{:}, hyperparameters.cov, train_x(nz, :) ...
               test_x(ind,:));

    % prior mean
    prior_mean = feval(mean_function{:}, hyperparameters.mean, ...
                       test_x(ind, :));

    % posterior latent mean
    latent_mean(ind) = prior_mean + Ks' * full(alpha(nz));

    if (Ltril)
      % L is triangular => use Cholesky parameters (alpha, sW, L)
      V = L' \ (repmat(sW, 1, length(ind)) .* (A * Ks));
      latent_variance(ind) = kss - sum(V .* V, 1)';
    else
      % L is not triangular => use alternative parametrisation
      latent_variance(ind) = kss + sum(Ks .* (L * Ks), 1)';
    end

    if (nargin < 11)
      [log_probabilites(ind), output_means(ind), output_variance(ind)] = ...
          likelihood(hyperparameters.lik, [], latent_mean(ind), ...
                     latent_variance(ind));
    else
      [log_probabilites(ind), output_means(ind), output_variance(ind)] = ...
          likelihood(hyperparameters.lik, test_y, latent_mean(ind), ...
                     latent_variance(ind));
    end
    % set counter to index of last processed data point
    num_processed = ind(end);
  end

  % remove numerical noise i.e. negative variances
  latent_variance = max(latent_variance, 0);

  % assign output arguments
  if (nargin < 10)
    varargout = {output_mean, output_variance, latent_mean, ...
                 latent_variance, fault_mean, fault_variance, [], posterior};
  else
    varargout = {output_mean, output_variance, latent_mean, ...
                 latent_variance, fault_mean, fault_variance, ...
                 log_probabilities, posterior};
  end
end
