% provides a gpml-compatiable likelihood function for
% poisson-distributed observations. this can be used to bulld a
% log-gaussian cox model for fitting a nonhomogeneous poisson process
% to data. the likelihood is given by
%
% p(y | f) = \prod_i Poisson(y_i | exp(f_i)).

function [varargout] = likPoisson(hyperparameters, observations, ...
        latent_means, latent_variances, inference_method, hyperparameter_ind)

  % report number of hyperparameters
  if (nargin < 2)
    varargout = { '0' };
    return;
  end

  num_points = numel(latent_means);

  % create dummy observations if needed
  if (numel(observations) == 0)
    observations = ones(num_points, 1);
  end

  % evaluates log(p(y | log(\lambda)))
  log_poisson_likelihood = @(observations, log_lambda) ...
      observations .* log_lambda - ...
      exp(log_lambda) - ...
      gammaln(observations + 1);

  % evaluates log(N(log(\lambda); \mu, \sigma^2))
  log_gaussian_likelihood = @(log_lambda, latent_mean, latent_std) ...
      (-0.5 * ((log_lambda - latent_mean) ./ latent_std).^2 - ...
       log(latent_std) - ...
       log(2 * pi) / 2);

  % evaluates Poisson(y; \lambda) N(log(\lambda); \mu, \sigma^2)
  likelihood = ...
      @(observations, log_lambda, latent_mean, latent_std) ...
      exp(log_poisson_likelihood(observations, log_lambda) + ...
          log_gaussian_likelihood(log_lambda, latent_mean, latent_std));
  
  % prediction mode
  if (nargin < 5)

    % check for the case that only latent means have been passed in
    empty_variances = ...
        ((numel(latent_variances) == 0) || ...
         ((numel(latent_variances) == 1) && (latent_variances == 0)));
    
    % provide p(y | log(\lambda) = \mu)
    if (empty_variances)
      lambda = exp(latent_means);
      log_probabilities = log_poisson_likelihood(observations, lambda);
      varargout = {log_probabilities};
    else

      log_probabilities = zeros(num_points, 1);
      if (nargout > 1)
        observation_means = zeros(num_points, 1);
        if (nargout > 2)
          observation_variances = zeros(num_points, 1);
        else
          observation_variances = [];
        end
      else
        observation_means = [];
      end
      
      for i = 1:num_points
        latent_mean     = latent_means(i);
        latent_variance = latent_variances(i);
        latent_std      = sqrt(latent_variance);

        lower_limit = latent_mean - 6 * latent_std;
        upper_limit = latent_mean + 6 * latent_std;

        % calculates p(y | x, D) as a function of y at the given point
        predictive_distribution = ...
            @(y) ...
            quadgk(@(log_lambda) ...
                   likelihood(y, log_lambda, latent_mean, latent_std), ...
                   lower_limit, upper_limit);
        
        log_probabilities(i) = log(predictive_distribution(observations(i)));

        % evaluates E[y | x, D] and Var[y | x, D] at this point
        % (using analytic results for log normal distributions)
        if (nargout > 1)
          % E[y | x, D]
          observation_means(i) = exp(latent_mean + latent_variance / 2); 
          if (nargout > 2)
            % Var[y | x, D]
            observation_variances(i) = ...
                observation_means(i) + ...
                (exp(latent_variance) - 1) * exp(2 * latent_mean + latent_variance);
          end
        end
      end
      varargout = {log_probabilities, observation_means, observation_variances};
    end

  % inference mode
  else
    switch (inference_method)
      case 'infLaplace'
        % calculates first second and third derivatives of the log
        % predictive distribution, log(p(y | log(\lambda))) with respect
        % to log(\lambda)
        if (nargin == 5)
          log_probabilities = ...
              likPoisson(hyperparameters, observations, latent_means, []);

          first_derivative  = [];
          second_derivative = [];
          third_derivative  = [];
          
          if (nargout > 1)
            first_derivative = observations - exp(latent_means);
            if (nargout > 2)
              second_derivative = first_derivative - observations;
              if (nargout > 3)
                third_derivative = second_derivative;
              end
            end
          end
          varargout = {sum(log_probabilities), first_derivative, ...
                       second_derivative, third_derivative};

        % calculates derivatives with respect to hyperparamters
        % (there are none)
        else
          varargout = {[]};
        end
    end
  end
end
