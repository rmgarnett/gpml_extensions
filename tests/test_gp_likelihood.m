number_of_runs = 5;

sizes = [10 20 50 100 200 500 1000 2000 5000];

for size = sizes
  data = linspace(0, 100, size)';
  responses = sin(data(:));

  inference_method = @infExact;
  mean_function = @meanZero;
  covariance_function = @covSEiso;
  likelihood = @likGauss;

  hyperparameters.lik = log(1e-2);
  hyperparameters.cov = [log(pi); log(1)];

  [hyperparameters, inference_method, mean_function, ...
   covariance_function, likelihood] = ...
      check_gp_arguments(hyperparameters, inference_method, ...
                         mean_function, covariance_function, likelihood, ...
                         data, responses);

  tic;
  for i = 1:number_of_runs
    [nlZ dnlZ post] = gp_likelihood(hyperparameters, inference_method, ...
                                    mean_function, covariance_function, ...
                                    likelihood, data, responses);
  end
  elapsed = toc;

  disp('gp_likelihood answer: ');
  disp(['               log likelihood: ' num2str(-nlZ)]);
  for i = 1:length(dnlZ.cov)
    disp(['  log likelihood derivative ' num2str(i) ': ' num2str(dnlZ.cov(i))]);
  end
  disp(['                   total time: ' num2str(elapsed) 's.']);
  disp(['                 average time: ' num2str(elapsed / number_of_runs) 's.']);

  tic
  for i = 1:number_of_runs
    [nlZ dnlZ] = gp(hyperparameters, inference_method, ...
                    mean_function, covariance_function, ...
                    likelihood, data, responses);
  end
  elapsed = toc;

  disp('gp answer: ');
  disp(['               log likelihood: ' num2str(-nlZ)]);
  for i = 1:length(dnlZ.cov)
    disp(['  log likelihood derivative ' num2str(i) ': ' num2str(dnlZ.cov(i))]);
  end
  disp(['                   total time: ' num2str(elapsed) 's.']);
  disp(['                 average time: ' num2str(elapsed / number_of_runs) 's.']);
end