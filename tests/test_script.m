hyperparameters.lik = log(1e-2);
hyperparameters.cov = [log(pi); log(1)];
   inference_method = @infExact;
      mean_function = @meanZero;
covariance_function = @covSEiso;
         likelihood = @likGauss;

              sizes = [10 20 50 100 200 500 1000];

for size = sizes
       data = linspace(0, 100, size)';
  responses = sin(data(:));

  [hyperparameters, inference_method, mean_function, ...
   covariance_function, likelihood] = ...
      check_gp_arguments(hyperparameters, inference_method, ...
                         mean_function, covariance_function, likelihood, ...
                         data, responses);
  
  passed = test_gp_likelihood(hyperparameters, inference_method, ...
                              mean_function, covariance_function, ...
                              likelihood, data, responses, ...
                              number_of_runs);
  if (passed)
    disp(['likelihood size ' num2str(size) ' passed!']);
  else
    disp(['likelihood size ' num2str(size) ' FAILED!']);
  end

       test_data = linspace(50, 200, size)';
  test_responses = sin(data(:));

  passed = test_gp_test(hyperparameters, inference_method, mean_function, ...
                        covariance_function, likelihood, data, ...
                        responses, test_data, test_responses, number_of_runs);

  if (passed)
    disp(['test size ' num2str(size) ' passed!']);
  else
    disp(['test size ' num2str(size) ' FAILED!']);
  end

end