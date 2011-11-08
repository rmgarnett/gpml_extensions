function passed = test_gp_likelihood(hyperparameters, inference_method, ...
        mean_function, covariance_function, likelihood, data, ...
        responses, number_of_runs)

  tic
  for i = 1:number_of_runs
    [gpml_nlZ gpml_dnlZ] = gp(hyperparameters, inference_method, mean_function, ...
                              covariance_function, likelihood, data, responses);
  end
  gpml_elapsed = toc;

  tic;
  for i = 1:number_of_runs
    [gp_likelihood_nlZ gp_likelihood_dnlZ] = gp_likelihood(hyperparameters, ...
            inference_method, mean_function, covariance_function, ...
            likelihood, data, responses);
  end
  gp_likelihood_elapsed = toc;

  passed = (gpml_nlZ == gp_likelihood_nlZ) && ...
           all(unwrap(gpml_dnlZ) == unwrap(gp_likelihood_dnlZ));

  disp('gp answer: ');
  disp(['               log likelihood: ' num2str(-gpml_nlZ)]);
  for i = 1:length(gpml_dnlZ.cov)
    disp( ['  log likelihood derivative ' num2str(i) ': ' num2str(gpml_dnlZ.cov(i))]);
  end
  disp(['                   total time: ' num2str(gpml_elapsed) 's.']);
  disp(['                 average time: ' num2str(gpml_elapsed / number_of_runs) 's.']);

  disp('gp_likelihood answer: ');
  disp(['               log likelihood: ' num2str(-gp_likelihood_nlZ)]);
  for i = 1:length(gp_likelihood_dnlZ.cov)
    disp( ['  log likelihood derivative ' num2str(i) ': ' num2str(gp_likelihood_dnlZ.cov(i))]);
  end
  disp(['                   total time: ' num2str(gp_likelihood_elapsed) 's.']);
  disp(['                 average time: ' num2str(gp_likelihood_elapsed / number_of_runs) 's.']);

end