% test test_gp_full_covariance function

data = linspace(0, 100)';
responses = sin(data(:));
test = linspace(0, 100, 1000)';

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

[ymu ys2 fmu fs2 lp nlZ dnlZ] = ... 
    gp_test_full_covariance(hyperparameters, inference_method, ...
                            mean_function, covariance_function, likelihood, ...
                            data, responses, test);
