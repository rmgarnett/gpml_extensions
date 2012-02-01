data_directory = '~/work/data/astronomy/quasars/processed/';
load([data_directory 'quasars.mat']);

train_x = wavelengths(:);

train_y = data(end, :);
train_y = train_y(:);

test_x = train_x;

fault_shape      = linspace(-1, 1).^2 - 1;
fault_scale      = 20;
fault_start_time = 1220;
fault_end_time   = 1235;

length_scale = log(5);
output_scale = log(1);

noise_scale = log(1);

inference_method = @infExactFault;
mean_function = @meanConst;
covariance_function = ...
    {@covSum, {@covSEiso, {@covDrift, {@covSEiso}}}};
likelihood = @likGauss;

a_function = @meanOne;
b_function = {@meanScale, {@meanDrift, fault_shape}};

hyperparameters.mean = mean(train_y);
hyperparameters.cov = ...
     [length_scale; output_scale; ...
      fault_start_time; fault_end_time; length_scale; output_scale];
hyperparameters.lik = noise_scale;

hyperparameters.a = [];
hyperparameters.b = [fault_scale; fault_start_time; fault_end_time];

[hyperparameters inference_method mean_function covariance_function ...
 likelihood a_function b_function] = ...
    check_gp_fault_arguments(hyperparameters, inference_method, ...
                             mean_function, covariance_function, ...
                             likelihood, a, b, train_x);

[output_means output_variances latent_means latent_variances] = ...
    gp_fault(hyperparameters, inference_method, mean_function, ...
             covariance_function, likelihood, a_function, b_function, ...
             train_x, train_y, test_x);

make_gp_plot(train_x, latent_means, sqrt(latent_variances), train_x, ...
             train_y, [min(wavelengths) max(wavelengths) 0 35], 7, ...
             'wavelength', 'energy', 'SouthWest', 25, 6);
title('latent function -- good');

make_gp_plot(train_x, output_means, sqrt(output_variances), train_x, ...
             train_y, [min(wavelengths) max(wavelengths) 0 35], 7, ...
             'wavelength', 'energy', 'SouthWest', 25, 6);
title('outputs');

negative_log_likelihood = ...
    gp_fault(hyperparameters, inference_method, mean_function, ...
             covariance_function, likelihood, a, b, train_x, ...
             train_y);

disp(['likelihood: ' num2str(-negative_log_likelihood)]);
