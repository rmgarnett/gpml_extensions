% Based on code from:
%
% GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 3.1
%    for GNU Octave 3.2.x and Matlab 7.x
% Copyright (c) 2005-2010 Carl Edward Rasmussen & Hannes Nickisch. All
% rights reserved.
%
% Copyright (c) 2012 Roman Garnett. All rights reserved.
%
% See also covFunctions.m in the GPML toolkit.

function K = covScaling(scaling_function, covariance_function, ...
                        hyperparameters, x, z, hyperparameter_ind)

  % report number of hyperparameters
  if (~exist('x', 'var'))
    K = [feval(scaling_function{:}) '+' feval(covariance_function{:})];
    return;
  end

  % find dimension of input data; this must be "D" due to the way
  % GPML reports hyperparameter counts
  D = size(x, 2);
  
  num_scaling_hyperparameters    = eval(feval(scaling_function{:}));

  scaling_hyperparameters    = hyperparameters(1:num_scaling_hyperparameters);
  covariance_hyperparameters = hyperparameters((num_scaling_hyperparameters + 1):end);
  
  % make sure z exists
  if (~exist('z', 'var'))
    z = [];
  end

  % determine mode
  training = isempty(z);
  diagonal = (strcmp(z, 'diag'));
  derivatives = exist('hyperparameter_ind', 'var');
  
  % training set scaling vector
  scaling_x = feval(scaling_function{:}, scaling_hyperparameters, x);

  % vector kxx
  if (diagonal)

    % diagonal of training covariance
    if (~derivatives)
      K = feval(covariance_function{:}, covariance_hyperparameters, x, 'diag');
      K = K .* scaling_x.^2;

    % derivatives of diagonal of training covariance
    else

      % scaling hyperparameters
      if (hyperparameter_ind <= num_scaling_hyperparameters)
        K = feval(covariance_function{:}, covariance_hyperparameters, x, 'diag');
        scaling_derivative = feval(scaling_function{:}, ...
                                   scaling_hyperparameters, x, hyperparameter_ind);
        K = K .* (2 * scaling_x .* scaling_derivative);

      % covariance hyperparameters
      else
        K = feval(covariance_function{:}, covariance_hyperparameters, ...
                  x, 'diag', hyperparameter_ind - num_scaling_hyperparameters);
        K = K .* scaling_x.^2;
      end
    end

  % symmetric matrix Kxx
  else
    if (training)
      % training covariance
      if (~derivatives)
        K = feval(covariance_function{:}, covariance_hyperparameters, x);
        K = bsxfun(@times, scaling_x, bsxfun(@times, scaling_x', K));

      % derivatives of training covariance
      else

        % scaling hyperparameters
        if (hyperparameter_ind <= num_scaling_hyperparameters)
          K = feval(covariance_function{:}, covariance_hyperparameters, x);
          scaling_derivative = feval(scaling_function{:}, ...
                                     scaling_hyperparameters, x, hyperparameter_ind);
          K = bsxfun(@times, scaling_x, bsxfun(@times, scaling_derivative', K)) + ...
              bsxfun(@times, scaling_derivative, bsxfun(@times, scaling_x', K));

        % covariance hyperparameters
        else
          K = feval(covariance_function{:}, covariance_hyperparameters, ...
                    x, [], hyperparameter_ind - num_scaling_hyperparameters);
          K = bsxfun(@times, scaling_x, bsxfun(@times, scaling_x', K));
        end
      end

    % cross covariances Kxz
    else

      % test set scaling vector
      scaling_z = feval(scaling_function{:}, scaling_hyperparameters, z);

      % cross covariance matrix
      if (~derivatives)
        K = feval(covariance_function{:}, covariance_hyperparameters, x, z);
        K = bsxfun(@times, scaling_x, bsxfun(@times, scaling_z', K));

      % derivatives of cross covariance matrix
      else
        % scaling hyperparameters
        if (hyperparameter_ind <= num_scaling_hyperparameters)
          K = feval(covariance_function{:}, covariance_hyperparameters, x, z);
          scaling_derivative_x = feval(scaling_function{:}, ...
                                       scaling_hyperparameters, x, hyperparameter_ind);
          scaling_derivative_z = feval(scaling_function{:}, ...
                                       scaling_hyperparameters, z, hyperparameter_ind);
          K = bsxfun(@times, scaling_x, bsxfun(@times, scaling_derivative_z', K)) + ...
              bsxfun(@times, scaling_derivative_x, bsxfun(@times, scaling_z', K));

        % covariance hyperparameters
        else
          K = feval(covariance_function{:}, covariance_hyperparameters, ...
                    x, z, hyperparameter_ind - num_scaling_hyperparameters);
          K = bsxfun(@times, scaling_x, bsxfun(@times, scaling_z', K));
        end
      end
    end
  end
end