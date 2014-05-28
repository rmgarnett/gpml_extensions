% SCALED_COVARIANCE meta-covariance for K' = a(x) K(x, x') a(x')
%
% This provides a GPML-compatible meta-covariance implementing a
% properly scaled covariance function for modeling functions of the
% form
%
%   g(x) = a(x) f(x),
%
% where a(x) is a given function and f ~ GP(mu, K). The resulting
% covariance has the form
%
%   K'(x, x') = a(x) K(x, x') a(x').
%
% In this implementation, the scaling function a(x) is specified as a
% (possibly parameterized) GPML mean function. This implementation
% also computes the derivatives of the covariance function with
% respect to both the parameters of a(x) and K(x, x').
%
% The hyperparameters should be specified as:
%
% hyperparameters = [    scaling_function hyperparameters(:)
%                     covariance_function hyperparameters(:) ];
%
% See also COVFUNCTIONS.

% Copyright (c) 2012--2014 Roman Garnett.

function result = scaled_covariance(scaling_function, covariance_function, ...
          hyperparameters, x, z, i)

  % check for scaling and covariance functions
  if (nargin < 2)
    error('gpml_extensions:missing_arguments', ...
          'scaling_function and covariance_function must be specified!');
  end

  % report number of hyperparameters
  if (nargin <= 3)
    result = ['((' feval(scaling_function{:}) ') + (' feval(covariance_function{:}) '))'];
    return;
  end

  % find dimension of input data; this must be "D" due to the way
  % GPML reports hyperparameter counts
  D = size(x, 2);

  num_scaling_hyperparameters = eval(feval(scaling_function{:}));

  scaling_hyperparameters    = hyperparameters(1:num_scaling_hyperparameters);
  covariance_hyperparameters = hyperparameters((num_scaling_hyperparameters + 1):end);

  scaling = @(varargin) ...
            feval(scaling_function{:}, scaling_hyperparameters, varargin{:});

  covariance = @(varargin) ...
               feval(covariance_function{:}, covariance_hyperparameters, varargin{:});

  % calculates diag(a) * K * diag(b) for a, b column vectors
  % [a: (n x 1), b: (m x 1), K: (n x m)]
  scale_matrix = @(a, K, b) bsxfun(@times, a, bsxfun(@times, K, b'));

  % create empty z if it does not exist
  if (nargin == 4)
    z = [];
  end

  % determine mode
  training = isempty(z);
  diagonal = (strcmp(z, 'diag'));
  gradient_mode = (nargin == 6);

  % training set scaling vector
  scaling_x = scaling(x);

  % vector kxx
  if (diagonal)

    % diagonal of training covariance
    if (~gradient_mode)
      result = scaling_x.^2 .* covariance(x, 'diag');

    % gradient of diagonal of training covariance
    else

      % scaling hyperparameters
      if (i <= num_scaling_hyperparameters)
        result = 2 * scaling_x .* scaling(x, i) .* covariance(x, 'diag');

      % covariance hyperparameters
      else
        i = i - num_scaling_hyperparameters;
        result = scaling_x.^2 .* covariance(x, 'diag', i);
      end
    end

  % symmetric matrix Kxx
  else
    if (training)
      % training covariance
      if (~gradient_mode)
        result = scale_matrix(scaling_x, covariance(x), scaling_x);

      % gradient of training covariance
      else

        % scaling hyperparameters
        if (i <= num_scaling_hyperparameters)
          result = scale_matrix(scaling_x, covariance(x), scaling(x, i));
          result = result + result';

        % covariance hyperparameters
        else
          i = i - num_scaling_hyperparameters;
          result = scale_matrix(scaling_x, covariance(x, [], i), scaling_x);
        end
      end

    % cross covariances Kxz
    else

      % cross covariance matrix
      if (~gradient_mode)
        result = scale_matrix(scaling_x, covariance(x, z), scaling(z));

      % derivatives of cross covariance matrix
      else
        % scaling hyperparameters
        if (i <= num_scaling_hyperparameters)
          K = covariance(x, z);
          result = scale_matrix(scaling_x,     K, scaling(z, i)) + ...
                   scale_matrix(scaling(x, i), K, scaling(z));

        % covariance hyperparameters
        else
          i = i - num_scaling_hyperparameters;
          result = scale_matrix(scaling_x, covariance(x, z, i), scaling(z));
        end
      end
    end
  end

end