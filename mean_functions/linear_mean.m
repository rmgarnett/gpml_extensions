% Linear mean function. The mean function is parameterized as:
%
% m(x) = sum_i a_i * x_i;
%
% The hyperparameter is:
%
% hyp = [ a_1
%         a_2
%         ..
%         a_D ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-01-10.

% See also MEANFUNCTIONSx.

function result = linear_mean(hyperparameters, x, i, j)

  % report number of hyperparameters
  if (nargin <= 1)
    result = 'D';
    return;
  end

  num_points = size(x, 1);

  % mean
  if (nargin == 2)
    result = x * hyperparameters(:);

  % derivative
  elseif (nargin == 3)
    result = x(:, i);

  % second derivative
  elseif (nargin == 4)
    result = zeros(num_points, 1);
  end

end