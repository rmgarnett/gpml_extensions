% Based on code from:
%
% GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 3.1
%    for GNU Octave 3.2.x and Matlab 7.x
% Copyright (c) 2005-2010 Carl Edward Rasmussen & Hannes Nickisch. All
% rights reserved.
%
% Copyright (c) 2012 Roman Garnett. All rights reserved.
%
% See also covFunctions.m.

function result = covConstMatrix(K, hyperparameters, train_ind, test_ind, i)

  % number of hyperparameters
  if (nargin < 3)
    result = '0';

  % training covariance
  elseif (nargin == 3)
    result = K(train_ind, train_ind);

  % diagonal training variance
  elseif ((nargin == 4) && (strcmp(test_ind, 'diag')))
    diagonal = diag(K);
    result = diagonal(train_ind);

  % test covariance
  elseif (nargin == 4)
    result = K(train_ind, test_ind);

  % training derivatives
  elseif (numel(test_ind) == 0)
    result = zeros(numel(train_ind), numel(train_ind));

  % diagonal training derivatives
  elseif (strcmp(i, 'diag'))
    result = zeros(numel(train_ind), 1);

  % test derivatives
  else
    result = zeros(numel(train_ind), numel(test_ind));
  end

end