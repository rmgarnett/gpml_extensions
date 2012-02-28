% Based on code from:
%
% GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 3.1
%    for GNU Octave 3.2.x and Matlab 7.x
% Copyright (c) 2005-2010 Carl Edward Rasmussen & Hannes Nickisch. All
% rights reserved.
%
% Copyright (c) 2012 Roman Garnett. All rights reserved.
%
% See also meanFunctions.m.

function result = meanConstVector(mean_vector, hyperparameters, train_ind, i)

  % number of hyperparameters
  if (nargin < 3)
    result = '0';

  % mean vector
  elseif (nargin == 3)
    result = mean_vector(train_ind);

  % derivatives wrt hyperparameters
  else
    result = zeros(numel(train_ind), 1);
end
