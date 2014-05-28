% IS_CHOL determines whether L is a Cholesky decomposition.
%
% Given a matrix L, this function determines whether L is a Cholesky
% factorization; that is, whether LL' is a positive-definite
% matrix. The requirements are:
%
% - L is square
% - L is lower triangular
% - L has real and strictly positive diagonal entries
%
% Usage
% -----
%
%   result = is_chol(L)
%
% Input:
%
%   L: matrix to test
%
% Output:
%
%   result: a boolean indicating whether L is a Cholesky decomposition
%           of some matrix

% Copyright (c) 2014 Roman Garnett.
%
% This was adapted from code in the GPML toolbox:
%
% GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 3.4
%    for GNU Octave 3.2.x and Matlab 7.x
%
% Copyright (c) 2005--2013 Carl Edward Rasmussen & Hannes Nickisch.

function result = is_chol(L)

  result = (ismatrix(L) && ...
            (size(L, 1) == size(L, 2)) && ...
            isreal(diag(L))  && ...
            all(diag(L) > 0) && ...
            isequal(L, triu(L)));

end