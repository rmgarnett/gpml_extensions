% CHOL_DELETE updates a Cholesky factorization after deleting one row/column.
%
%

function new_L = chol_delete(L, ind)

  % % remove ind'th column
  % L = L(:, [1:(ind - 1), (ind + 1):end]);

  % n = size(L, 2);

  % % use a series of Given's rotation to remove entries below diagonal
  % for column = ind:n
  %   rows = [column, (column + 1)];
  %   L(rows, column:end) = planerot(L(rows, column)) * L(rows, column:end);
  % end

  % % remove bottom row, now all zeros
  % L = L(1:n, :);

  v = L(ind, (ind + 1):end);

  new_L = zeros(size(L, 1) - 1);
  new_L(1:(ind - 1), :) = L(1:(ind - 1), :);
  new_L(ind:end, ind:end) = ...
      chol(v' * v

end