% determine whether L is a Cholesky decomposition

function result = is_chol(L)

  result = (isreal(diag(L))  && ...
            all(diag(L) > 0) && ...
            all(all(tril(L, -1) == 0)));

end