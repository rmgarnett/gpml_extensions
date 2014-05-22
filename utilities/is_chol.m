% determine whether L is a Cholesky decomposition

function result = is_chol(L)

  result = (isreal(diag(L))  && ...
            all(diag(L) > 0) && ...
            isequal(L, triu(L)));

end