% The below was an internal function in GPML's infLaplace, which we
% reuse unmodified for laplace_inference.

% Compute the log determinant ldA and the inverse iA of a square nxn matrix
% A = eye(n) + K*diag(w) from its LU decomposition; for negative definite A, we
% return ldA = Inf. We also return mwiA = -diag(w)/A.
function [ldA,iA,mwiA] = logdetA_gpml(K,w)
  [m,n] = size(K); if m~=n, error('K has to be nxn'), end
  A = eye(n)+K.*repmat(w',n,1);
  [L,U,P] = lu(A); u = diag(U);           % compute LU decomposition, A = P'*L*U
  signU = prod(sign(u));                                             % sign of U
  detP = 1;                 % compute sign (and det) of the permutation matrix P
  p = P*(1:n)';
  for i=1:n                                                       % swap entries
    if i~=p(i), detP = -detP; j = find(p==i); p([i,j]) = p([j,i]); end
  end
  if signU~=detP  % log becomes complex for negative values, encoded by infinity
    ldA = Inf;
  else            % det(L) = 1 and U triangular => det(A) = det(P)*prod(diag(U))
    ldA = sum(log(abs(u)));
  end
  if nargout>1, iA = U\(L\P); end               % return the inverse if required
  if nargout>2, mwiA = -repmat(w,1,n).*iA; end
end