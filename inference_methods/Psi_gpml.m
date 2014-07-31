% The below was an internal function in GPML's infLaplace, which we
% reuse unmodified for laplace_inference.

% Evaluate criterion Psi(alpha) = alpha'*K*alpha + likfun(f), where
% f = K*alpha+m, and likfun(f) = feval(lik{:},hyp.lik,y,  f,  [],inf).
function [psi,dpsi,f,alpha,dlp,W] = Psi_gpml(alpha,m,K,likfun)
  f = K*alpha+m;
  [lp,dlp,d2lp] = likfun(f); W = -d2lp;
  psi = alpha'*(f-m)/2 - sum(lp);
  if nargout>1, dpsi = K*(alpha-dlp); end
end