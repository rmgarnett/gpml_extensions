% The below was an internal function in GPML's infLaplace, which we
% reuse unmodified for laplace_inference.

% Run IRLS Newton algorithm to optimise Psi(alpha).
function alpha = irls_gpml(alpha, m,K,likfun, opt)
  if isfield(opt,'irls_maxit'), maxit = opt.irls_maxit; % max no of Newton steps
  else maxit = 20; end                                           % default value
  if isfield(opt,'irls_Wmin'),  Wmin = opt.irls_Wmin; % min likelihood curvature
  else Wmin = 0.0; end                                           % default value
  if isfield(opt,'irls_tol'),   tol = opt.irls_tol;     % stop Newton iterations
  else tol = 1e-6; end                                           % default value

  smin_line = 0; smax_line = 2;           % min/max line search steps size range
  nmax_line = 10;                          % maximum number of line search steps
  thr_line = 1e-4;                                       % line search threshold
  Psi_line = @(s,alpha,dalpha) Psi_gpml(alpha+s*dalpha, m,K,likfun); % line search
  pars_line = {smin_line,smax_line,nmax_line,thr_line};  % line seach parameters
  search_line = @(alpha,dalpha) brentmin(pars_line{:},Psi_line,5,alpha,dalpha);

  f = K*alpha+m; [lp,dlp,d2lp] = likfun(f); W = -d2lp; n = size(K,1);
  Psi_new = Psi_gpml(alpha,m,K,likfun);
  Psi_old = Inf;  % make sure while loop starts by the largest old objective val
  it = 0;                          % this happens for the Student's t likelihood
  while Psi_old - Psi_new > tol && it<maxit                       % begin Newton
    Psi_old = Psi_new; it = it+1;
    % limit stepsize
    W = max(W,Wmin); % reduce step size by increasing curvature of problematic W
    sW = sqrt(W); L = chol(eye(n)+sW*sW'.*K);            % L'*L=B=eye(n)+sW*K*sW
    b = W.*(f-m) + dlp;
    dalpha = b - sW.*solve_chol(L,sW.*(K*b)) - alpha; % Newton dir + line search
    [s_line,Psi_new,n_line,dPsi_new,f,alpha,dlp,W] = search_line(alpha,dalpha);
  end                                                  % end Newton's iterations
end