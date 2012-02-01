function [post nlZ dnlZ] = infExactFault(hyp, mean, cov, lik, a, b, fault_cov, x, y)
% Exact inference for a GP with Gaussian likelihood. Compute a parametrization
% of the posterior, the negative log marginal likelihood and its derivatives
% w.r.t. the hyperparameters. See also "help infMethods".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18
%
% See also INFMETHODS.M.

likstr = lik; 
if (~ischar(lik))
  likstr = func2str(lik); 
end 

% NOTE: no explicit call to likGauss
if (~strcmp(likstr,'likGauss'))
  error('Exact inference only possible with Gaussian likelihood');
end
 
n = size(x, 1);

K = feval(cov{:}, hyp.cov, x);                      % evaluate covariance matrix
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector

a = feval(a{:}, hyp.a, x);
A = diag(a);              % evaluate A transformation matrix
Ad = diag(a-1);
b = feval(b{:}, hyp.b, x);                    % evaluate b transformation vector

% fault covariance
AKA = A * K * A;

% noise variance of likGauss
sn2 = exp(2 * hyp.lik);

% covariance of potentially faulty observation likelihood
fault_K = feval(fault_cov{:}, hyp.fault_covariance_function, x);

m = feval(mean{:}, hyp.mean, x);

% Cholesky factor of covariance with noise
L = chol((fault_K + K + Ad * K * Ad) / sn2 + eye(n)) ;
alpha = solve_chol(L, (y - (A * m + b))) / sn2;

post.alpha = alpha;                            % return the posterior parameters
post.sW = ones(n, 1) / sqrt(sn2);               % sqrt of noise precision vector
post.L = L;                                         % L = chol(eye(n)+sW*sW'.*K)

% do we want the marginal likelihood?
if (nargout > 1)
  nlZ = (y - (A * m + b))' * (alpha / 2) + ...
        sum(log(diag(L))) + n * log(2 * pi * sn2) / 2;
  % do we want derivatives?
  if (nargout > 2)
    % allocate space for derivatives
    dnlZ = hyp; 
    Q = solve_chol(L, eye(n)) / sn2 - alpha * alpha';
    for i = 1:numel(hyp.cov)
      dnlZ.cov(i) = sum(sum(Q .* feval(cov{:}, hyp.cov, x, [], i))) / 2;
    end
    dnlZ.lik = sn2 * trace(Q);
    for i = 1:numel(hyp.mean), 
      dnlZ.mean(i) = -feval(mean{:}, hyp.mean, x, i)' * alpha;
    end
  end
end
