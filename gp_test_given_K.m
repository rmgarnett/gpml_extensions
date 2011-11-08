function [varargout] = gp_test_given_K(hyp, inf, cov, mean, lik, K, ...
        train, test, x, y, xs, ys)
% Calculate test set predictive probabilities given a specified
% Gaussian process prior, trianing data, and test points. This
% function assumes that the prior covariance is already known, 
% which might be useful, for example, when the training and test
% set will always come from a fixed set of points, thereby avoiding
% recomputing K for every call.
%
% If desired, the negative log marginal likelihood of a Gaussian
% process and its partial derivatives with respect to the
% hyperparameters may be returned as well.
%
% If the full posterior covariance is desired, you may set:
%   hyp.full_covariance = true
%
% Please call check_gp_arguments before calling this function.
%
% Usage:
%   [ymu ys2 fmu fs2 lp nlZ dnlZ] = ... 
%     gp_test_given_K(hyp, inf, mean, lik, K, train_ind, test_ind, y, ys);
%
% where:
%
%   hyp      column vector of hyperparameters
%   inf      function specifying the inference method 
%   cov      prior covariance function (see below)
%   mean     prior mean function
%   lik      likelihood function
%   K        an n x n prior covariance matrix
%   train    index into K marking which are the training points
%   test     index into K marking which are the test points
%   x        n by D matrix of training inputs
%   y        column vector of length n of training targets
%   xs       ns by D matrix of test inputs
%   ys       column vector of length nn of test targets
%
%   ymu      column vector (of length ns) of predictive output means
%   ys2      column vector (of length ns) of predictive output variances
%   fmu      column vector (of length ns) of predictive latent means
%   fs2      column vector (of length ns) of predictive latent variances
%   lp       column vector (of length ns) of log predictive probabilities
%   nlZ      returned value of the negative log marginal likelihood
%   dnlZ     column vector of partial derivatives of the negative
% 
% See also covFunctions.m, infMethods.m, likFunctions.m, meanFunctions.m.
%
% Based on code from:
%
% GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox version 3.1
%    for GNU Octave 3.2.x and Matlab 7.x
% Copyright (c) 2005-2010 Carl Edward Rasmussen & Hannes Nickisch. All
% rights reserved.
%
% Copyright (c) 2011 Roman Garnett. All rights reserved.

% call the inference method
 nlZ = [];
dnlZ = [];
try
  % no likelihood desired
  if (nargout < 6) 
               post = inf(hyp, mean, cov, lik, x, y);
  % likelihood desired
  elseif (nargout < 7) 
         [post nlZ] = inf(hyp, mean, cov, lik, x, y);
  % likelihood and derivatives desired
  else
    [post nlZ dnlZ] = inf(hyp, mean, cov, lik, x, y);
  end
catch msgstr
  disp('Inference method failed!');
  rethrow(msgstr);
end

alpha = post.alpha;
   sW = post.sW;
    L = post.L; 

% handle things for sparse representations
if (issparse(alpha))
  % determine nonzero indices
  nz = (alpha ~= 0);

  % convert L and sW if necessary
  if (issparse(L))
    L = full(L(nz, nz)); 
  end
  if (issparse(sW))
    sW = full(sW(nz)); 
  end
else 
  % non-sparse representation
  nz = true(size(alpha)); 
end

% number of data points
ns = size(xs, 1);

% in case L is not provided, we compute it
if (numel(L) == 0)
  L = chol(eye(ns) + sW * sW' .* K(train, train));
end

% is L an upper triangular matrix?
Ltril = all(all(tril(L, -1) == 0));

if (isfield(hyp, 'full_covariance') && hyp.full_covariance)
  % prior mean
  ms = feval(mean{:}, hyp.mean, xs);

  % cross-covariances
  Ks  = K(train, train);

  % predictive means
  fmu = ms + Ks' * alpha;

  % prior test covariance
  Kss = K(test, test);

  % predictive covariance
  V  = L' \ (repmat(sW, 1, size(xs, 1)) .* Ks);
  fs2 = Kss - V' * V;

  % assign output arguments
  if (nargin < 9)
    [ ~, ymu, ys2] = lik(hyp.lik, [], fmu, diag(fs2));
    varargout = {ymu, ys2, fmu, fs2, [], nlZ, dnlZ};
  else
    [lp, ymu, ys2] = lik(hyp.lik, ys, fmu, diag(fs2));
    varargout = {ymu, ys2, fmu, fs2, lp, nlZ, dnlZ};
  end

  return;
else

  % allocate outputs
  ymu = zeros(ns, 1); 
  ys2 = zeros(ns, 1);
  fmu = zeros(ns, 1);
  fs2 = zeros(ns, 1);
   lp = zeros(ns, 1);
  
  % number of data points per mini batch
  nperbatch = 5000;
  % number of already processed test data points
  nact = 0;
  
  diag_K = diag(K);
  
  % process minibatches of test cases to save memory
  while (nact < ns)
    % data points to process
    id = (nact + 1):min(nact + nperbatch, ns);
    
    % cross-covariances
    Ks  = K(train(nz), test(id));
    % self-variance
    kss = diag_K(id);
    
    % prior mean
    ms = feval(mean{:}, hyp.mean, xs(id, :));
    
    % predictive means
    fmu(id) = ms + Ks' * full(alpha(nz));
    
    % predictive variances
    if (Ltril)
      % L is triangular => use Cholesky parameters (alpha, sW, L)
      V  = L' \ (repmat(sW, 1, length(id)) .* Ks);
      if (nargout > 2)
        % predictive variances
        fs2(id) = kss - sum(V .* V, 1)';
      end
    else
      % L is not triangular => use alternative parametrisation
      fs2(id) = kss + sum(Ks .* (L * Ks), 1)';
    end
    % remove numerical noise i.e. negative variances
    fs2(id) = max(fs2(id), 0);
    
    if (nargin < 9)
      [lp(id) ymu(id) ys2(id)] = lik(hyp.lik,     [], fmu(id), fs2(id));
    else
      [lp(id) ymu(id) ys2(id)] = lik(hyp.lik, ys(id), fmu(id), fs2(id));
    end
    
    % set counter to index of last processed data point  
    nact = id(end);
  end
  
  % assign output arguments
  varargout = {ymu, ys2, fmu, fs2, lp, nlZ, dnlZ};
  return;
end

