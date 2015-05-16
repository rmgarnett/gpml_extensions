% LAPLACE_INFERENCE infLaplace replacement supporting additional derivatives.
%
% This provides a GPML-compatible inference method performing
% approximate inference via a Laplace approximation. This
% implementation supports an extended API allowing the calculation of:
%
% - the partial derivatives of \alpha with respect to \theata,
% - the partial derivatives of diag W^{-1} with respect to \theta, and
% - the Hessian of the negative log likelihood at \theta.
%
% The fomer two can be used to compute the gradient of the latent
% predictive mean and variance of the approximate posterior GP with
% respect to the hyperparameters.
%
% This can be used as a drop-in replacement for infLaplace with no
% extra computational cost.
%
% Usage
% -----
%
% The API is identical to GPML inference methods, expect for three
% additional optional arguments:
%
%   [posterior, nlZ, dnlZ, dalpha, dWinv, HnlZ] = ...
%       laplace_inference(theta, mean_function, covariance_function, ...
%                         likelihood, x, y);
%
% dalpha and dWinv provide the partial derivatives of the posterior
% parameters \alpha and W^{-1} with respect to \theta. These
% arragement of these structs is similar to the dnlZ struct. For
% example, dalpha.cov(:, 1) gives the derivative of \alpha with
% respect to the first covariance hyperparameter.
%
% HnlZ proivdes the Hessian of -\log p(y | X, \theta). See hessians.m
% for information regarding the Hessian struct HnlZ.
%
% Requirements
% ------------
%
% The posterior derivatives dalpha and dWinv can be used with
% unmodified GPML mean, covariance, and likelihood functions.
%
% To computate the Hessian HnlZ, both the mean and covariance
% functions must support an extended GPML syntax that allows for
% calculating the Hessian of the training mean mu or training
% covariance K with respect to any pair of hyperparameters. The syntax
% for mean functions is:
%
%   d2mu_didj = mean_function(theta, x, i, j);
%
% where d2mu_didj is \partial^2 mu(x) / \partial \theta_i \partial \theta_j.
%
% The syntax for covariance functions is similar:
%
%   d2K_didj = covariance_function(theta, x, [], i, j);
%
% where d2K_didj is \partial^2 K(x, x) / \partial \theta_i \partial \theta_j.
%
% Furthermore, the likelihood must also support an extended GPML
% syntax for calculating further derivatives; a summary is below.
%
% The likelihood syntax with five outputs:
%
%   [lp, dlp, d2lp, d3lp, d4lp] = ...
%           likelihood(theta, y, f, [], 'infLaplace');
%
% returns the additional value
%
%   d4lp = d^4 log p(y | f) / df^4
%
% The gradient syntax with four outputs:
%
%   [lp_dhyp, dlp_dhyp, d2lp_dhyp, d3lp_dhyp] = ...
%           likelihood(theta, y, f, [], 'infLaplace', i);
%
% returns the additional value
%
%   d3lp_dhyp = d^4 log p(y | f) / (df^3 d\theta_i)
%
% Finally, a new Hessian syntax:
%
%   [lp_dhyp2, dlp_dhyp2, d2lp_dhyp2] = ...
%           likelihood(theta, y, f, [], 'infLaplace', i, j);
%
% returns
%
%     lp_dhyp2 = d^2 log p(y | f) / (     d\theta_i d\theta_j)
%    dlp_dhyp2 = d^3 log p(y | f) / (df   d\theta_i d\theta_j)
%   d2lp_dhyp2 = d^4 log p(y | f) / (df^2 d\theta_i d\theta_j)
%
% See also INFMETHODS, HESSIANS.

% Copyright (c) 2013--2015 Roman Garnett.

function [posterior, nlZ, dnlZ, dalpha, dWinv, HnlZ] = ...
      laplace_inference(theta, mean_function, covariance_function, ...
          likelihood, x, y, options)

  persistent last_alpha;

  % ensure options argument exists
  if (nargin <= 6)
    options = [];
  end

  % If addditional outputs are not requested, simply call infLaplace and
  % return.
  if (nargout <= 3)
    if (nargout == 1)
      posterior = ...
          infLaplace(theta, mean_function, covariance_function, likelihood, x, y, options);

    elseif (nargout == 2)
      [posterior, nlZ] = ...
          infLaplace(theta, mean_function, covariance_function, likelihood, x, y, options);

    elseif (nargout == 3)
      [posterior, nlZ, dnlZ] = ...
          infLaplace(theta, mean_function, covariance_function, likelihood, x, y, options);
    end

    % remember alpha for next call
    last_alpha = posterior.alpha;
    return;
  end

  % determine what needs to be computed
  compute_dalpha = (nargout >= 4);
  compute_dWinv  = (nargout >= 5);
  compute_HnlZ   = (nargout >= 6);

  n = size(x, 1);
  I = eye(n);

  num_cov  = numel(theta.cov);
  num_lik  = numel(theta.lik);
  num_mean = numel(theta.mean);
  num_hyperparameters = num_cov + num_lik + num_mean;

  covariance_ind = 1:num_cov;
  likelihood_ind = (num_cov + 1):(num_cov + num_lik);
  mean_ind       = (num_cov + num_lik + 1):num_hyperparameters;

  % initialize output
  dnlZ = theta;

  if (compute_dalpha)
    dalpha.cov  = zeros(n, num_cov);
    dalpha.lik  = zeros(n, num_lik);
    dalpha.mean = zeros(n, num_mean);
  end

  if (compute_dWinv)
    dWinv.cov  = zeros(n, num_cov);
    dWinv.lik  = zeros(n, num_lik);
    dWinv.mean = zeros(n, num_mean);
  end

  if (compute_HnlZ)
    HnlZ.value              = zeros(num_hyperparameters);
    HnlZ.covariance_ind = covariance_ind;
    HnlZ.likelihood_ind = likelihood_ind;
    HnlZ.mean_ind       = mean_ind;
  end

  % indices of the diagonal entries of an (n x n) matrix
  diag_ind = (1:(n + 1):(n * n))';

  % computes diag(d) * A
  DA = @(d, A) bsxfun(@times, d(:), A);

  % computes A * diag(d)
  AD = @(A, d) bsxfun(@times, d(:)', A);

  % computes diag(d) * A * diag(d)
  DAD = @(d, A) DA(d, AD(A, d));

  % computes diag(AB)
  AB_diag = @(A, B) (sum(B .* A')');

  % convenience handles
  mu  = @(   varargin) feval(mean_function{:},       theta.mean, x, varargin{:});
  K   = @(   varargin) feval(covariance_function{:}, theta.cov,  x, [], varargin{:});
  ell = @(f, varargin) feval(likelihood{:},          theta.lik, ...
          y, f, [], 'infLaplace', varargin{:});

  % prior mean and covariance
  mu_x = mu();
  K_x  = K();

  % address potential numerical issues with K_x that arise during
  % Hessian computation
  if (compute_HnlZ)

    % add some jitter to avoid tiny eigenvalues
    jitter = 1e-8;
    K_x(diag_ind) = K_x(diag_ind) + jitter;

    % symmetrize
    K_x = (K_x + K_x') / 2;
  end

  % use last alpha as starting point if possible
  if (any(size(last_alpha) ~= [n, 1]) || any(isnan(last_alpha)))
    alpha = zeros(n, 1);
  else
    alpha = last_alpha;
    if (Psi_gpml(alpha, mu_x, K_x, ell) > -sum(ell(mu_x)))
      alpha = zeros(n, 1);
    end
  end

  % find \hat{f} via IRLS
  alpha = irls_gpml(alpha, mu_x, K_x, ell, options);
  last_alpha = alpha;

  % mode of training latent value posterior
  f = K_x * alpha + mu_x;

  if (compute_HnlZ)
    [lp, ~, d2lp, d3lp, d4lp] = ell(f);
  else
    [lp, ~, d2lp, d3lp]       = ell(f);
  end

  % diagonal of W matrix and its inverse
  w     = -d2lp;
  w_inv = 1 ./ w;

  posterior.alpha = alpha;
  posterior.sW    = sqrt(abs(w)) .* sign(w);

  % here we compute the following, using the correct parameterization of
  % the posterior structure:
  %
  %   log_det_B: log(det(B)), B = I + W^{1/2} K W^{1/2}
  %       V_inv: (K + W^{-1})^{-1}
  %           S: (I + KW)^{-1} = I - KV^{-1}
  %
  % additionally, we create a function handle V_inv_times(x), which
  % will compute V^{-1}x.

  if (any(w < 0))
    % posterior.L contains -V^{-1}

    [log_det_B, S, posterior.L] = logdetA_gpml(K_x, w);

    V_inv = -posterior.L;
    V_inv_times = @(x) (V_inv * x);
  else
    % posterior.L contains chol(I + W^{1/2} K W^{1/2}) = chol(B)

    B = DAD(posterior.sW, K_x);
    B(diag_ind) = B(diag_ind) + 1;
    posterior.L = chol(B);

    log_det_B = 2 * sum(log(diag(posterior.L)));

    V_inv_times = @(x) ...
        (DA(posterior.sW, ...
            solve_chol(posterior.L, DA(posterior.sW, x))));
    V_inv = V_inv_times(I);

    S = -V_inv_times(K_x)';
    S(diag_ind) = S(diag_ind) + 1;
  end

  % define
  %
  %   A^{-1} = SK = K - K V^{-1} K.
  %
  % we always need diag(A^{-1}), and in the case of Hessian
  % computation, need the full matrix A^{-1}. we calculate this below.
  if (compute_HnlZ)
    A_inv = S * K_x;
    a_inv = diag(A_inv);
  else
    a_inv = AB_diag(S, K_x);
  end

  nlZ = -sum(lp) + 0.5 * (alpha' * (f - mu_x) + log_det_B);

  dL_df = 0.5 * (a_inv .* d3lp)';

  implicit_gradient = @(df_dtheta) -(dL_df * df_dtheta);

  % when computing Hessian, store various intermediate computations
  % for reuse
  if (compute_HnlZ)

    % derivatives of mu, K, and likelihood with respect to hyperparameters
    dms = zeros(n, num_mean);
    dKs = zeros(n, n, num_cov);

      lp_dhyps = zeros(n, num_lik);
     dlp_dhyps = zeros(n, num_lik);
    d2lp_dhyps = zeros(n, num_lik);
    d3lp_dhyps = zeros(n, num_lik);

    % derivatives of \hat{f} with respect to \theta
    df_dthetas = zeros(n, num_hyperparameters);

    % K^{-1} [dK / dK_i] and V^{-1} [dK / dK_i]
    L = chol(K_x);
    K_inv_times = @(x) solve_chol(L, x);

    K_inv_dKs = zeros(n, n, num_cov);
    V_inv_dKs = zeros(n, n, num_cov);
  end

  for i = 1:num_cov
    dK = K(i);

    V_inv_dK = V_inv_times(dK);
    dK_alpha = dK * alpha;

    df_dtheta = S * dK_alpha;

    dnlZ.cov(i) = 0.5 * (trace(V_inv_dK) - alpha' * dK_alpha) + ...
        implicit_gradient(df_dtheta);

    if (compute_dalpha)
      dalpha.cov(:, i) = d2lp .* df_dtheta;
    end

    if (compute_dWinv)
      dWinv.cov(:, i) = d3lp .* df_dtheta;
    end

    % store intermediate calculations if necessary
    if (compute_HnlZ)
      dKs(:, :, i) = dK;

      V_inv_dKs(:, :, i) = V_inv_dK;
      K_inv_dKs(:, :, i) = K_inv_times(dK);

      df_dthetas(:, covariance_ind(i)) = df_dtheta;
    end
  end

  for i = 1:num_lik

    % compute and store d3lp_dhyp if necessary
    if (compute_HnlZ)
      [lp_dhyp, dlp_dhyp, d2lp_dhyp, d3lp_dhyps(:, i)] = ell(f, i);
    else
      [lp_dhyp, dlp_dhyp, d2lp_dhyp]                   = ell(f, i);
    end

    df_dtheta = S * (K_x * dlp_dhyp);

    dnlZ.lik(i) = -sum(lp_dhyp) - 0.5 * a_inv' * d2lp_dhyp + ...
        implicit_gradient(df_dtheta);

    if (compute_dalpha)
      dalpha.lik(:, i) = dlp_dhyp + d2lp .* df_dtheta;
    end

    if (compute_dWinv)
      dWinv.lik(:, i) = d2lp_dhyp + d3lp .* df_dtheta;
    end

    % store intermediate calculations if necessary
    if (compute_HnlZ)
        lp_dhyps(:, i) =   lp_dhyp;
       dlp_dhyps(:, i) =  dlp_dhyp;
      d2lp_dhyps(:, i) = d2lp_dhyp;

      df_dthetas(:, likelihood_ind(i)) = df_dtheta;
    end
  end

  for i = 1:num_mean
    dm = mu(i);

    df_dtheta = S * dm;

    dnlZ.mean(i) = -alpha' * dm + implicit_gradient(df_dtheta);

    if (compute_dalpha)
      dalpha.mean(:, i) = d2lp .* df_dtheta;
    end

    if (compute_dWinv)
      dWinv.mean(:, i) = d3lp .* df_dtheta;
    end

    % store intermediate calculations if necessary
    if (compute_HnlZ)
      dms(:, i) = dm;

      df_dthetas(:, mean_ind(i)) = df_dtheta;
    end
  end

  % dWinv currently contains dW / dtheta
  if (compute_dWinv)
    d = (w_inv .* w_inv);
    dWinv.cov  = bsxfun(@times, d, dWinv.cov);
    dWinv.lik  = bsxfun(@times, d, dWinv.lik);
    dWinv.mean = bsxfun(@times, d, dWinv.mean);
  end

  if (~compute_HnlZ)
    return;
  end

  % converts to column vector
  vectorize = @(x) (x(:));

  % computes tr(AB)
  AB_trace = @(A, B) (vectorize(A')' * B(:));

  % computes tr(A diag(a) B diag(b))
  ADBD_trace = @(A, a, B, b) (b' * (A .* B') * a);

  d2f_dtheta2s   = zeros(n, num_hyperparameters, num_hyperparameters);
  d2L_dtheta_dfs = zeros(num_hyperparameters, n);

  A_inv_d3lp = AD(A_inv, d3lp);

  for i = 1:num_cov
    ind_i = covariance_ind(i);

    % compute once
    if (i == 1)
      V_inv_w_inv = AD(V_inv, w_inv);
    end

    % diag[W^{-1} V^{-1} [dK / dK_i] V^{-1} W^{-1}]
    beta = AB_diag(DA(w_inv, V_inv_dKs(:, :, i)), V_inv_w_inv);

    d2L_dtheta_dfs(ind_i, :) = ...
        (K_inv_dKs(:, :, i) * alpha + 0.5 * beta .* d3lp)';

    for j = i:num_cov
      ind_j = covariance_ind(j);

      HK = K(i, j);

      d2f_dtheta2s(:, ind_i, ind_j) = ...
          S * ( ...
              HK * alpha + ...
              dKs(:, :, i) * (d2lp .* df_dthetas(:, ind_j)) + ...
              dKs(:, :, j) * (d2lp .* df_dthetas(:, ind_i)) + ...
              K_x * (d3lp .* df_dthetas(:, ind_i) .* df_dthetas(:, ind_j)) ...
              );

      HnlZ.value(ind_i, ind_j) = ...
          alpha' * dKs(:, :, i) * K_inv_dKs(:, :, j) * alpha + ...
          0.5 * (AB_trace(V_inv, HK) - ...
                 AB_trace(V_inv_dKs(:, :, i), V_inv_dKs(:, :, j)) - ...
                 alpha' * HK * alpha);
    end

    for j = 1:num_lik
      ind_j = likelihood_ind(j);

      d2f_dtheta2s(:, ind_i, ind_j) = ...
          S * ( ...
              dKs(:, :, i) * (dlp_dhyps(:, j) + d2lp .* df_dthetas(:, ind_j)) + ...
              K_x * (d2lp_dhyps(:, j) .* df_dthetas(:, ind_i) + ...
                     d3lp .* df_dthetas(:, ind_i) .* df_dthetas(:, ind_j)) ...
              );

      HnlZ.value(ind_i, ind_j) = -0.5 * beta' * d2lp_dhyp(:, j);
    end

    for j = 1:num_mean
      ind_j = mean_ind(j);

      d2f_dtheta2s(:, ind_i, ind_j) = ...
          S * ( ...
              dKs(:, :, i) * (d2lp .* df_dthetas(:, ind_j)) + ...
              K_x * (d3lp .* df_dthetas(:, ind_i) .* df_dthetas(:, ind_j)) ...
              );

      HnlZ.value(ind_i, ind_j) = alpha' * K_inv_dKs(:, :, i)' * dms(:, j);
    end
  end

  for i = 1:num_lik
    ind_i = likelihood_ind(i);

    d2L_dtheta_dfs(ind_i, :) = ...
        (dlp_dhyps(:, i) + ...
         0.5 * (a_inv .* d3lp_dhyps(:, i) + ...
                AB_diag(AD(A_inv, d2lp_dhyps(:, i)), A_inv_d3lp)))';

    for j = i:num_lik
      ind_j = likelihood_ind(j);

      [lp_dhyp2, dlp_dhyp2, d2lp_dhyp2] = ell(f, i, j);

      d2f_dtheta2s(:, ind_i, ind_j) = ...
          S * (K_x * (dlp_dhyp2 + ...
                      d2lp_dhyps(:, i) .* df_dthetas(:, ind_j) + ...
                      d2lp_dhyps(:, j) .* df_dthetas(:, ind_i) + ...
                      d3lp .* df_dthetas(:, ind_i) .* df_dthetas(:, ind_j)));

      HnlZ.value(ind_i, ind_j) = -sum(lp_dhyp2) - ...
          0.5 * (a_inv' * d2lp_dhyp2 + ...
                 ADBD_trace(A_inv, d2lp_dhyps(:, i), A_inv, d2lp_dhyps(:, j)));
    end

    for j = 1:num_mean
      ind_j = mean_ind(j);

      d2f_dtheta2s(:, ind_i, ind_j) = ...
          S * (K_x * (d2lp_dhyps(:, i) .* dms(:, j) + ...
                      d3lp .* df_dthetas(:, ind_i) .* df_dthetas(:, ind_j)));

      % explicit derivative is zero
    end
  end

  for i = 1:num_mean
    ind_i = mean_ind(i);

    d2L_dtheta_dfs(ind_i, :) = K_inv_times(dms(:, i))';

    for j = i:num_mean
      ind_j = mean_ind(j);

      d2m = mu(i, j);

      d2f_dtheta2s(:, ind_i, ind_j) = ...
          S * (d2m + ...
               K_x * (d3lp .* df_dthetas(:, ind_i) .* df_dthetas(:, ind_j)) ...
               );

      HnlZ.value(ind_i, ind_j) = ...
          -alpha' * d2m + dms(:, i)' * K_inv_times(dms(:, j));
    end
  end

  % correct Hessian due to dependence of \hat{f} on \theta
  d2log_det_B_df2 = 0.5 * A_inv_d3lp .* A_inv_d3lp';
  d2log_det_B_df2(diag_ind) = d2log_det_B_df2(diag_ind) + ...
      0.5 * (a_inv .* d4lp);

  for i = 1:num_hyperparameters
    d2L_df2_df_dtheta_i = ...
        -(A_inv \ df_dthetas(:, i)) + d2log_det_B_df2 * df_dthetas(:, i);

    for j = i:num_hyperparameters
      HnlZ.value(i, j) = HnlZ.value(i, j)         - ...
          dL_df * d2f_dtheta2s(:, i, j)           - ...
          d2L_df2_df_dtheta_i' * df_dthetas(:, j) - ...
          d2L_dtheta_dfs(i, :) * df_dthetas(:, j) - ...
          d2L_dtheta_dfs(j, :) * df_dthetas(:, i);
    end
  end

  % symmetrize Hessian
  HnlZ.value = HnlZ.value + triu(HnlZ.value, 1)';

end
