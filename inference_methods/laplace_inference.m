% LAPLACE_INFERENCE infLaplace replacement supporting Hessian calculation.
%
% This provides a GPML-compatible inference method performing
% approximate inference via a Laplace approximation. This
% implementation supports an extended API allowing the calculation of:
%
% - the Hessian of the negative log likelihood at \theta,
% - the partial derivatives of \alpha with respect to \theata, and
% - the partial derivatives of diag W^{-1} with respect to \theta.
%
% The latter two can be used to compute the gradient of the predictive
% mean and variance of the approximate posterior GP with respect to
% the hyperparameters.
%
% This can be used as a drop-in replacement for infLaplace with no
% extra computational cost.
%%
% Usage
% -----
%
% The API is identical to GPML inference methods, expect for three
% additional optional arguments:
%
%   [posterior, nlZ, dnlZ, HnlZ, dalpha, dWinv] = ...
%       laplace_inference(hyperparameters, mean_function, ...
%                         covariance_function, likelihood, x, y);
%
% HnlZ proivdes the Hessian of -\log p(y | X, \theta). See hessians.m
% for information regarding the Hessian struct HnlZ.
%
% dalpha and dWinv provide the partial derivatives of the posterior
% parameters \alpha and W^{-1} with respect to \theta. These
% arragement of these structs is similar to the dnlZ struct. For
% example, dalpha.cov(:, 1) gives the derivative of \alpha with
% respect to the first covariance hyperparameter.
%
% Requirements
% ------------
%
% To use this functionality, both the mean and covariance functions
% must support an extended GPML syntax that allows for calculating the
% Hessian of the training mean mu or training covariance K with
% respect to any pair of hyperparameters. The syntax for mean
% functions is:
%
%   dmu_didj = mean_function(hyperparameters, x, i, j);
%
% where dmu_didj is \partial^2 mu(x) / \partial \theta_i \partial \theta_j.
%
% The syntax for covariance functions is similar:
%
%   dK2_didj = covariance_function(hyperparameters, x, [], i, j);
%
% where dK2_didj is \partial^2 K(x, x) / \partial \theta_i \partial \theta_j.
%
% Furhtermore, the likelihood must also support an extended GPML
% syntax for calculating further derivatives; a summary is below.
%
% The likelihood syntax with five outputs:
%
%   [lp, dlp, d2lp, d3lp, d4lp] = ...
%           likelihood(hyperparameters, y, f, [], 'infLaplace');
%
% returns the additional value
%
%   d4lp = d^4 log p(y | f) / df^4
%
% The gradient syntax with four outputs:
%
%   [lp_dhyp, dlp_dhyp, d2lp_dhyp, d3lp_dhyp] = ...
%           likelihood(hyperparameters, y, f, [], 'infLaplace', i);
%
% returns the additional value
%
%   d3lp_dhyp = d^4 log p(y | f) / (df^3 d\theta_i)
%
% Finally, a new Hessian syntax:
%
%   [lp_dhyp2, dlp_dhyp2, d2lp_dhyp2] = ...
%           likelihood(hyperparameters, y, f, [], 'infLaplace', i, j);
%
% returns
%
%     lp_dhyp2 = d^2 log p(y | f) / (     d\theta_i d\theta_j)
%    dlp_dhyp2 = d^3 log p(y | f) / (df   d\theta_i d\theta_j)
%   d2lp_dhyp2 = d^4 log p(y | f) / (df^2 d\theta_i d\theta_j)
%
% Finally, this inference method also supports two further optional
% outputs.
%
% See also INFMETHODS, HESSIANS.

% Copyright (c) 2013--2014 Roman Garnett.

function [posterior, nlZ, dnlZ, HnlZ, dalpha, dWinv] = ...
      laplace_inference(theta, mean_function, covariance_function, ...
          likelihood, x, y, options)

  persistent last_alpha;

  % ensure options argument exists
  if (nargin <= 6)
    options = [];
  end

  % If Hessian is not requested, simply call infLaplace and return. This
  % allows us to assume the Hessian is needed for the remainder of the
  % code, making it more readible.

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

    last_alpha = posterior.alpha;
    return;
  end

  n = size(x, 1);
  I = eye(n);

  % initialize output
  dnlZ = theta;

  num_cov  = numel(theta.cov);
  num_lik  = numel(theta.lik);
  num_mean = numel(theta.mean);
  num_hyperparameters = num_cov + num_lik + num_mean;

  covariance_ind = 1:num_cov;
  likelihood_ind = (num_cov + 1):(num_cov + num_lik);
  mean_ind       = (num_cov + num_lik + 1):num_hyperparameters;

  HnlZ.H              = zeros(num_hyperparameters);
  HnlZ.covariance_ind = covariance_ind;
  HnlZ.likelihood_ind = likelihood_ind;
  HnlZ.mean_ind       = mean_ind;

  if (nargout >= 4)
    dalpha.cov  = zeros(n, num_cov);
    dalpha.lik  = zeros(n, num_lik);
    dalpha.mean = zeros(n, num_mean);
  end

  if (nargout >= 5)
    dWinv.cov  = zeros(n, num_cov);
    dWinv.lik  = zeros(n, num_lik);
    dWinv.mean = zeros(n, num_mean);
  end

  % convenience handles
  mu  = @(   varargin) feval(mean_function{:},       theta.mean, x, varargin{:});
  K   = @(   varargin) feval(covariance_function{:}, theta.cov,  x, [], varargin{:});
  ell = @(f, varargin) feval(likelihood{:},          theta.lik, ...
          y, f, [], 'infLaplace', varargin{:});

  % prior mean and covariance
  mu_x = mu();
  K_x  = K();

  % computes diag(d) * A
  DA = @(d, A) bsxfun(@times, d(:), A);

  % computes A * diag(d)
  AD = @(A, d) bsxfun(@times, d(:)', A);

  % computes diag(d) * A * diag(d)
  DAD = @(d, A) DA(d, AD(A, d));

  % indices of the diagonal entries of an (n x n) matrix
  diag_ind = (1:(n + 1):(n * n))';

  % converts to column vector
  vectorize = @(x) (x(:));

  % computes tr(AB)
  AB_trace = @(A, B) (vectorize(A')' * B(:));

  % computes tr(A diag(a) B diag(b))
  ADBD_trace = @(A, a, B, b) (b' * (A .* B') * a);

  % computes diag(AB)
  AB_diag = @(A, B) (sum(B .* A')');

  % computes diag(A diag(a) B diag(b))
  ADBD_diag = @(A, a, B, b) ((A .* B') * a .* b);

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

  [lp, ~, d2lp, d3lp, d4lp] = ell(f);

  w     = -d2lp;
  w_inv = 1 ./ w;

  posterior.alpha = alpha;
  posterior.sW    = sqrt(abs(w)) .* sign(w);

  L = chol(K_x);
  K_inv_times = @(x) solve_chol(L, x);
  K_inv = K_inv_times(I);

  A = K_inv;
  A(diag_ind) = A(diag_ind) + w;

  if (any(w < 0))
    % posterior.L contains -V^{-1}

    [log_det_B, S, posterior.L] = logdetA_gpml(K_x, w);

    V_inv = -posterior.L;
    V_inv_times = @(x) (V_inv * x);
  else
    % posterior.L contains chol(I + W^{1/2} K W^{1/2})

    B = DAD(posterior.sW, K_x);
    B(diag_ind) = B(diag_ind) + 1;
    posterior.L = chol(B);

    log_det_B = 2 * sum(log(diag(posterior.L)));

    V_inv_times = @(x) ...
        (DA(posterior.sW, ...
            solve_chol(posterior.L, DA(posterior.sW, x))));
    V_inv = V_inv_times(I);

    S = -K_x * V_inv;
    S(diag_ind) = S(diag_ind) + 1;
  end

  A_inv = S * K_x;
  a_inv = diag(A_inv);

  nlZ = -sum(lp) + 0.5 * (alpha' * (f - mu_x) + log_det_B);

  dL_df = (0.5 * a_inv .* d3lp)';

  d2L_df2 = -A;
  d2L_df2(diag_ind) = d2L_df2(diag_ind) + ...
      0.5 * (a_inv .* d4lp + ...
             ADBD_diag(A_inv, d3lp, A_inv, d3lp));

  df_dtheta     = zeros(n, num_hyperparameters);
  d2f_dtheta2   = zeros(n, num_hyperparameters, num_hyperparameters);
  d2L_dtheta_df = zeros(num_hyperparameters, n);

  implicit_gradient = @(df_dtheta) -(dL_df * df_dtheta);
  implicit_hessian  = ...
      @(df_dtheta_i, df_dtheta_j, ...
        d2f_dtheta_i_dtheta_j, ...
        d2L_dtheta_i_df, d2L_dtheta_j_df) ...
      -(          dL_df * d2f_dtheta_i_dtheta_j + ...
           df_dtheta_i' * d2L_df2 * df_dtheta_j + ...
        d2L_dtheta_i_df * df_dtheta_j           + ...
        d2L_dtheta_j_df * df_dtheta_i);

  % store derivatives of mu, K, and likelihood with respect to hyperparameters for reuse
  dm = zeros(n, num_mean);
  dK = zeros(n, n, num_cov);

    lp_dhyp = zeros(n, num_lik);
   dlp_dhyp = zeros(n, num_lik);
  d2lp_dhyp = zeros(n, num_lik);

  % store K^{-1} [dK / dK_i] and V^{-1} [dK / dK_i] for reuse
  K_inv_dK = zeros(n, n, num_cov);
  V_inv_dK = zeros(n, n, num_cov);

  % store diag[W^{-1} V^{-1} [dK / dK_i] V^{-1} W^{-1}] for reuse
  beta = zeros(n, num_cov);

  for i = 1:num_cov
    ind = covariance_ind(i);

    dK(:, :, i) = K(i);

    dK_alpha = dK(:, :, i) * alpha;

    K_inv_dK(:, :, i) = K_inv_times(dK(:, :, i));
    V_inv_dK(:, :, i) = V_inv_times(dK(:, :, i));

    df_dtheta(:, ind) = S * dK_alpha;

    % compute once
    if (i == 1)
      V_inv_w_inv = AD(V_inv, w_inv);
    end

    beta(:, i) = AB_diag(DA(w_inv, V_inv_dK(:, :, i)), V_inv_w_inv);

    d2L_dtheta_df(ind, :) = ...
        (K_inv_dK(:, :, i) * alpha + 0.5 * beta(:, i) .* d3lp)';

    dnlZ.cov(i) = 0.5 * (trace(V_inv_dK(:, :, i)) - alpha' * dK_alpha) + ...
        implicit_gradient(df_dtheta(:, ind));

    if (nargout >= 5)
      dalpha.cov(:, i) = d2lp .* df_dtheta(:, ind);
    end

    if (nargout >= 6)
      dWinv.cov(:, i) = d3lp .* df_dtheta(:, ind);
    end
  end

  for i = 1:num_lik
    ind = likelihood_ind(i);

    [lp_dhyp(:, i), dlp_dhyp(:, i), d2lp_dhyp(:, i), d3lp_dhyp] = ell(f, i);

    df_dtheta(:, ind) = S * (K_x * dlp_dhyp(:, i));

    d2L_dtheta_df(ind, :) = ...
        (dlp_dhyp(:, i) + ...
         0.5 * (a_inv .* d3lp_dhyp + ...
                ADBD_diag(A_inv, d2lp_dhyp(:, i), A_inv, d3lp)))';

    dnlZ.lik(i) = -sum(lp_dhyp(:, i)) - 0.5 * a_inv' * d2lp_dhyp(:, i) + ...
        implicit_gradient(df_dtheta(:, ind));

    if (nargout >= 5)
      dalpha.lik(:, i) = dlp_dhyp(:, i) + d2lp .* df_dtheta(:, ind);
    end

    if (nargout >= 6)
      dWinv.lik(:, i) = d2lp_dhyp(:, i) + d3lp .* df_dtheta(:, ind);
    end
  end

  for i = 1:num_mean
    ind = mean_ind(i);

    dm(:, i) = mu(i);

    df_dtheta(:, ind) = S * dm(:, i);

    d2L_dtheta_df(ind, :) = K_inv_times(dm(:, i))';

    dnlZ.mean(i) = -alpha' * dm(:, i) + ...
        implicit_gradient(df_dtheta(:, ind));

    if (nargout >= 5)
      dalpha.mean(:, i) = d2lp .* df_dtheta(:, ind);
    end

    if (nargout >= 6)
      dWinv.mean(:, i) = d3lp .* df_dtheta(:, ind);
    end
  end

  for i = 1:num_cov
    ind_i = covariance_ind(i);

    for j = i:num_cov
      ind_j = covariance_ind(j);

      HK = K(i, j);

      d2f_dtheta2(:, ind_i, ind_j) = ...
          S * ( ...
              HK * alpha + ...
              dK(:, :, i) * (d2lp .* df_dtheta(:, ind_j)) + ...
              dK(:, :, j) * (d2lp .* df_dtheta(:, ind_i)) + ...
              K_x * (d3lp .* df_dtheta(:, ind_i) .* df_dtheta(:, ind_j)) ...
              );

      HnlZ.H(ind_i, ind_j) = ...
          alpha' * dK(:, :, i) * K_inv_dK(:, :, j) * alpha + ...
          0.5 * (AB_trace(V_inv, HK) - ...
                 AB_trace(V_inv_dK(:, :, i), V_inv_dK(:, :, j)) - ...
                 alpha' * HK * alpha);
    end

    for j = 1:num_lik
      ind_j = likelihood_ind(j);

      d2f_dtheta2(:, ind_i, ind_j) = ...
          S * ( ...
              dK(:, :, i) * (dlp_dhyp(:, j) + d2lp .* df_dtheta(:, ind_j)) + ...
              K_x * (d2lp_dhyp(:, j) .* df_dtheta(:, ind_i) + ...
                     d3lp .* df_dtheta(:, ind_i) .* df_dtheta(:, ind_j)) ...
              );

      HnlZ.H(ind_i, ind_j) = -0.5 * beta(:, i)' * d2lp_dhyp(:, j);
    end

    for j = 1:num_mean
      ind_j = mean_ind(j);

      d2f_dtheta2(:, ind_i, ind_j) = ...
          S * ( ...
              dK(:, :, i) * (d2lp .* df_dtheta(:, ind_j)) + ...
              K_x * (d3lp .* df_dtheta(:, ind_i) .* df_dtheta(:, ind_j)) ...
              );

      HnlZ.H(ind_i, ind_j) = alpha' * K_inv_dK(:, :, i)' * dm(:, j);
    end
  end

  for i = 1:num_lik
    ind_i = likelihood_ind(i);

    for j = i:num_lik
      ind_j = likelihood_ind(j);

      [lp_dhyp2, dlp_dhyp2, d2lp_dhyp2] = ell(f, i, j);

      d2f_dtheta2(:, ind_i, ind_j) = ...
          S * (K_x * (dlp_dhyp2 + ...
                      d2lp_dhyp(:, i) .* df_dtheta(:, ind_j) + ...
                      d2lp_dhyp(:, j) .* df_dtheta(:, ind_i) + ...
                      d3lp .* df_dtheta(:, ind_i) .* df_dtheta(:, ind_j)));

      HnlZ.H(ind_i, ind_j) = -sum(lp_dhyp2) - ...
          0.5 * (a_inv' * d2lp_dhyp2 + ...
                 ADBD_trace(A_inv, d2lp_dhyp(:, i), A_inv, d2lp_dhyp(:, j)));
    end

    for j = 1:num_mean
      ind_j = mean_ind(j);

      d2f_dtheta2(:, ind_i, ind_j) = ...
          S * (K_x * (d2lp_dhyp(:, i) .* dm(:, j) + ...
                      d3lp .* df_dtheta(:, ind_i) .* df_dtheta(:, ind_j)));

      % explicit derivative is zero
    end
  end

  for i = 1:num_mean
    ind_i = mean_ind(i);

    for j = i:num_mean
      ind_j = mean_ind(j);

      d2m = mu(i, j);

      d2f_dtheta2(:, ind_i, ind_j) = ...
          S * (d2m + ...
               K_x * (d3lp .* df_dtheta(:, ind_i) .* df_dtheta(:, ind_j)) ...
               );

      HnlZ.H(ind_i, ind_j) = ...
          -alpha' * d2m + dm(:, i)' * K_inv_times(dm(:, j));
    end
  end

  % correct Hessian due to dependence of \hat{f} on \theta
  for i = 1:num_hyperparameters
    for j = i:num_hyperparameters
      HnlZ.H(i, j) = HnlZ.H(i, j) + ...
          implicit_hessian(...
              df_dtheta(:, i),      ...
              df_dtheta(:, j),      ...
              d2f_dtheta2(:, i, j), ...
              d2L_dtheta_df(i, :),  ...
              d2L_dtheta_df(j, :)   ...
              );
    end
  end

  % symmetrize Hessian
  HnlZ.H = HnlZ.H + triu(HnlZ.H, 1)';

  % dWinv current contains dW / dtheta
  if (nargout >= 6)
    d = (w_inv .* w_inv);
    dWinv.cov  = bsxfun(@times, d, dWinv.cov);
    dWinv.lik  = bsxfun(@times, d, dWinv.lik);
    dWinv.mean = bsxfun(@times, d, dWinv.mean);
  end

end
