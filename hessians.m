% Notes on Hessian struct HnlZ
% ----------------------------
%
% Several files in this project support computing the Hessian with
% respect to the values of the hyperparameters of various functions:
%
% - the negative log training likelihood (see exact_inference),
% - a negative log prior placed on the hyperparameters (see
%   independent_prior), and
% - the (unnormalized) negative log training posterior (see gp_posterior).
%
% These Hessians are stored in a struct called HnlZ.
%
% In all cases, HnlZ contains the _full_ Hessian matrix with respect
% to the hyperparameters, including cross terms (for example,
% involving covariance/mean parameter pairs). The Hessian matrix is
% stored in the field
%
%   .H: Hessian matrix
%
% The dimensions of HnlZ.H are (m x m), where m is the number of
% hyperparameters:
%
%   m = numel(unwrap(hyperparameters)).
%
% The order of the Hessian entries is the same as that produced by
% unwrap(dnlZ): covariance parameters, likelihood parameters, then
% mean parameters. This ordering allows the Hessian to be used in
% second-order optimization routines easily, by pairing it with the
% full gradient vector given by unwrap(dnlZ).
%
% Sometimes only the portion of the Hessian pertaining to a subset of
% the hyperparameters is desired. For this reason, the HnlZ struct
% also contains the following fields for convenient indexing:
%
%   .covariance_ind: index into HnlZ.H corresponding to covariance terms
%   .likelihood_ind: index into HnlZ.H corresponding to likelihood terms
%         .mean_ind: index into HnlZ.H corresponding to mean terms
%
% See also EXACT_INFERENCE, INDEPENDENT_PRIOR, GP_POSTERIOR.

% Copyright (c) 2014 Roman Garnett.