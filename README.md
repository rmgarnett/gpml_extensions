GPML Extensions
===============

This repository contains a collection of extensions to the popular
GPML toolbox for Gaussian process inference in `MATLAB`, available
here:

http://www.gaussianprocess.org/gpml/code/matlab/doc/

We provide code for:

* Incorporating arbitrary hyperparameter priors ![p(\theta)][1] into
  any inference method, allowing for MAP rather than MLE inference
  during hyperparameter learning.
* An extended API for mean and covariance functions for computing
  Hessians with respect to their hyperparameters.
* An extended API for inference methods to compute the (potentially
  approximate) Hessian of the negative log likelihood/posterior with
  respect to hyperparameters.
* Implementations of several new mean and covariance functions.
* A number of additional utilities, for example, for computing
  rank-one updates to quickly update a posterior when performing
  online GP regression.

Hyperparameter Priors
---------------------

We establish a new, simple API for specifying arbitrary hyperparameter
priors ![p(\theta)][1]. The API is:

    [nlZ, dnlZ, HnlZ] = prior(hyperparameters)

Where the input is:

* `hyperparameters`: a GPML hyperparameter struct specifying ![\theta][2]

and the outputs are

* `nlZ`: the negative of the log prior evaluated at ![\theta][2],
  ![-\log p(\theta)][3]
* `dnlZ`: a struct containing the gradient of the negative log prior
  evaluated at ![\theta][2],
  ![\nabla -\log p(\theta)][4]
* `HnlZ`: (optional) a struct containing the Hessian of the negative
  log prior evaluated at ![\theta][2],
  ![H( -\log p(\theta) )][5]

The `dnlZ` struct is specified in the same way as is typical for GPML
(for example, as the second output of `gp.m` in training mode), and,
if needed, the Hessian is specified as described in the section on
Hessians below.

We provide an implementation of a flexible family of such priors in
`independent_prior.m`. This implements a meta-prior of the form

![p(\theta) = \prod_i p(\theta_i)][6]

where we have placed independent priors on each hyperparameter
![\theta_i][7]. Several elementwise priors are provided, including:

* normal priors (see `gaussian_prior`): ![p(\theta_i) = N(\mu, \sigma^2)][8]
* uniform priors (see `uniform_prior`): ![p(\theta_i) = U}(\ell, u)][9]
* Laplace priors (see `laplace_prior`): ![p(\theta_i) = Lapalce(\mu, b)][10]
* improper constant priors (see `constant_prior`): ![p(\theta_i) = 1][11]

Once a hyperparameter prior is specified, a special meta-inference
method, `inference_with_prior` allows the user to incorporate the
prior into any arbitrary GPML inference method. Except for extra
inputs specifying the inference method and prior, the API of
`inference_with_prior` is identical to the standard GPML inference
method API, except that the negative log likelihood `nlZ`, its
gradient `dnlZ`, and, optionally, its Hessian `HnlZ`, are replaced
with the equivalent expressions for the negative (unnormalized) log
posterior ![-\log p(y | x, D, \theta) - \log p(\theta)][12].

Here is a demonstration of incorporating a hyperparameter prior to a
GPML model:

    inference_method    = @infExact;
    mean_function       = {@meanConst};
    covariance_function = {@covSEiso};

    % initial hyperparameters
    offset       = 1;
    length_scale = 1;
    output_scale = 1;
    noise_std    = 0.05;

    hyperparameters.mean = offset;
    hyperparameters.cov  = log([length_scale; output_scale]);
    hyperparameters.lik  = log(noise_std);

    % add normal priors to each hyperparameter
    priors.mean = {get_prior(@gaussian_prior, 0, 1)};
    priors.cov  = {get_prior(@gaussian_prior, 0, 1), ...
                   get_prior(@gaussian_prior, 0, 1)};
    priors.lik  = {get_prior(@gaussian_prior, log(0.01), 1)};

    % add prior to inference method
    prior = get_prior(@independent_prior, priors);
    inference_method = add_prior_to_inference_method(inference_method, prior);

    % find MAP hyperparameters
    map_hyperparameters = minimize(hyperparameters, @gp, 50, inference_method, ...
            mean_function, covariance_function, [], x, y);

Hessians
--------

We establish a simple extension to the GPML API for mean and
covariance functions, allowing us to compute their Hessians with
respect to their hyperparameters.

To compute the second (mixed) partial derivative of ![\mu(x)][13] with
respect to the pair ![(\theta_i, \theta_j)][14],

![\partial^2 / \partial \theta_i \partial \theta_j \mu(x)][15]

the interface is:

    mu = mean_function(hyperparameters, x, i, j)

which differs from the interface for computing gradients only by the
additional input argument `j`. Several mean function implementations
compliant with this extended interface are provided:

* `zero_mean`: a drop-in replacement for `meanZero`
* `constant_mean`: a drop-in replacement for `meanConst`
* `linear_mean`: a drop-in replacement for `meanLinear`

Similarly, to compute the second (mixed) partial derivative of
![K(x, z)][16] with respect to the pair ![(\theta_i, \theta_j)][14],

![\partial^2 / \partial \theta_i \partial \theta_j K(x, z][17]

the interface is:

    K = covariance_function(hyperparameters, x, z, i, j)

Several covariance function implementations compliant with this
extended interface are provided:

* `isotropic_sqdexp_covariance`: a drop-in replacement for `covSEiso`
* `ard_sqdexp_covariance`: a drop-in replacement for `covSEard`
* `factor_sqdexp_covariance`: an implementation of a squared
  exponential "factor" covariance, where an isotropic squared
  exponential is applied to data after a linear map to a
  lower-dimensional space.

These Hessians can ultimately be used to compute the Hessian of the
log likelihood with respect to the hyperparameters. In particular, we
provide:

* `exact_inference`: a drop-in replacement for `infExact`
* `laplace_inference`: a drop-in replacement for `infLaplace`.

Both support the extended inference method API

    [posterior, nlZ, dnlZ, HnlZ] = ...
        inference_method(hyperparameters, mean_function, ...
                         covariance_function, likelihood, x, y);

The last output, `HnlZ`, is a struct describing the Hessian of the
negative log likelihood with respect to ![\theta][2], including with
respect to "off-block-diagonal" terms such as mean/covariance,
mean/likelihood, and covariance/likelihood hyperparameter pairs. See
`hessians.m` for a description of this struct.

Other
-----

A number of additional files are included, providing additional
functionality. These include:

* New mean functions:
    * `step_mean`: a simple "step" changepoint mean
	* `discrete_mean`/`fixed_discrete_mean`: free-form mean vectors
      for discrete data; `discrete_mean` treats the entries of this
      vector as hyperparameters, enabling learning.
* New covariance functions:
    * `discrete_covariance`/`fixed_discrete_covariance`: free-form
      covariance matrices for discrete data; `discrete_covariance`
      treats the entries of this matrix as hyperparameters, enabling
      learning. A log-Cholesky parameterization is used, allowing for
      unconstrained optimization.
	* `scaled_covariance`: a meta-covariance for modeling functions of
      the form ![a(x)f(x)][18], where ![p(f) = GP(\mu, K)][19] and
      ![a(x)][20] is a fixed function. ![a(x)][20] is specified as a
      GPML mean function, and ![K(x, x')][21] is specified as a GPML
      covariance function. `scaled_covariance` computes the gradient
      of the covariance with respect to both the parameters of
      ![a(x)][20] and ![K(x, x'][21].
* Rank-one updates of GPML posterior structs: `update_posterior`
  allows the user to update an existing GPML posterior struct (for
  regression with Gaussian observation noise) given a single new
  observation ![(x*, y*)][22]. This can significantly decrease the
  total time needed to perform sequential online GP regression.
* Computing likelihoods assuming datasets are from independent draws
  from a joint GP prior: see `gp_likelihood_independent` for more
  information.

[1]: http://latex.codecogs.com/svg.latex?p(%5Ctheta)
[2]: http://latex.codecogs.com/svg.latex?%5Ctheta
[3]: http://latex.codecogs.com/svg.latex?-%5Clog%20p(%5Ctheta)
[4]: http://latex.codecogs.com/svg.latex?%5Cnabla%20-%5Clog%20p(%5Ctheta)
[5]: http://latex.codecogs.com/svg.latex?H%5Cbigl(-%5Clog%20p(%5Ctheta)%20%5Cbigr)
[6]: http://latex.codecogs.com/svg.latex?p(%5Ctheta)%20%3D%20%5Cprod_i%20p(%5Ctheta_i)
[7]: http://latex.codecogs.com/svg.latex?%5Ctheta_i
[8]: http://latex.codecogs.com/svg.latex?p(%5Ctheta_i%20%5Cmid%20%5Cmu%2C%20%5Csigma%5E2)%20%3D%20%5Cmathcal%7BN%7D(%5Ctheta_i%3B%20%5Cmu%2C%20%5Csigma%5E2)%20
[9]: http://latex.codecogs.com/svg.latex?p(%5Ctheta_i%20%5Cmid%20%5Cell%2C%20u)%20%3D%20%5Cmathcal%7BU%7D(%5Cell%2C%20u)
[10]: http://latex.codecogs.com/svg.latex?p(%5Ctheta_i%20%5Cmid%20%5Cmu%2C%20b)%20%3D%20%5Ctext%7BLaplace%7D(%5Cmu%2C%20b)
[11]: http://latex.codecogs.com/svg.latex?p(%5Ctheta_i)%20%3D%201
[12]: http://latex.codecogs.com/svg.latex?-%5Clog%20p(y%20%5Cmid%20X%2C%20%5Cmathcal%7BD%7D%2C%20%5Ctheta)%20-%20%5Clog%20p(%5Ctheta)
[13]: http://latex.codecogs.com/svg.latex?%5Cmu(x)
[14]: http://latex.codecogs.com/svg.latex?(%5Ctheta_i%2C%20%5Ctheta_j)
[15]: http://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%20%5Ctheta_i%20%5Cpartial%20%5Ctheta_j%7D%20%5Cmu(x)
[16]: http://latex.codecogs.com/svg.latex?K(x%2C%20z)
[17]: http://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%20%5Ctheta_i%20%5Cpartial%20%5Ctheta_j%7D%20K(x%2C%20z)
[18]: http://latex.codecogs.com/svg.latex?a(x)f(x)
[19]: http://latex.codecogs.com/svg.latex?p(f)%20%3D%20%5Cmathcal%7BGP%7D(f%3B%20%5Cmu%2C%20K)
[20]: http://latex.codecogs.com/svg.latex?a(x)
[21]: http://latex.codecogs.com/svg.latex?K(x%2C%20x%27)
[22]: http://latex.codecogs.com/svg.latex?(x%5E%5Cast%2C%20y%5E%5Cast)
