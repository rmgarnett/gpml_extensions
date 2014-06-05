GPML Extensions
===============

This repository contains a collection of extensions to the popular
GPML toolbox for Gaussian process inference in `MATLAB`, providing
code for:

* Incorporating aribitrary hyperparameter priors ![p(\theta)][1] into
  any inference method, allowing for MAP rather than MLE inference
  during hyperparameter learning.
* An extended API for mean and covariance functions for computing
  Hessians with respect to their hyperparameters.
* An extneded API for inference methods to compute the (potentially
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

    [nlZ, dnlZ, HnlZ] = prior(theta)

Where the input is:

* `theta`: a GPML hyperparameter struct specifying ![\theta][2]

and the outputs are

* `nlZ`: the negative of the log prior evaluated at ![\theta][2],
  ![-\log p(\theta)][3]
* `dnlZ`: a struct containing the gradient of the negative log prior
  evaluated at ![\theta][2],
  ![\nabla -\log p(\theta)][4]
* `HnlZ`: (optional) a struct containing the Hessian of the negative
  log prior evalauted at ![\theta][2],
  ![H( -\log p(\theta) )][5]

The `dnlZ` struct is specified in the same way as is typical for GPML
(for example, as the second output of `gp.m` in training mode), and,
if needed, the Hessian is specified as described in the section on
Hessians below.

We provide an implementation of a flexible family of such priors in
`indpendent_prior.m`. This implements a meta-prior of the form

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

Here is a demonstaration of incorporating a hyperparameter prior to a
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

    % add standard normal priors to each hyperparameter
    priors.mean = {get_prior(@gaussian_prior, 0, 1)};
    priors.cov  = ...
        {get_prior(@gaussian_prior, 0, 1), ...
         get_prior(@gaussian_prior, 0, 1};

    priors.lik  = {get_prior(@gaussian_prior, 0, 1};

    prior = get_prior(@independent_prior, priors);
    inference_method = add_prior_to_inference_method(inference_method, prior);

    % find MAP hyperparameters
    map_hyperparameters = minimize(hyperparameters, @gp, 50, inference_method, ...
            mean_function, covariance_function, [], x, y);

    % get predictions from GP conditioned on MAP hyperparameters
    [~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
        gp(map_hyperparameters, inference_method, mean_function, ...
           covariance_function, [], x, y, x_star, y_star);

[1]: http://latex.codecogs.com/svg.latex?p(%5Ctheta)
[2]: http://latex.codecogs.com/svg.latex?%5Ctheta
[3]: http://latex.codecogs.com/svg.latex?-%5Clog%20p(%5Ctheta)
[4]: http://latex.codecogs.com/svg.latex?%5Cnabla%20-%5Clog%20p(%5Ctheta)
[5]: http://latex.codecogs.com/svg.latex?H%5Cbigl(-%5Clog%20p(%5Ctheta)%20%5Cbigr)
[6]: http://latex.codecogs.com/svg.latex?p(%5Ctheta)%20%3D%20%5Cprod_i%20p(%5Ctheta_i)
[7]: http://latex.codecogs.com/svg.latex?%5Ctheta_i
[8]: http://latex.codecogs.com/svg.latex?p(%5Ctheta_i%20%5Cmid%20%5Cmu%2C%20%5Csigma%5E2)%20%3D%20%5Cmathcal%7BN%7D(%5Ctheta_i%3B%20%5Cmu%2C%20sigma%5E2)%20
[9]: http://latex.codecogs.com/svg.latex?p(%5Ctheta_i%20%5Cmid%20%5Cell%2C%20u)%20%3D%20%5Cmathcal%7BU%7D(%5Cell%2C%20u)
[10]: http://latex.codecogs.com/svg.latex?p(%5Ctheta_i%20%5Cmid%20%5Cmu%2C%20b)%20%3D%20%5Ctext%7BLaplace%7D(%5Cmu%2C%20b)
[11]: http://latex.codecogs.com/svg.latex?p(%5Ctheta_i)%20%3D%201
[12]: http://latex.codecogs.com/svg.latex?-%5Clog%20p(y%20%5Cmid%20X%2C%20%5Cmathcal%7BD%7D%2C%20%5Ctheta)%20-%20%5Clog%20p(%5Ctheta)
