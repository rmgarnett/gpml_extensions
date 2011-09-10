/* calculate squared exponential covariance */

#include "mex.h"
#include <math.h>
#define APPROX_EQUAL(x, y, tol) (abs((x) - (y)) <= (tol))

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	double *d, *hyperparameters, *train, *test, *out, *hyperparameter;

	int num_train, num_test, dim, i, j, k;
	double input_scale, inverse_input_scale, inverse_squared_input_scale,
		SQRT_3_inverse_input_scale, SQRT_5_inverse_input_scale,
		output_scale, twice_output_scale;
	double squared_distance = 0, difference, argument;

	const double SQRT_3 = 1.732050807568877;
	const double SQRT_5 = 2.236067977499790;
	const double ONE_THIRD = 0.333333333333333;
	const double tolerance = 1e-6;

	/* number of hyperparameters */
  if ((nlhs <= 2) && (nrhs == 0)) {
		plhs[0] = mxCreateString("2");
		return;
	}

	train = mxGetPr(prhs[2]);
	num_train = mxGetM(prhs[2]);
	dim = mxGetN(prhs[2]);

	/* minimal error checking */
	if ((mxGetN(prhs[1]) * mxGetM(prhs[1])) != 2) {
		mexErrMsgTxt("wrong number of hyperparameters!");
		return;
	}
	
	d = mxGetPr(prhs[0]);
	if (!APPROX_EQUAL(d[0], 1, tolerance) && 
			!APPROX_EQUAL(d[0], 3, tolerance) &&
			!APPROX_EQUAL(d[0], 5, tolerance)) {
		mexErrMsgTxt("only 1, 3 and 5 allowed for d");
		return;
	}
	
	hyperparameters = mxGetPr(prhs[1]);
	inverse_input_scale = exp(-hyperparameters[0]);
	SQRT_3_inverse_input_scale = SQRT_3 * inverse_input_scale;
	SQRT_5_inverse_input_scale = SQRT_5 * inverse_input_scale;
	inverse_squared_input_scale = 
		1 / (inverse_input_scale * inverse_input_scale);
	output_scale = exp(2 * hyperparameters[1]);
	twice_output_scale = 2 * output_scale;

	/* training covariance */
	if ((nrhs == 3) || ((nrhs == 4) && (mxGetNumberOfElements(prhs[3]) == 0))) {
		plhs[0] = mxCreateDoubleMatrix(num_train, num_train, mxREAL);
		out = mxGetPr(plhs[0]);
		
		for (i = 0; i < num_train; i++) {
			for (j = i; j < num_train; j++) {

				squared_distance = 0;
				for (k = 0; k < dim; k++) {
					difference = (train[i + num_train * k] - train[j + num_train * k]);
					squared_distance += difference * difference;
				}
				
				if (APPROX_EQUAL(d[0], 1, tolerance)) {
					argument = inverse_input_scale * sqrt(squared_distance);
					out[i + num_train * j] = 
						output_scale * exp(-argument);
				}
				else if (APPROX_EQUAL(d[0], 3, tolerance)) {
					argument = SQRT_3_inverse_input_scale * sqrt(squared_distance);
					out[i + num_train * j] = 
						output_scale * (1 + argument) * exp(-argument);
				}
				else if (APPROX_EQUAL(d[0], 5, tolerance)) {
					argument = SQRT_5_inverse_input_scale * sqrt(squared_distance);
					out[i + num_train * j] = 
						output_scale * 
						(1 + argument + ONE_THIRD * argument * argument) * 
						exp(-argument);
				}
				
				/* symmetric output */
				out[j + num_train * i] = out[i + num_train * j];
			}
		}
		return;
	}

	/* test covariances */
	else if (nrhs == 4) {

		if (mxIsChar(prhs[3])) {

			/* self covariances */
			if (strcmp((const char *) mxArrayToString(prhs[3]), "diag") == 0) { 
				plhs[0] = mxCreateDoubleMatrix(num_train, 1, mxREAL);
				out = mxGetPr(plhs[0]);

				for (i = 0; i < num_train; i++) {
					out[i] = output_scale;
				}
				return;
			}	 
			/* a string but not 'diag' ? */
			else {
				mexErrMsgTxt("unacceptable argument, did you mean 'diag'?");
				return;
			}
		}

		/* cross covariances */
		else {

			/* more error checking */
			if (mxGetN(prhs[3]) != dim) {
				mexErrMsgTxt("training and testing points do not have the same dimension!");
				return;
			}

			test = mxGetPr(prhs[3]);
			num_test = mxGetM(prhs[3]);
			
			plhs[0] = mxCreateDoubleMatrix(num_train, num_test, mxREAL);
			out = mxGetPr(plhs[0]);

			for (i = 0; i < num_test; i++) {
				for (j = 0; j < num_train; j++) {

					squared_distance = 0;
					for (k = 0; k < dim; k++) {
						difference = (test[i + num_test * k] - train[j + num_train * k]);
						squared_distance += difference * difference;
					}
					
					if (APPROX_EQUAL(d[0], 1, tolerance)) {
						argument = inverse_input_scale * sqrt(squared_distance);
						out[j + num_train * i] = 
							output_scale * exp(-argument);
					}
					else if (APPROX_EQUAL(d[0], 3, tolerance)) {
						argument = SQRT_3_inverse_input_scale * sqrt(squared_distance);
						out[j + num_train * i] = 
							output_scale * (1 + argument) * exp(-argument);
					}
					else if (APPROX_EQUAL(d[0], 5, tolerance)) {
						argument = SQRT_5_inverse_input_scale * sqrt(squared_distance);
						out[j + num_train * i] = 
							output_scale * 
							(1 + argument + ONE_THIRD * argument * argument) * 
							exp(-argument);
					}
				}
			}
			return;
		}
	}
		
	/* derivatives with respect to hyperparamters */
	else {

		/* which hyperparameter to take derivative with respect to */
		hyperparameter = mxGetPr(prhs[4]);
			
		/* diagonal derivatives */
		if (mxIsChar(prhs[3])) {
			if (strcmp((const char *) mxArrayToString(prhs[3]), "diag") == 0) { 
					
				plhs[0] = mxCreateDoubleMatrix(num_train, 1, mxREAL);
				out = mxGetPr(plhs[0]);
					
				/* input scale */
				if (APPROX_EQUAL(hyperparameter[0], 1, tolerance)) { 
					return;
				}
				/* output scale */
				else if (APPROX_EQUAL(hyperparameter[0], 2, tolerance)) {
					for (i = 0; i < num_train; i++) {
						out[i] = twice_output_scale;
					}
					return;
				}
				else {
					mexErrMsgTxt("hyperparameter index out of range!");
					return;
				}
			}
			/* a string but not 'diag' ? */
			else {
				mexErrMsgTxt("unacceptable argument, did you mean 'diag'?");
				return;
			}
		}
		else {
			
			/* training derivatives */
			if (mxGetNumberOfElements(prhs[3]) == 0) {
				
				plhs[0] = mxCreateDoubleMatrix(num_train, num_train, mxREAL);
				out = mxGetPr(plhs[0]);
				
				for (i = 0; i < num_train; i++) 
					for (j = i; j < num_train; j++) {
						
						squared_distance = 0;
						for (k = 0; k < dim; k++) {
							difference = (train[i + num_train * k] - train[j + num_train * k]);
							squared_distance += difference * difference;
						}
						
						/* input scale */
						if (hyperparameter[0] == 1) {
							if (APPROX_EQUAL(d[0], 1, tolerance)) {
								argument = inverse_input_scale * sqrt(squared_distance);
								out[i + num_train * j] = 
									output_scale * argument * exp(-argument);
							}
							else if (APPROX_EQUAL(d[0], 3, tolerance)) {
								argument = SQRT_3_inverse_input_scale * sqrt(squared_distance);
								out[i + num_train * j] = 
									output_scale * argument * argument * exp(-argument);
							}
							else if (APPROX_EQUAL(d[0], 5, tolerance)) {
								argument = SQRT_5_inverse_input_scale * sqrt(squared_distance);
								out[i + num_train * j] = 
									output_scale * 
									argument *
									ONE_THIRD * (argument + argument * argument) * 
									exp(-argument);
							}
						}
						/* output scale */
						else if (hyperparameter[0] == 2) {
							if (APPROX_EQUAL(d[0], 1, tolerance)) {
								argument = inverse_input_scale * sqrt(squared_distance);
								out[i + num_train * j] = 
									twice_output_scale * exp(-argument);
							}
							else if (APPROX_EQUAL(d[0], 3, tolerance)) {
								argument = SQRT_3_inverse_input_scale * sqrt(squared_distance);
								out[i + num_train * j] = 
									twice_output_scale * (1 + argument) * exp(-argument);
							}
							else if (APPROX_EQUAL(d[0], 5, tolerance)) {
								argument = SQRT_5_inverse_input_scale * sqrt(squared_distance);
								out[i + num_train * j] = 
									twice_output_scale * 
									(1 + argument + ONE_THIRD * argument * argument) * 
									exp(-argument);
							}
						}
						else {
							mexErrMsgTxt("hyperparameter index out of range!");
							return;
						}

						/* symmetric output */
						out[j + num_train * i] = out[i + num_train * j];
					}
				return;
			}
			else {

				test = mxGetPr(prhs[3]);
				num_test = mxGetM(prhs[3]);
			
				plhs[0] = mxCreateDoubleMatrix(num_train, num_test, mxREAL);
				out = mxGetPr(plhs[0]);

				for (i = 0; i < num_test; i++) {
					for (j = 0; j < num_train; j++) {
						
						squared_distance = 0;
						for (k = 0; k < dim; k++) {
							difference = (test[i + num_test * k] - train[j + num_train * k]);
							squared_distance += difference * difference;
						}
						
						/* input scale */
						if (hyperparameter[0] == 1) {
							if (APPROX_EQUAL(d[0], 1, tolerance)) {
								argument = inverse_input_scale * sqrt(squared_distance);
								out[j + num_train * i] = 
									output_scale * argument * exp(-argument);
							}
							else if (APPROX_EQUAL(d[0], 3, tolerance)) {
								argument = SQRT_3_inverse_input_scale * sqrt(squared_distance);
								out[j + num_train * i] = 
									output_scale * argument * argument * exp(-argument);
							}
							else if (APPROX_EQUAL(d[0], 5, tolerance)) {
								argument = SQRT_5_inverse_input_scale * sqrt(squared_distance);
								out[j + num_train * i] = 
									output_scale * 
									ONE_THIRD * (argument + argument * argument) * 
									argument * 
									exp(-argument);
							}
						}
						/* output scale */
						else if (hyperparameter[0] == 2) {
							if (APPROX_EQUAL(d[0], 1, tolerance)) {
								argument = inverse_input_scale * sqrt(squared_distance);
								out[j + num_train * i] = 
									twice_output_scale * exp(-argument);
							}
							else if (APPROX_EQUAL(d[0], 3, tolerance)) {
								argument = SQRT_3_inverse_input_scale * sqrt(squared_distance);
								out[j + num_train * i] = 
									twice_output_scale * (1 + argument) * exp(-argument);
							}
							else if (APPROX_EQUAL(d[0], 5, tolerance)) {
								argument = SQRT_5_inverse_input_scale * sqrt(squared_distance);
								out[j + num_train * i] = 
									twice_output_scale * 
									(1 + argument + ONE_THIRD * argument * argument) * 
									exp(-argument);
							}
						}
						else {
							mexErrMsgTxt("hyperparameter index out of range!");
							return;
						}
					}
				}
			}
		}
	}
}
	
