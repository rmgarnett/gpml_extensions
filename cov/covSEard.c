/* calculate squared exponential covariance, automatic relevence determination */

#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	double *hyperparameters, *train, *test, *out, *hyperparameter;

	int num_train, num_test, dim, i, j, k;
	double *input_scales, output_scale;
	double squared_distance = 0, difference, index_distance = 0;

	/* number of hyperparameters */
  if ((nlhs <= 1) && (nrhs == 0)) {
		plhs[0] = mxCreateString("(D+1)");
		return;
	}

	train = mxGetPr(prhs[1]);
	num_train = mxGetM(prhs[1]);
	dim = mxGetN(prhs[1]);

	/* minimal error checking */
	if ((mxGetN(prhs[0]) * mxGetM(prhs[0])) != (dim + 1)) {
		mexErrMsgTxt("wrong number of hyperparameters!");
		return;
	}

	hyperparameters = mxGetPr(prhs[0]);

	input_scales = (double *)(malloc(dim * sizeof(double)));
	for (i = 0; i < dim; i++)
		input_scales[i] = exp(hyperparameters[i]);
	output_scale = exp(2 * hyperparameters[dim]);

	/* training covariance */
	if ((nrhs == 2) || ((nrhs == 3) && (mxGetNumberOfElements(prhs[2]) == 0))) {
		plhs[0] = mxCreateDoubleMatrix(num_train, num_train, mxREAL);
		out = mxGetPr(plhs[0]);
		
		for (i = 0; i < num_train; i++) {
			for (j = i; j < num_train; j++) {

				squared_distance = 0;
				for (k = 0; k < dim; k++) {
					difference = (train[i + num_train * k] - 
												train[j + num_train * k]) / input_scales[k];
					squared_distance += difference * difference;
				}

				out[i + num_train * j] = 
					output_scale * exp(-squared_distance / 2);

				/* symmetric output */
				out[j + num_train * i] = out[i + num_train * j];
			}
		}
		return;
	}
	
	/* test covariances */
	else if (nrhs == 3) {

		if (mxIsChar(prhs[2])) {

			/* self covariances */
			if (strcmp((const char *) mxArrayToString(prhs[2]), "diag") == 0) { 
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
			if (mxGetN(prhs[2]) != dim) {
				mexErrMsgTxt("training and testing points do not have the same dimension!");
				return;
			}

			test = mxGetPr(prhs[2]);
			num_test = mxGetM(prhs[2]);
			
			plhs[0] = mxCreateDoubleMatrix(num_train, num_test, mxREAL);
			out = mxGetPr(plhs[0]);

			for (i = 0; i < num_test; i++) {
				for (j = 0; j < num_train; j++) {

					squared_distance = 0;
					for (k = 0; k < dim; k++) {
						difference = (test[i + num_test * k] - train[j + num_train * k]) / 
							input_scales[k];
						squared_distance += difference * difference;
					}
					
					out[j + num_train * i] = output_scale * exp(-squared_distance / 2);
				}
			}
		}

	}

	/* derivatives with respect to hyperparamters */
	else {

		/* which hyperparameter to take derivative with respect to */
		hyperparameter = mxGetPr(prhs[3]);
		
		/* diagonal derivatives */
		if (mxIsChar(prhs[2])) {
			if (strcmp((const char *) mxArrayToString(prhs[2]), "diag") == 0) { 

				plhs[0] = mxCreateDoubleMatrix(num_train, 1, mxREAL);
				out = mxGetPr(plhs[0]);
				
				/* input scale */
				if (hyperparameter[0] <= dim) { 
					for (i = 0; i < num_train; i++) {
						out[i] = 0;
					}
					return;
				}
				/* output scale */
				else if (hyperparameter[0] == (dim + 1)) {
					for (i = 0; i < num_train; i++) {
						out[i] = 2 * output_scale;
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
			if (mxGetNumberOfElements(prhs[2]) == 0) {
				plhs[0] = mxCreateDoubleMatrix(num_train, num_train, mxREAL);
				out = mxGetPr(plhs[0]);
			
				for (i = 0; i < num_train; i++) 
					for (j = i; j < num_train; j++) {
						
						squared_distance = 0;
						for (k = 0; k < dim; k++) {
							difference = (train[i + num_train * k] - 
														train[j + num_train * k]) / input_scales[k];
							squared_distance += difference * difference;
							
							if (k == (hyperparameter[0] - 1)) {
								index_distance = difference * difference;
							}
						}
						
						/* input scale */
						if ((hyperparameter[0] > 0) && (hyperparameter[0] <= dim)) {
							out[i + num_train * j] = 
								output_scale * index_distance * exp(-squared_distance / 2);
						}
						/* output scale */
						else if (hyperparameter[0] == (dim + 1)) {
							out[i + num_train * j] = 
								2 * output_scale * exp(-squared_distance / 2);
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

				test = mxGetPr(prhs[2]);
				num_test = mxGetM(prhs[2]);
			
				plhs[0] = mxCreateDoubleMatrix(num_train, num_test, mxREAL);
				out = mxGetPr(plhs[0]);

				for (i = 0; i < num_test; i++) {
					for (j = 0; j < num_train; j++) {
						
						squared_distance = 0;
						for (k = 0; k < dim; k++) {
							difference = (test[i + num_test * k] - train[j + num_train * k]) / 
								input_scales[k];
							squared_distance += difference * difference;

							if (k == (hyperparameter[0] - 1)) {
								index_distance = difference * difference;
							}
						}
						
						/* input scale */
						if ((hyperparameter[0] > 0) && (hyperparameter[0] <= dim)) {
							out[j + num_train * i] = output_scale * index_distance *  exp(-squared_distance / 2);
						}
						/* output scale */
						else if (hyperparameter[0] == (dim + 1)) {
							out[j + num_train * i] = 2 * output_scale * exp(-squared_distance / 2);
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
