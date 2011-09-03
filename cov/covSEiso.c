/* calculate squared exponential covariance */

#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	/* number of hyperparameters */
  if ((nlhs <= 1) && (nrhs == 0)) {
		plhs[0] = mxCreateString("2");
		return;
	}

	/* training covariance */
	else if (nrhs == 2) {
		double *hyperparameters, *in, *out;

		int num_points, dim, i, j, k;
		double input_scale, output_scale;
		double squared_distance = 0, difference;

		/* minimal error checking */
		if (mxGetM(prhs[0]) != 2 || mxGetN(prhs[0]) != 1) {
			mexErrMsgTxt("wrong number of hyperparameters!");
			return;
		}

		num_points = mxGetM(prhs[1]);
		dim = mxGetN(prhs[1]);
		in = mxGetPr(prhs[1]);

		/* grab hyperparamters, first is log(input_scale), second is
			 log(output_scale) */
		hyperparameters = mxGetPr(prhs[0]);
		input_scale = exp(hyperparameters[0]);
		output_scale = exp(2 * hyperparameters[1]);
		
		plhs[0] = mxCreateDoubleMatrix(num_points, num_points, mxREAL);
		out = mxGetPr(plhs[0]);
		
		for (i = 0; i < num_points; i++) {
			for (j = i; j < num_points; j++) {

				squared_distance = 0;
				for (k = 0; k < dim; k++) {
					difference = (in[i + num_points * k] - 
												in[j + num_points * k]) / input_scale;
					squared_distance += difference * difference;
				}

				/* symmetric output */
				out[i + num_points * j] = 
					output_scale * exp(-squared_distance / 2);
				out[j + num_points * i] = out[i + num_points * j];
			}
		}
		return;
	}
	
	/* test covariances */
	else if (nrhs == 3) {

		double *hyperparameters, *training_points, *testing_points, 
			*self_covariances, *cross_covariances;

		int num_training_points, num_testing_points, dim, i, j, k;
		double input_scale, output_scale;
		double squared_distance = 0, difference;

		char *string;

		/* minimal error checking */
		if (mxGetM(prhs[0]) != 2 || mxGetN(prhs[0]) != 1) {
			mexErrMsgTxt("wrong number of hyperparameters!");
			return;
		}

		num_training_points = mxGetM(prhs[1]);
		dim = mxGetN(prhs[1]);
		training_points = mxGetPr(prhs[1]);

		/* grab hyperparamters, first is log(input_scale), second is
			 log(output_scale) */
		hyperparameters = mxGetPr(prhs[0]);
		input_scale = exp(hyperparameters[0]);
		output_scale = exp(2 * hyperparameters[1]);

		if (mxIsChar(prhs[2])) {
			string = mxArrayToString(prhs[2]);
			if (strcmp((const char *) string, "diag") == 0) { 
				plhs[0] = mxCreateDoubleMatrix(num_training_points, 1, mxREAL);
				self_covariances = mxGetPr(plhs[0]);

				for (i = 0; i < num_training_points; i++)
					self_covariances[i] = output_scale;

				return;
			}	 
		}
		else {
			num_testing_points = mxGetM(prhs[2]);
			testing_points = mxGetPr(prhs[2]);
			
			plhs[0] = 
				mxCreateDoubleMatrix(num_training_points, num_testing_points, mxREAL);
			cross_covariances = mxGetPr(plhs[0]);

			/* more error checking */
			if (mxGetN(prhs[2]) != dim) {
				mexErrMsgTxt("training and testing points do not have the same dimension!");
				return;
			}

			for (i = 0; i < num_testing_points; i++) {
				for (j = 0; j < num_training_points; j++) {

					squared_distance = 0;
					for (k = 0; k < dim; k++) {
						difference = (testing_points[i + num_testing_points * k] - 
													training_points[j + num_training_points * k]) /
							input_scale;
						squared_distance += difference * difference;
					}
					
					cross_covariances[j + num_training_points * i] = 
						output_scale * exp(-squared_distance / 2);
				}
			}
			return;
		}
		
	}

	/* derivatives with respect to hyperparamters */
	else {
		double *hyperparameter, *hyperparameters, *in, *out;

		int num_points, dim, i, j, k;
		double input_scale, output_scale;
		double squared_distance = 0, difference;

		num_points = mxGetM(prhs[1]);
		dim = mxGetN(prhs[1]);
		in = mxGetPr(prhs[1]);

		/* minimal error checking */
		if (mxGetM(prhs[0]) != 2 || mxGetN(prhs[0]) != 1) {
			mexErrMsgTxt("wrong number of hyperparameters!");
			return;
		}

		/* grab hyperparamters, first is log(input_scale), second is
			 log(output_scale) */
		hyperparameters = mxGetPr(prhs[0]);
		input_scale = exp(hyperparameters[0]);
		output_scale = exp(2 * hyperparameters[1]);

		hyperparameter = mxGetPr(prhs[3]);
		
		plhs[0] = mxCreateDoubleMatrix(num_points, num_points, mxREAL);
		out = mxGetPr(plhs[0]);
		
		/* input scale */
 		if (hyperparameter[0] == 1) { 

			for (i = 0; i < num_points; i++) 
				for (j = i; j < num_points; j++) {
					
					squared_distance = 0;
					for (k = 0; k < dim; k++) {
						difference = (in[i + num_points * k] - 
													in[j + num_points * k]) / input_scale;
						squared_distance += difference * difference;
					}
					
					/* symmetric output */
					out[i + num_points * j] = 
						output_scale * squared_distance * exp(-squared_distance / 2);
					out[j + num_points * i] = out[i + num_points * j];
					
				}
		}

		/* output scale */
 		else { 

			for (i = 0; i < num_points; i++) 
				for (j = i; j < num_points; j++) {
					
					squared_distance = 0;
					for (k = 0; k < dim; k++) {
						difference = (in[i + num_points * k] - 
													in[j + num_points * k]) / input_scale;
						squared_distance += difference * difference;
					}
						
					/* symmetric output */
					out[i + num_points * j] = 
						2 * output_scale * exp(-squared_distance / 2);
					out[j + num_points * i] = out[i + num_points * j];
				}
			return;
		}
		
		return;
	}
}
