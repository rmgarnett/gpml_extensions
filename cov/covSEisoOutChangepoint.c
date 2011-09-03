/* calculate squared exponential covariance */

#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	double *hyperparameters, *in, *out, *training_points, *testing_points, 
		*self_covariances, *cross_covariances, *hyperparameter;

	int num_points, num_training_points, num_testing_points, dim;
	int i, j, k;
	int first, second;
	double squared_distance = 0, difference;
	double input_scale, output_scales[2], output_scale;

	char *string;

	/* number of hyperparameters */
  if ((nlhs <= 1) && (nrhs == 0)) {
		plhs[0] = mxCreateString("3");
		return;
	}
	
	/* minimal error checking */
	if (mxGetM(prhs[0]) != 3 || mxGetN(prhs[0]) != 1) {
		mexErrMsgTxt("wrong number of hyperparameters!");
		return;
	}

	hyperparameters = mxGetPr(prhs[0]);
	input_scale = exp(hyperparameters[0]);
	output_scales[0] = exp(2 * hyperparameters[1]);
	output_scales[1] = exp(2 * hyperparameters[2]);

	/* training covariance */
	if ((nrhs == 2) || 
			((nrhs == 3) && (mxGetM(prhs[2]) == 0))) {

		num_points = mxGetM(prhs[1]);
		dim = mxGetN(prhs[1]) - 1;
		in = mxGetPr(prhs[1]);

		plhs[0] = mxCreateDoubleMatrix(num_points, num_points, mxREAL);
		out = mxGetPr(plhs[0]);
		
		for (i = 0; i < num_points; i++) {
			for (j = i; j < num_points; j++) {
				
				first = (int)(in[i + num_points * dim]);
				second = (int)(in[j + num_points * dim]);

				squared_distance = 0;
				for (k = 0; k < dim; k++) {
					difference = 
						(in[i + num_points * k] - in[j + num_points * k]) / 
						input_scale;
					squared_distance += difference * difference;
				}

				output_scale = sqrt(output_scales[first] * 
														output_scales[second]);

				/* symmetric output */
				out[i + num_points * j] = output_scale * exp(-squared_distance / 2);
				out[j + num_points * i] = out[i + num_points * j];
			}
		}
		return;
	}
	
	/* test covariances */
	else if (nrhs == 3) {

		num_training_points = mxGetM(prhs[1]);
		dim = mxGetN(prhs[1]) - 1;
		training_points = mxGetPr(prhs[1]);

		if (mxIsChar(prhs[2])) {
			string = mxArrayToString(prhs[2]);
			if (strcmp((const char *) string, "diag") == 0) { 
				plhs[0] = mxCreateDoubleMatrix(num_training_points, 1, mxREAL);
				self_covariances = mxGetPr(plhs[0]);

				for (i = 0; i < num_training_points; i++)
					self_covariances[i] =
						output_scales[(int)(training_points[i + num_training_points * dim])];
				
				return;
			}	 
			else {
				mexErrMsgTxt("third parameter must be a matrix or 'diag'!");
				return;		
			}
		}
		else {
			num_testing_points = mxGetM(prhs[2]);
			testing_points = mxGetPr(prhs[2]);
			
			plhs[0] = 
				mxCreateDoubleMatrix(num_training_points, num_testing_points, mxREAL);
			cross_covariances = mxGetPr(plhs[0]);

			for (i = 0; i < num_testing_points; i++) {
				for (j = 0; j < num_training_points; j++) {

					first = (int)(testing_points[i + num_testing_points * dim]);
					second = (int)(training_points[j + num_training_points * dim]);

					squared_distance = 0;
					for (k = 0; k < dim; k++) {
						difference = 
							(testing_points[i + num_testing_points * k] - 
							 training_points[j + num_training_points * k]) / 
							input_scale;
						squared_distance += difference * difference;
					}

					output_scale = sqrt(output_scales[first] * 
															output_scales[second]);
					
					cross_covariances[j + num_training_points * i] = 
						output_scale * exp(-squared_distance / 2);
				}
			}
			return;
		}
		
	}

	/* derivatives with respect to hyperparamters */
	else {
		num_points = mxGetM(prhs[1]);
		dim = mxGetN(prhs[1]) - 1;
		in = mxGetPr(prhs[1]);

		hyperparameter = mxGetPr(prhs[3]);
		
		plhs[0] = mxCreateDoubleMatrix(num_points, num_points, mxREAL);
		out = mxGetPr(plhs[0]);
		
		/* first input scale */
 		if (hyperparameter[0] == 1) { 

			for (i = 0; i < num_points; i++) 
				for (j = i; j < num_points; j++) {
					
					first = (int)(in[i + num_points * dim]);
					second = (int)(in[j + num_points * dim]);

					squared_distance = 0;
					for (k = 0; k < dim; k++) {
						difference = 
							(in[i + num_points * k] - in[j + num_points * k]) / 
							input_scale;
						squared_distance += difference * difference;
					}

					output_scale = sqrt(output_scales[first] * 
															output_scales[second]);
					
					/* symmetric output */
					out[i + num_points * j] = 
						output_scale * squared_distance * exp(-squared_distance / 2);
					out[j + num_points * i] = out[i + num_points * j];
				
				}

			return;
		}

		/* first output scale */
 		else if (hyperparameter[0] == 2) { 

			for (i = 0; i < num_points; i++) 
				for (j = i; j < num_points; j++) {
					
					first = (int)(in[i + num_points * dim]);
					second = (int)(in[j + num_points * dim]);
					
					if ((first == 1) && (second == 1)) {
						output_scale = 0;
						continue;
					}
					else {

						squared_distance = 0;
						for (k = 0; k < dim; k++) {
							difference = 
								(in[i + num_points * k] - in[j + num_points * k]) / 
								input_scale;
							squared_distance += difference * difference;
						}

						output_scale = sqrt(output_scales[first] * 
																output_scales[second]);
					}

					/* symmetric output */
					out[i + num_points * j] = output_scale * exp(-squared_distance / 2);
					out[j + num_points * i] = out[i + num_points * j];
				}
			
			return;
		}

		/* second output scale */
 		else if (hyperparameter[0] == 3) { 

			for (i = 0; i < num_points; i++) 
				for (j = i; j < num_points; j++) {
					
					first = (int)(in[i + num_points * dim]);
					second = (int)(in[j + num_points * dim]);
					
					if ((first == 0) && (second == 0)) {
						output_scale = 0;
						continue;
					}
					else {

						squared_distance = 0;
						for (k = 0; k < dim; k++) {
							difference = 
								(in[i + num_points * k] - in[j + num_points * k]) / 
								input_scale;
							squared_distance += difference * difference;
						}

						output_scale = sqrt(output_scales[first] * 
																output_scales[second]);
					}

					/* symmetric output */
					out[i + num_points * j] = output_scale * exp(-squared_distance / 2);
					out[j + num_points * i] = out[i + num_points * j];
				}
			
			return;
		}
		
		return;
	}
}
