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
	double squared_distance = 0, difference, first, second;
	double input_scales[2], output_scales[2], output_scale, split;

	char *string;

	/* number of hyperparameters */
  if ((nlhs <= 1) && (nrhs == 0)) {
		plhs[0] = mxCreateString("5");
		return;
	}
	
	/* minimal error checking */
	if (mxGetM(prhs[0]) != 5 || mxGetN(prhs[0]) != 1) {
		mexErrMsgTxt("wrong number of hyperparameters!");
		return;
	}

	hyperparameters = mxGetPr(prhs[0]);
	input_scales[0] = exp(hyperparameters[0]);
	output_scales[0] = exp(2 * hyperparameters[1]);
	input_scales[1] = exp(hyperparameters[2]);
	output_scales[1] = exp(2 * hyperparameters[3]);
	split = hyperparameters[4];

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
				
				squared_distance = 0;

				first = in[i + num_points * dim];
				second = in[j + num_points * dim];

				if ((first < split) && (second < split)) {
					for (k = 0; k < dim; k++) {
						difference = 
							(in[i + num_points * k] - in[j + num_points * k]) / 
							input_scales[0];
						squared_distance += difference * difference;
					}
					output_scale = output_scales[0];
				}
				else if ((first >= split) && (second >= split)) {
					for (k = 0; k < dim; k++) {
						difference = 
							(in[i + num_points * k] - in[j + num_points * k]) / 
							input_scales[1];
						squared_distance += difference * difference;
					}
					output_scale = output_scales[1];
				}
				else if ((first < split) && (second >= split)) {
					for (k = 0; k < dim; k++) {
						difference = ((split - in[i + num_points * k]) / 
													input_scales[0] +
													(in[j + num_points * k] - split) / 
													input_scales[1]);
						squared_distance += difference * difference;
					}
					output_scale = sqrt(output_scales[0] * output_scales[1]);
				}
				else {
					for (k = 0; k < dim; k++) {
						difference = ((in[i + num_points * k] - split) / 
													input_scales[1] +
													(split - in[j + num_points * k]) / 
													input_scales[0]);
						squared_distance += difference * difference;
					}
					output_scale = sqrt(output_scales[0] * output_scales[1]);
				}

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
					if ((training_points[i + num_training_points * dim]) < split)
						self_covariances[i] = output_scales[0];
					else
						self_covariances[i] = output_scales[1];

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

					squared_distance = 0;

					first = testing_points[i + num_testing_points * dim];
					second = training_points[j + num_training_points * dim];

					if ((first < split) && (second < split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								(testing_points[i + num_testing_points * k] - 
								 training_points[j + num_training_points * k]) / 
								input_scales[0];
							squared_distance += difference * difference;
						}
						output_scale = output_scales[0];
					}
					else if ((first >= split) && (second >= split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								(testing_points[i + num_testing_points * k] - 
								 training_points[j + num_training_points * k]) / 
								input_scales[1];
							squared_distance += difference * difference;
						}
						output_scale = output_scales[1];
					}
					else if ((first < split) && (second >= split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								((split - testing_points[i + num_testing_points * k]) / 
								 input_scales[0] +
								 (training_points[j + num_training_points * k] - split) / 
								 input_scales[1]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);
					}
					else {
						for (k = 0; k < dim; k++) {
							difference = 
								((testing_points[i + num_testing_points * k] - split) / 
								 input_scales[1] +
								 (split - training_points[j + num_training_points * k]) / 
								 input_scales[0]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);
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
					
					squared_distance = 0;
				
					first = in[i + num_points * dim];
					second = 	in[j + num_points * dim];

					if ((first < split) && (second < split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								(in[i + num_points * k] - in[j + num_points * k]) / 
								input_scales[0];
							squared_distance += difference * difference;
						}
						output_scale = output_scales[0];

						/* symmetric output */
						out[i + num_points * j] = 
							output_scale * squared_distance * exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];

					}
					else if ((first >= split) && (second >= split)) {
						/* symmetric output */
						out[i + num_points * j] = 0;
						out[j + num_points * i] = out[i + num_points * j];

						continue;
					}
					else if ((first < split) && (second >= split)) {
						for (k = 0; k < dim; k++) {
							difference = ((split - in[i + num_points * k]) / input_scales[0] +
														(in[j + num_points * k] - split) / input_scales[1]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);

						/* symmetric output */
						out[i + num_points * j] = 
							output_scale * 
							((split - in[i + num_points * k]) * 
							 (split - in[i + num_points * k]) / 
							 input_scales[0] / input_scales[0] + 
							 (split - in[i + num_points * k]) * 
							 (in[j + num_points * k] - split) / 
							 input_scales[0] / input_scales[1]) *
							exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
					else {
						for (k = 0; k < dim; k++) {
							difference = 
								((in[i + num_points * k] - split) / input_scales[1] +
								 (split - in[j + num_points * k]) / input_scales[0]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);

						/* symmetric output */
						out[i + num_points * j] = 
							output_scale * 
							((in[i + num_points * k] - split) * 
							 (split - in[j + num_points * k]) / 
							 input_scales[0] / input_scales[1] + 
							 (split - in[j + num_points * k]) * 
							 (split - in[j + num_points * k]) / 
							 input_scales[0] / input_scales[0]) *
							exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
				
				}
		}

		/* first output scale */
 		else if (hyperparameter[0] == 2) { 

			for (i = 0; i < num_points; i++) 
				for (j = i; j < num_points; j++) {
					
					first = in[i + num_points * dim];
					second = 	in[j + num_points * dim];

					squared_distance = 0;

					if ((first < split) && (second < split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								(in[i + num_points * k] - in[j + num_points * k]) / 
								input_scales[0];
							squared_distance += difference * difference;
						}
						output_scale = output_scales[0];

						/* symmetric output */
						out[i + num_points * j] = 
							2 * output_scale * exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];

					}
					else if ((first >= split) && (second >= split)) {
						/* symmetric output */
						out[i + num_points * j] = 0;
						out[j + num_points * i] = out[i + num_points * j];

						continue;
					}
					else if ((first < split) && (second >= split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								((split - in[i + num_points * k]) / input_scales[0] +
								 (in[j + num_points * k] - split) / input_scales[1]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);

						/* symmetric output */
						out[i + num_points * j] = output_scale * exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
					else {
						for (k = 0; k < dim; k++) {
							difference = 
								((in[i + num_points * k] - split) / input_scales[1] +
								 (split - in[j + num_points * k]) / input_scales[0]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);

						/* symmetric output */
						out[i + num_points * j] = output_scale * exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
				}
			return;
		}

		/* second input scale */
 		if (hyperparameter[0] == 3) { 

			for (i = 0; i < num_points; i++) 
				for (j = i; j < num_points; j++) {
					
					squared_distance = 0;
				
					first = in[i + num_points * dim];
					second = 	in[j + num_points * dim];

					if ((first < split) && (second < split)) {
						/* symmetric output */
						out[i + num_points * j] = 0;
						out[j + num_points * i] = out[i + num_points * j];

						continue;
					}
					else if ((first >= split) && (second >= split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								(in[i + num_points * k] - in[j + num_points * k]) / 
								input_scales[0];
							squared_distance += difference * difference;
						}
						output_scale = output_scales[0];

						/* symmetric output */
						out[i + num_points * j] = 
							output_scale * squared_distance * exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
					else if ((first < split) && (second >= split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								((split - in[i + num_points * k]) / input_scales[0] +
								 (in[j + num_points * k] - split) / input_scales[1]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);

						/* symmetric output */
						out[i + num_points * j] = 
							output_scale * 
							((split - in[i + num_points * k]) * 
							 (split - in[i + num_points * k]) / 
							 input_scales[0] / input_scales[1] + 
							 (split - in[i + num_points * k]) * 
							 (in[j + num_points * k] - split) / 
							 input_scales[1] / input_scales[1]) *
							exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
					else {
						for (k = 0; k < dim; k++) {
							difference = ((in[i + num_points * k] - split) / input_scales[1] +
														(split - in[j + num_points * k]) / input_scales[0]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);

						/* symmetric output */
						out[i + num_points * j] = 
							output_scale * 
							((in[i + num_points * k] - split) * 
							 (in[i + num_points * k] - split) / 
							 input_scales[1] / input_scales[1] + 
							 (split - in[j + num_points * k]) * 
							 (in[i + num_points * k] - split) / 
							 input_scales[0] / input_scales[1]) *
							exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
				
				}
		}

		/* second output scale */
 		else if (hyperparameter[0] == 4) { 

			for (i = 0; i < num_points; i++) 
				for (j = i; j < num_points; j++) {
					
					first = in[i + num_points * dim];
					second = 	in[j + num_points * dim];

					squared_distance = 0;

					if ((first < split) && (second < split)) {
						/* symmetric output */
						out[i + num_points * j] = 0;
						out[j + num_points * i] = out[i + num_points * j];
						
						continue;
					}
					else if ((first >= split) && (second >= split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								(in[i + num_points * k] - in[j + num_points * k]) / 
								input_scales[0];
							squared_distance += difference * difference;
						}
						output_scale = output_scales[0];

						/* symmetric output */
						out[i + num_points * j] = 
							2 * output_scale * exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];

					}
					else if ((first < split) && (second >= split)) {
						for (k = 0; k < dim; k++) {
							difference = 
								((split - in[i + num_points * k]) / input_scales[0] +
								 (in[j + num_points * k] - split) / input_scales[1]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);

						/* symmetric output */
						out[i + num_points * j] = output_scale * exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
					else {
						for (k = 0; k < dim; k++) {
							difference = 
								((in[i + num_points * k] - split) / input_scales[1] +
								 (split - in[j + num_points * k]) / input_scales[0]);
							squared_distance += difference * difference;
						}
						output_scale = sqrt(output_scales[0] * output_scales[1]);

						/* symmetric output */
						out[i + num_points * j] = output_scale * exp(-squared_distance / 2);
						out[j + num_points * i] = out[i + num_points * j];
					}
				}
			return;
		}
		
		return;
	}
}
