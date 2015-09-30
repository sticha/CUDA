// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2015, September 7 - October 6
// ###
// ###
// ### Thomas Moellenhoff, Robert Maier, Caner Hazirbas
// ###
// ###
// ###


#include "helper.h"
#include "divergence.h"
#include "projections.h"
#include "imageTransform.h"
#include "update.h"
#include "updateSuperResolution.h"
#include "flow_color.h"
#include "energy.h"
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;


// Default input parameters
const string stdImgPath = "../../material/Images/";
const string stdImgName = "carwide_";
const string stdImgType = "png";
const int stdNumDigits = 2;
const int stdNumImgs = 2;
const int stdStartImg = 1;

// uncomment to use the camera
// #define CAMERA

// uncomment to compute flow field energy
// #define FLOW_ENERGY

// uncomment to compute super resolution energy
// #define SUPER_ENERGY


// struct to transport the pointer for data access on GPU memory
struct Data {

	float**	 d_f;		// [w_small * h_small * nc]: first low resolution input image f1

	float**	 d_u;		// [w * h * nc]: first high resolution output image u1 (also used for intermediate results in optimization process)
	float*   d_Au;		// [w_small * h_small * nc]: Blurred and downsampled version of a high resolution image used for one update step

	float**	 d_v1;		// [w * h]: x-direction of the final flow field v1 (also used for intermediate results in optimization process)
	float**	 d_v2;		// [w * h]: y-direction of the final flow field v2 (also used for intermediate results in optimization process)
	
	float**	 d_b;		// [w * h * nc]: defined as b = u2 - u1 for fix u1, u2 while flow field optimization
	float2** d_A;		// [w * h * nc * 2]: defined as A = gradient(u2) for fix d_u2 while flow field optimization

	float**	 d_v_p;		// [w * h * nc]: dual variable p used for maximization of <p, Av + b> in flow field optimization
	float2** d_v_q1;		// [w * h * 2]: dual variable q1 used for maximization of <q1, gradient(v1)> in flow field optimization
	float2** d_v_q2;		// [w * h * 2]: dual variable q2 used for maximization of <q2, gradient(v2)> in flow field optimization
	
	float**	 d_u_p;		// [w_small * h_small * nc]: dual variable p1 used for maximization of <p1, Au1 - f1> in super resolution optimization
	float**  d_u_Atp;	// [w * h * nc]: Upsampled and Blurred version of p1
	float2** d_u_q;		// [w * h * nc * 2]: dual variable q1 used for maximization of <q1, gradient(u1)> in super resolution optimization

	float**	 d_u_r;		// [w * h * nc]: dual variable r used for maximization of <r, Bu> in super resolution optimization

	float*   d_temp_big;	// [w * h * nc]: Intermediate result of a big sized image

	float**  d_flow;		// [(w + 2 * border) * (h + 2 * border) * 3]: stores the color coded final flow field as an output image


#if defined(FLOW_ENERGY) || defined(SUPER_ENERGY)
	float* d_energy;	// stores in a single value the energy of the previous calculated flow field
#endif

};


// Functions for GPU calculations

// Allocate memory on GPU for all arrays that are used for calculation
void allocateGPUMemory(Data& data, int numImgs, int w, int h, int w_small, int h_small, int nc, int colorBorder) {
	// Helper values
	size_t n_small = w_small*h_small*nc;
	size_t n = w*h*nc;
	int wborder = w + 2 * colorBorder;
	int hborder = h + 2 * colorBorder;

	cudaMalloc(&data.d_temp_big, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_Au, n_small*sizeof(float));
	CUDA_CHECK;

#if defined(FLOW_ENERGY) || defined(SUPER_ENERGY)
	cudaMalloc(&data.d_energy, sizeof(float));
	CUDA_CHECK;
#endif

	data.d_f = new float*[numImgs];
	data.d_u = new float*[numImgs];
	data.d_v1 = new float*[numImgs - 1];
	data.d_v2 = new float*[numImgs - 1];
	data.d_b = new float*[numImgs - 1];
	data.d_A = new float2*[numImgs-1];
	data.d_v_p = new float*[numImgs - 1];
	data.d_v_q1 = new float2*[numImgs - 1];
	data.d_v_q2 = new float2*[numImgs - 1];
	data.d_u_p = new float*[numImgs];
	data.d_u_Atp = new float*[numImgs];
	data.d_u_q = new float2*[numImgs];
	data.d_u_r = new float*[numImgs - 1];
	data.d_flow = new float*[numImgs];


	for (int i = 0; i < numImgs; i++) {
		// # Allocate GPU memory
		cudaMalloc(&data.d_f[i], n_small*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&data.d_u[i], n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&data.d_u_p[i], n_small*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&data.d_u_Atp[i], n*sizeof(float));
		CUDA_CHECK;	
		cudaMalloc(&data.d_u_q[i], n*sizeof(float2));
		CUDA_CHECK;
	}

	for (int i = 0; i < numImgs - 1; i++) {
		cudaMalloc(&data.d_v1[i], w*h*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&data.d_v2[i], w*h*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&data.d_b[i], n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&data.d_v_p[i], n*sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&data.d_A[i], n*sizeof(float2));
		CUDA_CHECK;
		cudaMalloc(&data.d_v_q1[i], w*h*sizeof(float2));
		CUDA_CHECK;
		cudaMalloc(&data.d_v_q2[i], w*h*sizeof(float2));
		CUDA_CHECK;
		cudaMalloc(&data.d_flow[i], wborder*hborder * 3 * sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&data.d_u_r[i], n*sizeof(float));
		CUDA_CHECK;
	}
	
}

// Initializes the arrays on GPU memory for optimization process
void InitializeGPUData(float** f, Data& data, int numImgs, int w, int h, int w_small, int h_small, int nc) {
	// Helper values
	size_t n_small = w_small*h_small*nc;
	size_t n = w * h * nc;
	
	// Calculate grid size
	dim3 block3d = dim3(16, 16, nc);
	dim3 grid3d = dim3((w + block3d.x - 1) / block3d.x, (h + block3d.y - 1) / block3d.y, 1);

	int smBytes = (block3d.x + 4) * (block3d.y + 4) * sizeof(float);
	for (int i = 0; i < numImgs; i++) {
		cudaMemset(data.d_u_p[i], 0, n_small*sizeof(float));
		CUDA_CHECK;
		// Copy images to GPU
		cudaMemcpy(data.d_f[i], f[i], n_small * sizeof(float), cudaMemcpyHostToDevice);
		CUDA_CHECK;
		// Upsample f to v_p (temporary result) and blur v_p to u
		initialUpsample<<<grid3d, block3d>>>(data.d_f[i], data.d_temp_big, w, h, w_small, h_small);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		gaussBlur5<<<grid3d, block3d, smBytes>>>(data.d_temp_big, data.d_u[i], w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;

		cudaMemset(data.d_u_q[i], 0, n*sizeof(float2));
		CUDA_CHECK;
	}

	for (int i = 0; i < numImgs - 1; i++) {
		// Fill arrays with 0
		cudaMemset(data.d_v1[i], 0, w*h*sizeof(float));
		CUDA_CHECK;
		cudaMemset(data.d_v2[i], 0, w*h*sizeof(float));
		CUDA_CHECK;
		// Initialize dual variables
		cudaMemset(data.d_v_p[i], 0, w*h*nc*sizeof(float));
		CUDA_CHECK;
		cudaMemset(data.d_v_q1[i], 0, w*h*sizeof(float2));
		CUDA_CHECK;
		cudaMemset(data.d_v_q2[i], 0, w*h*sizeof(float2));
		CUDA_CHECK;
		cudaMemset(data.d_u_r[i], 0, n*sizeof(float));
		CUDA_CHECK;
	}	
}

// Free all allocated GPU memory
void freeGPUMemory(Data& data, int numImgs) {
	cudaFree(data.d_Au);
	cudaFree(data.d_temp_big);
	for (int i = 0; i < numImgs; i++) {
		cudaFree(data.d_f[i]);
		cudaFree(data.d_u[i]);
		cudaFree(data.d_u_p[i]);
		cudaFree(data.d_u_Atp[i]);
		cudaFree(data.d_u_q[i]);
	}
	for (int i = 0; i < numImgs - 1; i++) {
		cudaFree(data.d_b[i]);
		cudaFree(data.d_v1[i]);
		cudaFree(data.d_v2[i]);
		cudaFree(data.d_v_p[i]);
		cudaFree(data.d_A[i]);
		cudaFree(data.d_v_q1[i]);
		cudaFree(data.d_v_q2[i]);
		cudaFree(data.d_flow[i]);
		cudaFree(data.d_u_r[i]);
	}
	
#if defined(FLOW_ENERGY) || defined(SUPER_ENERGY)
	cudaFree(data.d_energy);
#endif
	
	CUDA_CHECK;
}

// Computes the flow field (v1, v2) for fixed images u1 and u2
void calculateFlow(Data& data, int numImgs, float gamma, int iterations, int w, int h, int nc) {
	// Calculate grid size
	dim3 block3d = dim3(16, 16, nc);
	dim3 grid3d = dim3((w + block3d.x - 1) / block3d.x, (h + block3d.y - 1) / block3d.y, 1);

	dim3 block2d = dim3(16, 16, 1);
	dim3 grid2d = dim3((w + block2d.x - 1) / block2d.x, (h + block2d.y - 1) / block2d.y, 1);

#ifdef FLOW_ENERGY
	dim3 block1d = dim3(128, 1, 1);
	dim3 grid1d = dim3((w*h + block1d.x - 1) / block1d.x, 1, 1);
	int bytesSM1d = block1d.x * sizeof(float);
#endif

	for (int i = 0; i < numImgs - 1; i++) {
		// Compute b = u2 - u1
		imageDiff<<<grid3d, block3d>>>(data.d_u[i], data.d_u[i + 1], data.d_b[i], w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		// Compute A = gradient(u2)
		calculateGradientCD<<<grid3d, block3d>>>(data.d_u[i + 1], data.d_A[i], w, h, nc);
		cudaDeviceSynchronize();
		CUDA_CHECK;
	}

	// Step sizes
	float sigmaQ = 0.5f;

	// Update in an alternating fashion the dual variables p, q1, q2 and the primal variable (flow field) v
	for (int i = 0; i < iterations; i++) {
		
		for (int i = 0; i < numImgs - 1; i++) {
			// Update dual variable p
			flow_updateP<<<grid3d, block3d>>>(data.d_v_p[i], data.d_v1[i], data.d_v2[i], data.d_A[i], data.d_b[i], gamma, w, h);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Update dual variable q1
			flow_updateQ<<<grid2d, block2d>>>(data.d_v_q1[i], data.d_v1[i], sigmaQ, w, h);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Update dual variable q2
			flow_updateQ<<<grid2d, block2d>> >(data.d_v_q2[i], data.d_v2[i], sigmaQ, w, h);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Update flow field v
			flow_updateV<<<grid2d, block2d>>>(data.d_v1[i], data.d_v2[i], data.d_v_p[i], data.d_v_q1[i], data.d_v_q2[i], data.d_A[i], w, h, nc);
			cudaDeviceSynchronize();
			CUDA_CHECK; 
		}

#ifdef FLOW_ENERGY
		// Compute energy of latest flow field
		cudaMemset(data.d_energy, 0, sizeof(float));
		CUDA_CHECK;
		flowFieldEnergy<<<grid1d, block1d, bytesSM1d>>>(data.d_energy, data.d_A, data.d_b, data.d_v1, data.d_v2, gamma, w, h, nc);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		float energy;
		cudaMemcpy(&energy, data.d_energy, sizeof(float), cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		cout << "Flow field energy in iteration " << i << ": " << energy << endl;
#endif
	}
}

// Computes super resolution images u1, u2 for fixed flow field (v1, v2)
void calculateSuperResolution(Data& data, int numImgs, int iterations, float alpha, float beta, float gamma, int w, int h, int w_small, int h_small, int nc) {
	// Helper values
	int n = w*h*nc;
	int n_small = w_small*h_small*nc;

	// Calculate grid size
	dim3 block3d = dim3(16, 16, nc);
	dim3 grid3d = dim3((w + block3d.x - 1) / block3d.x, (h + block3d.y - 1) / block3d.y, 1);
	dim3 grid3d_small = dim3((w_small + block3d.x - 1) / block3d.x, (h_small + block3d.y - 1) / block3d.y, 1);

	dim3 block2d = dim3(16, 16, 1);
	dim3 grid2d = dim3((w + block2d.x - 1) / block2d.x, (h + block2d.y - 1) / block2d.y, 1);

	int smBytes = (block3d.x + 4) * (block3d.y + 4) * sizeof(float);

#ifdef SUPER_ENERGY
	dim3 block1d = dim3(128, 1, 1);
	dim3 grid1d = dim3((w*h + block1d.x - 1) / block1d.x, 1, 1);
	int bytesSM1d = block1d.x * sizeof(float);
#endif

	// Step sizes
	float sigmaP = 1.0f;
	float sigmaQ = 0.5f;

	// Update in an alternating fashion the dual variables p1, p2, q1, q2, r and the primal variables (super resolution images) u1, u2
	for (int i = 0; i < iterations; i++) {
		for (int i = 0; i < numImgs; i++) {
			// Blur u1
			gaussBlur5<<<grid3d, block3d, smBytes>>>(data.d_u[i], data.d_temp_big, w, h);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Downsample blurred u1
			downsample<<<grid3d_small, block3d>>>(data.d_temp_big, data.d_Au, w, h, w_small, h_small);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Update dual variable p1
			super_updateP<<<grid3d_small, block3d>>>(data.d_u_p[i], data.d_f[i], data.d_Au, sigmaP, alpha, w_small, h_small);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Update dual variable q1
			super_updateQ<<<grid3d, block3d>>>(data.d_u_q[i], data.d_u[i], sigmaQ, beta, w, h);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Upsample p1
			upsample<<<grid3d, block3d>>>(data.d_u_p[i], data.d_temp_big, w, h, w_small, h_small);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Blur upsampled p1
			gaussBlur5<<<grid3d, block3d, smBytes>>>(data.d_temp_big, data.d_u_Atp[i], w, h);
			cudaDeviceSynchronize();
			CUDA_CHECK;
		}
		for(int i = 0; i < numImgs - 1;i++){
			// Update dual variable r
			super_updateR<<<grid3d, block3d>>>(data.d_u_r[i], data.d_u[i], data.d_u[i+1], data.d_v1[i], data.d_v2[i], gamma, w, h);
			cudaDeviceSynchronize();
			CUDA_CHECK;
			// Update super resolution images u1, u2
			super_updateU<<<grid3d, block3d>>>(data.d_u[i], data.d_u[i+1], data.d_u_r[i], data.d_u_Atp[i], data.d_u_Atp[i+1], data.d_u_q[i], data.d_u_q[i+1], data.d_v1[i], data.d_v2[i], w, h);
			cudaDeviceSynchronize();
			CUDA_CHECK;
		}
		
		
#ifdef SUPER_ENERGY
		// Compute energy of latest super resolution
		cudaMemset(data.d_energy, 0, sizeof(float));
		CUDA_CHECK;
		superResolutionEnergy<<<grid1d, block1d, bytesSM1d>>>(data.d_energy, data.d_u1, data.d_u2, data.d_f1, data.d_f2, data.d_v1, data.d_v2, alpha, beta, gamma, w, h, nc);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		float energy;
		cudaMemcpy(&energy, data.d_energy, sizeof(float), cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		cout << "Super resolution energy in iteration " << i << ": " << energy << endl;
#endif
	}
}

// Get the results from the calculation
void getComputationResult(Data& data, int numImgs, float** v1, float** v2, float** flow, float** sr, int w, int h, int nc, int colorBorder) {
	// Helper values
	int wborder = w + 2 * colorBorder;
	int hborder = h + 2 * colorBorder;

	// Calculate grid size
	dim3 block2d = dim3(16, 16, 1);
	dim3 grid2d = dim3((w + block2d.x - 1) / block2d.x, (h + block2d.y - 1) / block2d.y, 1);

	dim3 grid2dborder = dim3((wborder + block2d.x - 1) / block2d.x, (hborder + block2d.y - 1) / block2d.y, 1);

	for (int i = 0; i < numImgs - 1; i++) {
		// Generate a color coding for the flow field
		createColorCoding<<<grid2dborder, block2d>>>(data.d_v1[i], data.d_v2[i], data.d_flow[i], wborder, hborder, colorBorder);
		//createColorCoding<<<grid2dborder, block2d>>>(data.d_u1, data.d_v1, data.d_v2, data.d_flow, wborder, hborder, nc, colorBorder);
		cudaDeviceSynchronize();
		CUDA_CHECK;

		// Copy results to Host
		cudaMemcpy(v1[i], data.d_v1[i], w * h * sizeof(float), cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		cudaMemcpy(v2[i], data.d_v2[i], w * h * sizeof(float), cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		cudaMemcpy(flow[i], data.d_flow[i], wborder * hborder * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		CUDA_CHECK;
	}
	for (int i = 0; i < numImgs; i++) {
		cudaMemcpy(sr[i], data.d_u[i], w * h * nc * sizeof(float), cudaMemcpyDeviceToHost);
		CUDA_CHECK;
	}
}


int main(int argc, char **argv) {
	// Before the GPU can process your kernels, a so called "CUDA context" must be initialized
	// This happens on the very first call to a CUDA function, and takes some time (around half a second)
	// We will do it right here, so that the run time measurements are accurate
	cudaDeviceSynchronize();  CUDA_CHECK;


	// Reading command line parameters:
	// getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
	// If "-param" is not specified, the value of "var" remains unchanged
	//
	// return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise


	// Number of computation repetitions to get a better run time measurement
	int repeats = 1;
	getParam("repeats", repeats, argc, argv);
	cout << "repeats: " << repeats << endl;

	// Load the input image as grayscale if "-gray" is specifed
	bool gray = false;
	getParam("gray", gray, argc, argv);
	cout << "gray: " << gray << endl;

	// Value for tuning similarity of the downsampled high resolution results to the low resolution input images
	float alpha = 1.f;
	getParam("alpha", alpha, argc, argv);
	cout << "alpha: " << alpha << endl;

	// Value for tuning the total variation of the high resolution images
	float beta = 0.01f;
	getParam("beta", beta, argc, argv);
	cout << "beta: " << beta << endl;

	// Value for tuning the importance of the flow constraint
	float gamma = 8.f;
	getParam("gamma", gamma, argc, argv);
	cout << "gamma: " << gamma << endl;

	// Thickness of the colored border in the output image of the color coded flow field
	int colorBorder = 4;
	getParam("border", colorBorder, argc, argv);
	cout << "color coding border: " << colorBorder << endl;

	// Number of iterations for each update of the flow field and the super resolution images
	int iterations = 200;
	getParam("iterations", iterations, argc, argv);
	cout << "iterations: " << iterations << endl;

	// Path to the low resolution input images
	string imgPath = stdImgPath;
	getParam("path", imgPath, argc, argv);
	cout << "Path to the images: " << imgPath << endl;

	// Common part of the name of the low resolution input images
	string imgName = stdImgName;
	getParam("name", imgName, argc, argv);
	cout << "Name of the images: " << imgName << endl;

	// Type of the input images (e.g. "png")
	string imgType = stdImgType;
	getParam("type", imgType, argc, argv);
	cout << "Type of the images: " << imgType << endl;
	imgType = "." + imgType;

	// Number of input images to load
	int numImgs = stdNumImgs;
	getParam("count", numImgs, argc, argv);
	cout << "How many images to load: " << numImgs << endl;

	// Number of digits that should be respected in loading the input images
	int numDigits = stdNumDigits;
	getParam("digits", numDigits, argc, argv);
	cout << "Number of digits in the name: " << numDigits << endl;

	// Index of the first input image to load
	int startImg = stdStartImg;
	getParam("start", startImg, argc, argv);
	cout << "Index of the start Image: " << startImg << endl;

	// Init camera / Load input image

#ifdef CAMERA
	// Init camera
	cv::VideoCapture camera(0);
	if (!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
	int camW = 640;
	int camH = 480;
	camera.set(CV_CAP_PROP_FRAME_WIDTH, camW);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, camH);
	// Read in first frame to get the dimensions
	numImgs = 2;
	cv::Mat* mIn = new cv::Mat[numImgs];
	camera >> mIn[0];
	camera >> mIn[1];
	// Convert to float representation (opencv loads image values as single bytes by default)
	mIn[0].convertTo(mIn[0], CV_32F);
	mIn[1].convertTo(mIn[1], CV_32F);
	// Convert range of each channel from [0, 255] to [0, 1]
	mIn[0] /= 255.f;
	mIn[1] /= 255.f;

#else
	// Load all of the images needed
	cv::Mat* mIn = new cv::Mat[numImgs];
	for (int i = 0; i < numImgs; i++){
		int imageIdx = startImg + i;
		// Generating the complete image path
		stringstream ss;
		ss << setw(numDigits) << setfill('0') << imageIdx;
		string image = imgPath + imgName + ss.str() + imgType;
		// Loading the image
		mIn[i] = cv::imread(image.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
		if (mIn[i].data == NULL) {
			cerr << "ERROR: Could not load image " << image << endl;
			system("pause");
			return 1;
		}
		// Convert to float representation (opencv loads image values as single bytes by default)
		mIn[i].convertTo(mIn[i], CV_32F);
		// Convert range of each channel from [0, 255] to [0, 1]
		mIn[i] /= 255.f;
	}
#endif

	// Get the dimensions of the images
	int w_small = mIn[0].cols;
	int h_small = mIn[0].rows;
	int w = 2 * w_small;
	int h = 2 * h_small;
	int nc = mIn[0].channels();
	cout << "input images: " << w_small << " x " << h_small << endl;

	
	// Set the output image format
	cv::Mat mSR(h, w, mIn[0].type());
	cv::Mat mFlow((h + 2 * colorBorder), (w + 2 * colorBorder), CV_32FC3);	
	cv::Mat mV1(h, w, CV_32FC1);
	cv::Mat mV2(h, w, CV_32FC1);

	//cv::Mat mOut(h, w, mIn.type());  // mOut will have the same number of channels as the input image, nc layers
	//cv::Mat mOut(h, w, CV_32FC3);    // mOut will be a color image, 3 layers
	//cv::Mat mOut(h, w, CV_32FC1);    // mOut will be a grayscale image, 1 layer

	// Allocate memory for an arbitary amount of input images
	float** imgIn = new float*[numImgs];
	float** imgFlow = new float*[numImgs - 1];
	float** imgV1 = new float*[numImgs - 1];
	float** imgV2 = new float*[numImgs - 1];
	float** imgSR = new float*[numImgs];

	for (int i = 0; i < numImgs; i++){
		imgIn[i] = new float[(size_t)(w_small*h_small*nc)];
		imgSR[i] = new float[(size_t)w*h*mSR.channels()];
	}

	for (int i = 0; i < numImgs - 1; i++) {
		// Allocate memory for output images
		imgFlow[i] = new float[(size_t)(w + 2 * colorBorder)*(h + 2 * colorBorder)*mFlow.channels()];
		imgV1[i] = new float[(size_t)w*h*mV1.channels()];
		imgV2[i] = new float[(size_t)w*h*mV2.channels()];
	}
	


#ifdef CAMERA

	Data data;
	// Allocate memory for gpu arrays
	allocateGPUMemory(data, w, h, 2, w_small, h_small, nc, colorBorder);

	// Read a camera image frame every 30 milliseconds:
	// cv::waitKey(30) waits 30 milliseconds for a keyboard input,
	// returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
	while (cv::waitKey(30) < 0) {
		float* temp = imgIn[0];
		imgIn[0] = imgIn[1];
		imgIn[1] = temp;
		// Get camera image
		camera >> mIn[1];
		// convert to float representation (opencv loads image values as single bytes by default)
		mIn[1].convertTo(mIn[1], CV_32F);
		// Convert range of each channel from [0, 255] to [0, 1]
		mIn[1] /= 255.f;

		// Init raw input image array
		// opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
		// But for CUDA it's better to work with layered images: rrr... ggg... bbb...
		// So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
		convert_mat_to_layered(imgIn[1], mIn[1]);
#else
		// Convert all images
		for (int i = 0; i < numImgs; i++) {
			convert_mat_to_layered(imgIn[i], mIn[i]);
		}

		// Allocate memory for gpu arrays
		Data data;
		allocateGPUMemory(data, numImgs, w, h, w_small, h_small, nc, colorBorder);
#endif

		// Fetch time before calculation
		Timer timer;
		timer.start();

		// # Call the CUDA computation
		// Initialize arrays with start values
		InitializeGPUData(imgIn, data, numImgs, w, h, w_small, h_small, nc);
		// Alternating optimization of flow field and super resolution images
		for (int i = 0; i < 1; i++) {
			// Compute flow estimation
			calculateFlow(data, numImgs, gamma, iterations, w, h, nc);
			// Compute super resolution
			calculateSuperResolution(data, numImgs, iterations, alpha, beta, gamma, w, h, w_small, h_small, nc);
		}
		// Get results from computation
		getComputationResult(data, numImgs, imgV1, imgV2, imgFlow, imgSR, w, h, nc, colorBorder);
		
		// Get time after calculation and compute duration
		timer.end();
		float t = timer.get();  // elapsed time in seconds
		cout << "time: " << t * 1000 << " ms" << endl;
		
		// show input image
#ifdef CAMERA
		convert_layered_to_mat(mIn[0], imgIn[0]);
#endif
		// Show input images
		for (int i = 0; i < numImgs; i++) {
			string name = "Input " + to_string(i);
			showImage(name, mIn[i], 100, 100);
			name = "Super Resolution " + to_string(i);
			convert_layered_to_mat(mSR, imgSR[i]);
			showImage(name, mSR, 100 + w_small + 40, 100);
		}
		
		// Show all output images
		for (int i = 0; i < numImgs - 1; i++) {
			convert_layered_to_mat(mV1, imgV1[i]);
			showImage("V1", (mV1 + 1.0f) / 2.0f, 100 + w_small + w + 80, 100);
			convert_layered_to_mat(mV2, imgV2[i]);
			showImage("V2", (mV2 + 1.0f) / 2.0f, 100 + w_small + w + 80, 100);
			convert_layered_to_mat(mFlow, imgFlow[i]);
			showImage("Flow Field", mFlow, 100 + w_small + w + 80, 100);
		}
		

#ifdef CAMERA
		// end of camera loop
	}
	// Free arrays on gpu memory
	freeGPUMemory(data);
#else
	// Free arrays on gpu memory
	freeGPUMemory(data, numImgs);
	// wait for key inputs
	cv::waitKey(0);
#endif


#ifdef SAVE
	// save input and result
	cv::imwrite("image_input.png", mIn*255.f);  // "imwrite" assumes channel range [0,255]
	cv::imwrite("image_V1.png", (mV1 + 1.0f) / 2.0f * 255.f);
	cv::imwrite("image_V2.png", (mV2 + 1.0f) / 2.0f * 255.f);
#endif

	// free allocated arrays
	delete[] imgIn;
	delete[] imgFlow;
	
	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}



