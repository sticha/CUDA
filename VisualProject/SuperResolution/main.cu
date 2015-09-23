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
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
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

//Input parameters
const string stdImgPath = "../../material/Images/";
const string stdImgName = "carwide_";
const string stdImgType = "png";
const int stdNumDigits = 2;
const int stdNumImgs = 2;
const int stdStartImg = 1;

// uncomment to use the camera
// #define CAMERA

struct Data {
	float* d_f1, *d_f2, *d_u1, *d_u2, *d_v1, *d_v2, *d_b, *d_p, *d_out, *d_energy;
	float2* d_A, *d_q1, *d_q2;
};

void allocateGPUMemory(Data& data, int w, int h, int w_small, int h_small, int nc, int colorBorder) {
	// helper values
	size_t n_small = w_small*h_small*nc;
	size_t n = w*h*nc;
	int wborder = w + 2 * colorBorder;
	int hborder = h + 2 * colorBorder;
	// # Allocate GPU memory
	cudaMalloc(&data.d_f1, n_small*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_f2, n_small*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_u1, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_u2, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_v1, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_v2, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_b, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_p, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_A, n*sizeof(float2));
	CUDA_CHECK;
	cudaMalloc(&data.d_q1, w*h*sizeof(float2));
	CUDA_CHECK;
	cudaMalloc(&data.d_q2, w*h*sizeof(float2));
	CUDA_CHECK;
	cudaMalloc(&data.d_out, wborder*hborder * 3 * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&data.d_energy, sizeof(float));
	CUDA_CHECK;
}
void InitializeGPUData(float* f1, float* f2, Data data, int w, int h, int w_small, int h_small, int nc) {
	// # Initalize arrays
	// Helper values
	size_t n_small = w_small*h_small*nc;
	size_t n = w*h*nc;
	// Fill arrays with 0
	cudaMemset(data.d_v1, 0, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMemset(data.d_v2, 0, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMemset(data.d_b, 0, n*sizeof(float));
	CUDA_CHECK;
	cudaMemset(data.d_p, 0, n*sizeof(float));
	CUDA_CHECK;
	cudaMemset(data.d_A, 0, n*sizeof(float2));
	CUDA_CHECK;
	cudaMemset(data.d_q1, 0, w*h*sizeof(float2));
	CUDA_CHECK;
	cudaMemset(data.d_q2, 0, w*h*sizeof(float2));
	CUDA_CHECK;
	// Copy images to GPU
	cudaMemcpy(data.d_f1, f1, n_small * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(data.d_f2, f2, n_small * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	// Calculate grid size
	dim3 block3d = dim3(16, 16, nc);
	dim3 grid3d = dim3((w + block3d.x - 1) / block3d.x, (h + block3d.y - 1) / block3d.y, 1);
	// Upsample f to u
	upsample<<<grid3d, block3d>>>(data.d_f1, data.d_u1, w_small, h_small);
	cudaDeviceSynchronize();
	CUDA_CHECK;
	upsample<<<grid3d, block3d>>>(data.d_f2, data.d_u2, w_small, h_small);
	cudaDeviceSynchronize();
	CUDA_CHECK;
	//cudaMemcpy(data.d_u1, f1, n * sizeof(float), cudaMemcpyHostToDevice);
	//CUDA_CHECK;
	//cudaMemcpy(data.d_u2, f2, n * sizeof(float), cudaMemcpyHostToDevice);
	//CUDA_CHECK;
}

void freeGPUMemory(Data& data) {
	// # Free GPU Memory
	cudaFree(data.d_f1);
	cudaFree(data.d_f2);
	cudaFree(data.d_u1);
	cudaFree(data.d_u2);
	cudaFree(data.d_v1);
	cudaFree(data.d_v2);
	cudaFree(data.d_b);
	cudaFree(data.d_p);
	cudaFree(data.d_A);
	cudaFree(data.d_q1);
	cudaFree(data.d_q2);
	cudaFree(data.d_out);
	cudaFree(data.d_energy);
	CUDA_CHECK;
}



// The difference d_b = d_u2 - d_u1
__global__ void imgDif(float * d_u1, float * d_u2, float *d_b, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	if (x >= w || y >= h)
		return;

	int idx = x + y * w + c * w * h;
	d_b[idx] = d_u2[idx] - d_u1[idx];
}

void calculateSuperResolution(Data data, int w, int h, int w_small, int h_small, int nc, float* out_u1, float* out_u2, int iterations, float alpha, float beta, float gamma, float* v1, float* v2) {
	float* d_u1, *d_u2, *d_v1, *d_v2, *d_p1, *d_p2, *d_r;
	float2* d_q1, *d_q2;
	// helper values
	int n = w*h*nc;
	int n_small = w_small*h_small*nc;

	cudaMalloc(&d_u1, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_u2, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_v1, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_v2, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_p1, n_small*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_p2, n_small*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_r, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_q1, n*sizeof(float2));
	CUDA_CHECK;
	cudaMalloc(&d_q2, n*sizeof(float2));
	CUDA_CHECK;
	
	// Copy images to GPU
	cudaMemcpy(d_v1, v1, w * h * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(d_v2, v2, w * h * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;

	// initalize temporary data
	cudaMemset(d_u1, 0, n*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_u2, 0, n*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_p1, 0, n_small*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_p2, 0, n_small*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_r, 0, n*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_q1, 0, n*sizeof(float2));
	CUDA_CHECK;
	cudaMemset(d_q2, 0, n*sizeof(float2));
	CUDA_CHECK;

	// Calculate grid size
	dim3 block3d = dim3(16, 16, nc);
	dim3 grid3d = dim3((w + block3d.x - 1) / block3d.x, (h + block3d.y - 1) / block3d.y, 1);

	dim3 block2d = dim3(16, 16, 1);
	dim3 grid2d = dim3((w + block2d.x - 1) / block2d.x, (h + block2d.y - 1) / block2d.y, 1);

	dim3 block1d = dim3(128, 1, 1);
	dim3 grid1d = dim3((w*h + block1d.x - 1) / block1d.x, 1, 1);
	int bytesSM1d = block1d.x * sizeof(float);

	float sigmaP = 1.0f;
	float sigmaQ = 0.5f;

	for (int i = 0; i < iterations; i++) {
		// Update p1, p2, q1, q2, r and u
		super_updateP<<<grid3d, block3d>>>(d_p1, data.d_f1, sigmaP, alpha, w_small, h_small);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		super_updateP<<<grid3d, block3d>>>(d_p2, data.d_f2, sigmaP, alpha, w_small, h_small);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		super_updateQ<<<grid3d, block3d>>>(d_q1, d_u1, sigmaQ, beta, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		super_updateQ<<<grid3d, block3d>>>(d_q2, d_u2, sigmaQ, beta, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		super_updateR<<<grid3d, block3d>>>(d_r, d_u1, d_u2, d_v1, d_v2, gamma, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		super_updateU<<<grid3d, block3d>>>(d_u1, d_u2, d_r, d_p1, d_p2, d_q1, d_q2, d_v1, d_v2, gamma, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
	}

	// Copy result to Host
	cudaMemcpy(out_u1, d_u1, n * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(out_u2, d_u2, n * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	// Free GPU Memory
	cudaFree(d_u1);
	cudaFree(d_u2);
	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_p1);
	cudaFree(d_p2);
	cudaFree(d_r);
	cudaFree(d_q1);
	cudaFree(d_q2);
	CUDA_CHECK;
}

/**
 * u1, u2, are input images with size w*h*nc
 * v1, v2, are vector components with size w*h describing the flow
 */
void calculateFlow(Data data, float gamma, int iterations, int w, int h, int nc) {
	// Calculate grid size
	dim3 block3d = dim3(16, 16, nc);
	dim3 grid3d = dim3((w + block3d.x - 1) / block3d.x, (h + block3d.y - 1) / block3d.y, 1);
	
	dim3 block2d = dim3(16, 16, 1);
	dim3 grid2d = dim3((w + block2d.x - 1) / block2d.x, (h + block2d.y - 1) / block2d.y, 1);

	dim3 block1d = dim3(128, 1, 1);
	dim3 grid1d = dim3((w*h + block1d.x - 1) / block1d.x, 1, 1);
	int bytesSM1d = block1d.x * sizeof(float);

	// Calculate b 
	imgDif<<<grid3d, block3d>>>(data.d_u1, data.d_u2, data.d_b, w, h);
	cudaDeviceSynchronize();
	CUDA_CHECK;
	// Calculate A
	calculateGradientCD<<<grid3d, block3d >> >(data.d_u2, data.d_A, w, h, nc);
	cudaDeviceSynchronize();
	CUDA_CHECK;
	float sigmaQ = 0.5f;

	for (int i = 0; i < iterations; i++) {
		// Update p, q1, q2 and v
		updateP<<<grid3d, block3d>>>(data.d_p, data.d_v1, data.d_v2, data.d_A, data.d_b, gamma, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		updateQ<<<grid2d, block2d>>>(data.d_q1, data.d_v1, sigmaQ, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		updateQ<<<grid2d, block2d>>>(data.d_q2, data.d_v2, sigmaQ, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		updateV<<<grid2d, block2d>>>(data.d_v1, data.d_v2, data.d_p, data.d_q1, data.d_q2, data.d_A, w, h, nc);
		cudaDeviceSynchronize();
		CUDA_CHECK;

		cudaMemset(data.d_energy, 0, sizeof(float));
		CUDA_CHECK;
		flowFieldEnergy<<<grid1d, block1d, bytesSM1d>>>(data.d_energy, data.d_A, data.d_b, data.d_v1, data.d_v2, gamma, w, h, nc);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		float energy;
		cudaMemcpy(&energy, data.d_energy, sizeof(float), cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		cout << "Flow field energy in iteration " << i << ": " << energy << endl;
	}
}

void getComputationResult(Data& data, float* v1, float* v2, float* flow, float* sr1, float* sr2, int w, int h, int nc, int colorBorder) {
	// helper values
	int wborder = w + 2 * colorBorder;
	int hborder = h + 2 * colorBorder;

	// Calculate grid size
	dim3 block2d = dim3(16, 16, 1);
	dim3 grid2d = dim3((w + block2d.x - 1) / block2d.x, (h + block2d.y - 1) / block2d.y, 1);

	dim3 grid2dborder = dim3((wborder + block2d.x - 1) / block2d.x, (hborder + block2d.y - 1) / block2d.y, 1);

	createColorCoding<<<grid2dborder, block2d>>>(data.d_v1, data.d_v2, data.d_out, wborder, hborder, colorBorder);
	//createColorCoding<<<grid2dborder, block2d>>>(d_u1, d_v1, d_v2, d_out, wborder, hborder, nc, colorBorder);
	cudaDeviceSynchronize();
	CUDA_CHECK;

	// Copy result to Host
	cudaMemcpy(v1, data.d_v1, w * h * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(v2, data.d_v2, w * h * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(flow, data.d_out, wborder * hborder * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(sr1, data.d_u1, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(sr2, data.d_u2, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
}


int main(int argc, char **argv)
{
	// Before the GPU can process your kernels, a so called "CUDA context" must be initialized
	// This happens on the very first call to a CUDA function, and takes some time (around half a second)
	// We will do it right here, so that the run time measurements are accurate
	cudaDeviceSynchronize();  CUDA_CHECK;




	// Reading command line parameters:
	// getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
	// If "-param" is not specified, the value of "var" remains unchanged
	//
	// return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

	// number of computation repetitions to get a better run time measurement
	int repeats = 1;
	getParam("repeats", repeats, argc, argv);
	cout << "repeats: " << repeats << endl;

	// load the input image as grayscale if "-gray" is specifedimgEnding
	bool gray = false;
	getParam("gray", gray, argc, argv);
	cout << "gray: " << gray << endl;

	// ### Define your own parameters here as needed    
	float gamma = 10.f;
	getParam("gamma", gamma, argc, argv);
	cout << "gamma: " << gamma << endl;

	int colorBorder = 4;
	getParam("border", colorBorder, argc, argv);
	cout << "color coding border: " << colorBorder << endl;

	float iterations = 200;
	getParam("iterations", iterations, argc, argv);
	cout << "iterations: " << iterations << endl;

	string imgPath = stdImgPath;
	getParam("path", imgPath, argc, argv);
	cout << "Path to the images: " << imgPath << endl;

	string imgName = stdImgName;
	getParam("name", imgName, argc, argv);
	cout << "Name of the images: " << imgName << endl;

	string imgType = stdImgType;
	getParam("type", imgType, argc, argv);
	cout << "Type of the images: " << imgType << endl;
	imgType = "." + imgType;

	int numImgs = stdNumImgs;
	getParam("count", numImgs, argc, argv);
	cout << "How many images to load: " << numImgs << endl;

	int numDigits = stdNumDigits;
	getParam("digits", numDigits, argc, argv);
	cout << "Number of digits in the name: " << numDigits << endl;

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
	// read in first frame to get the dimensions
	cv::Mat mIn1, mIn2;
	camera >> mIn1;
	camera >> mIn2;
	// convert to float representation (opencv loads image values as single bytes by default)
	mIn1.convertTo(mIn1, CV_32F);
	mIn2.convertTo(mIn2, CV_32F);
	// convert range of each channel to [0,1] (opencv default is [0,255])
	mIn1 /= 255.f;
	mIn2 /= 255.f;
	// get image dimensions
	int w_small = mIn1.cols;         // width of input image
	int h_small = mIn1.rows;         // height of input image
	int w = 2 * w_small;
	int h = 2 * h_small;
	int nc = mIn1.channels();  // number of channels
	

#else
	// Load all of the images needed
	cv::Mat * mIn = new cv::Mat[numImgs];
	for (int i = 0; i < numImgs; i++){
		int imageIdx = startImg + i;
		// Generating the complete img Path
		stringstream ss;
		ss << setw(numDigits) << setfill('0') << imageIdx;
		string image = imgPath + imgName + ss.str() + imgType;
		// Loading the img
		mIn[i] = cv::imread(image.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
		if (mIn[i].data == NULL) { cerr << "ERROR: Could not load image " << image << endl; system("pause"); return 1; }
		// convert to float representation (opencv loads image values as single bytes by default)
		mIn[i].convertTo(mIn[i], CV_32F);
		// convert range of each channel to [0,1] (opencv default is [0,255])
		mIn[i] /= 255.f;
	}
	// Its's assumed width and height is the same for all Images
	int w_small = mIn[0].cols;
	int h_small = mIn[0].rows;
	int w = 2 * w_small;
	int h = 2 * h_small;
	int nc = mIn[0].channels();
#endif
	cout << "image: " << w_small << " x " << h_small << endl;

	// Set the output image format
#ifdef CAMERA
	//cv::Mat mOut(h, w, mIn.type());  // mOut will have the same number of channels as the input image, nc layers
	cv::Mat mFlow((h + 2 * colorBorder), (w + 2 * colorBorder), CV_32FC3);    // mOut will be a color image, 3 layers
	//cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
	// ### Define your own output images here as needed
	cv::Mat mSR1(h, w, mIn.type());
	cv::Mat mSR2(h, w, mIn.type());
#else
	//cv::Mat mOut(h, w, mIn[0].type());  // mOut will have the same number of channels as the input image, nc layers
	cv::Mat mFlow((h+2*colorBorder),(w+2*colorBorder),CV_32FC3);    // mOut will be a color image, 3 layers
	//cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
	cv::Mat mSR1(h, w, mIn[0].type());
	cv::Mat mSR2(h, w, mIn[0].type());
#endif
	cv::Mat mV1(h, w, CV_32FC1);
	cv::Mat mV2(h, w, CV_32FC1);


	// Allocate arrays
	// input/output image width: w
	// input/output image height: h
	// input image number of channels: nc
	// output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)
#ifdef CAMERA
	// allocate raw input image array
	float *imgIn1 = new float[(size_t)w_small*h_small*nc];
	float *imgIn2 = new float[(size_t)w_small*h_small*nc];
#else
	// Allocate space for an arbitary amount of images
	float **imgIn = new float*[numImgs];
	for (int i = 0; i < numImgs; i++){
		imgIn[i] = new float[(size_t)w_small*h_small*nc];
	}
#endif
	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgFlow = new float[(size_t)(w+2*colorBorder)*(h+2*colorBorder)*mFlow.channels()];

	float *imgV1 = new float[(size_t)w*h*mV1.channels()];
	float *imgV2 = new float[(size_t)w*h*mV2.channels()];
	float *imgSR1 = new float[(size_t)w*h*mSR1.channels()];
	float *imgSR2 = new float[(size_t)w*h*mSR2.channels()];



	// For camera mode: Make a loop to read in camera frames
#ifdef CAMERA

	Data data;
	// Allocate memory for gpu arrays
	allocateGPUMemory(data, w, h, w_small, h_small, nc, colorBorder);

	// Read a camera image frame every 30 milliseconds:
	// cv::waitKey(30) waits 30 milliseconds for a keyboard input,
	// returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
	while (cv::waitKey(30) < 0)
	{
		float * temp = imgIn1;
		imgIn1 = imgIn2;
		imgIn2 = temp;
		// Get camera image
		camera >> mIn2;
		// convert to float representation (opencv loads image values as single bytes by default)
		mIn2.convertTo(mIn2, CV_32F);
		// convert range of each channel to [0,1] (opencv default is [0,255])
		mIn2 /= 255.f;


		// Init raw input image array
		// opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
		// But for CUDA it's better to work with layered images: rrr... ggg... bbb...
		// So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
		convert_mat_to_layered(imgIn2, mIn2);

		Timer timer; timer.start();
		// Initialize arrays with start values
		InitializeGPUData(imgIn1, imgIn2, data, w, h, w_small, h_small, nc);

#else
		// Convert all images
		for (int i = 0; i < numImgs; i++) {
			convert_mat_to_layered(imgIn[i], mIn[i]);
		}

		Timer timer; timer.start();

		// # Call the CUDA computation
		Data data;
		// Allocate memory for gpu arrays
		allocateGPUMemory(data, w, h, w_small, h_small, nc, colorBorder);
		// Initialize arrays with start values
		InitializeGPUData(imgIn[0], imgIn[1], data, w, h, w_small, h_small, nc);
		
#endif


		// Compute flow estimation
		calculateFlow(data, gamma, iterations, w, h, nc);

		// TODO: call calculation of super resolution
		

		// Get results from computation
		getComputationResult(data, imgV1, imgV2, imgFlow, imgSR1, imgSR2, w, h, nc, colorBorder);

		timer.end();  float t = timer.get();  // elapsed time in seconds
		cout << "time: " << t * 1000 << " ms" << endl;


		// show input image
#ifdef CAMERA
		convert_layered_to_mat(mIn1,imgIn1);
		showImage("In1", mIn1, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
		showImage("In2", mIn2, 100, 100 + h + 40);
#else
		showImage("Input1", mIn[0], 100, 100);
		showImage("Input2", mIn[1], 100, 100 + h_small + 50);
#endif
		
		// ### Display your own output images here as needed
		convert_layered_to_mat(mSR1, imgSR1);
		showImage("Super Resolution 1", mSR1, 100 + w_small + 40, 100);
		convert_layered_to_mat(mSR2, imgSR2);
		showImage("Super Resolution 2", mSR2, 100 + w_small + 40, 100);
		convert_layered_to_mat(mV1, imgV1);
		showImage("V1", (mV1 + 1.0f) / 2.0f, 100 + w_small + w + 80, 100);
		convert_layered_to_mat(mV2, imgV2);
		showImage("V2", (mV2 + 1.0f) / 2.0f, 100 + w_small + w + 80, 100);
		convert_layered_to_mat(mFlow, imgFlow);
		showImage("Flow Field", mFlow, 100 + w_small + w + 80, 100);

#ifdef CAMERA
		// end of camera loop
	}
	// Free arrays on gpu memory
	freeGPUMemory(data);
#else
	// Free arrays on gpu memory
	freeGPUMemory(data);
	// wait for key inputs
	cv::waitKey(0);
#endif



#ifdef SAVE
	// save input and result
	cv::imwrite("image_input.png", mIn*255.f);  // "imwrite" assumes channel range [0,255]
	cv::imwrite("image_V1.png", (mV1 + 1.0f) / 2.0f * 255.f);
	cv::imwrite("image_V2.png", (mV2 + 1.0f) / 2.0f * 255.f);
#endif

#ifdef CAMERA
	delete[] imgIn1;
	delete[] imgIn2;
#else
	// free allocated arrays
	delete[] imgIn;
#endif
	delete[] imgFlow;
	
	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}



