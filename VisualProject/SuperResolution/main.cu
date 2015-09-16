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
#include "update.h"
#include "flow_color.h"
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;

//Input parameters
const string imgPath = "../../material/Images/";
const string imgName = "carwide_";
const string imgEnding = ".png";
const int numDigits = 2;
const int numImgs = 2;
const int startImg = 1;

// uncomment to use the camera
//#define CAMERA


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


/**
 * u1, u2, are input images with size w*h*nc
 * v1, v2, are vector components with size w*h describing the flow
 */
void calculateFlow(float* u1, float* u2, float* v1, float* v2, float* out, float gamma, int iterations, int w, int h, int nc, int colorBorder) {
	// Allocate GPU memory
	float* d_u1, *d_u2, *d_v1, *d_v2, *d_b, *d_p, *d_out;
	float2* d_A, *d_q1, *d_q2;
	size_t n = w*h*nc;
	int wborder = w + 2 * colorBorder;
	int hborder = h + 2 * colorBorder;
	cudaMalloc(&d_u1, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_u2, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_v1, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_v2, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_b, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_p, n*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_A, n*sizeof(float2));
	CUDA_CHECK;
	cudaMalloc(&d_q1, w*h*sizeof(float2));
	CUDA_CHECK;
	cudaMalloc(&d_q2, w*h*sizeof(float2));
	CUDA_CHECK;
	cudaMalloc(&d_out, wborder*hborder * 3 * sizeof(float));
	CUDA_CHECK;

	// Copy images to GPU
	cudaMemcpy(d_u1, u1, n * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpy(d_u2, u2, n * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;

	cudaMemset(d_v1, 0, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_v2, 0, w*h*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_b, 0, n*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_p, 0, n*sizeof(float));
	CUDA_CHECK;
	cudaMemset(d_A, 0, n*sizeof(float2));
	CUDA_CHECK;
	cudaMemset(d_q1, 0, w*h*sizeof(float2));
	CUDA_CHECK;
	cudaMemset(d_q2, 0, w*h*sizeof(float2));
	CUDA_CHECK;



	// Calculate grid size
	dim3 block3d = dim3(16, 16, nc);
	dim3 grid3d = dim3((w + block3d.x - 1) / block3d.x, (h + block3d.y - 1) / block3d.y, 1);
	
	dim3 block2d = dim3(16, 16, 1);
	dim3 grid2d = dim3((w + block2d.x - 1) / block2d.x, (h + block2d.y - 1) / block2d.y, 1);

	dim3 grid2dborder = dim3((wborder + block2d.x - 1) / block2d.x, (hborder + block2d.y - 1) / block2d.y, 1);

	// Calculate b 
	imgDif<<<grid3d, block3d>>>(d_u1, d_u2, d_b, w, h);
	cudaDeviceSynchronize();
	CUDA_CHECK;
	// Calculate A
	calculateGradient<<<grid3d, block3d>>>(d_u2, d_A, w, h, nc);
	cudaDeviceSynchronize();
	CUDA_CHECK;
	float sigmaQ = 0.5f;

	for (int i = 0; i < 100; i++) {
		// Update p, q1, q2 and v
		updateP<<<grid3d, block3d>>>(d_p, d_v1, d_v2, d_A, d_b, gamma, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		updateQ<<<grid2d, block2d>>>(d_q1, d_v1, sigmaQ, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		updateQ<<<grid2d, block2d>>>(d_q2, d_v2, sigmaQ, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
		updateV<<<grid2d, block2d>>>(d_v1, d_v2, d_p, d_q1, d_q2, d_A, w, h);
		cudaDeviceSynchronize();
		CUDA_CHECK;
	}

	createColorCoding<<<grid2dborder, block2d>>>(d_v1, d_v2, d_out, wborder, hborder, colorBorder);
	cudaDeviceSynchronize();
	CUDA_CHECK;

	// Copy result to Host
	cudaMemcpy(v1, d_v1, w * h * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(v2, d_v2, w * h * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(out, d_out, wborder * hborder * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	// Free GPU Memory
	cudaFree(d_u1);
	cudaFree(d_u2);
	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_b);
	cudaFree(d_p);
	cudaFree(d_A);
	cudaFree(d_q1);
	cudaFree(d_q2);
	cudaFree(d_out);
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

	// load the input image as grayscale if "-gray" is specifed
	bool gray = false;
	getParam("gray", gray, argc, argv);
	cout << "gray: " << gray << endl;

	// ### Define your own parameters here as needed    
	float gamma = 0.1f;
	getParam("gamma", gamma, argc, argv);
	cout << "gamma: " << gamma << endl;

	int colorBorder = 4;
	getParam("border", colorBorder, argc, argv);
	cout << "color coding border: " << colorBorder << endl;

	float iterations = 100;
	getParam("iterations", iterations, argc, argv);
	cout << "iterations: " << iterations << endl;

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
	cv::Mat mIn;
	camera >> mIn;

	// Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
	cv::Mat mIn = cv::imread(image.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
	// check
	if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }

	// convert to float representation (opencv loads image values as single bytes by default)
	mIn.convertTo(mIn, CV_32F);
	// convert range of each channel to [0,1] (opencv default is [0,255])
	mIn /= 255.f;
	// get image dimensions
	int w = mIn.cols;         // width
	int h = mIn.rows;         // height
	int nc = mIn.channels();  // number of channels
	cout << "image: " << w << " x " << h << endl;

#endif
	// Load all of the images needed
	cv::Mat * mIn = new cv::Mat[numImgs];
	for (int i = 0; i < numImgs; i++){
		int imageIdx = startImg + i;
		// Generating the complete img Path
		stringstream ss;
		ss << setw(numDigits) << setfill('0') << imageIdx;
		string image = imgPath + imgName + ss.str() + imgEnding;
		// Loading the img
		mIn[i] = cv::imread(image.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
		if (mIn[i].data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
		// convert to float representation (opencv loads image values as single bytes by default)
		mIn[i].convertTo(mIn[i], CV_32F);
		// convert range of each channel to [0,1] (opencv default is [0,255])
		mIn[i] /= 255.f;
	}
	// Its's assumed width and height is the same for all Images
	int w = mIn[0].cols;
	int h = mIn[0].rows;
	int nc = mIn[0].channels();


	// Set the output image format
#ifdef CAMERA
	cv::Mat mOut(h, w, mIn.type());  // mOut will have the same number of channels as the input image, nc layers
	//cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
	//cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
	// ### Define your own output images here as needed
#else
	//cv::Mat mOut(h, w, mIn[0].type());  // mOut will have the same number of channels as the input image, nc layers
	cv::Mat mOut((h+2*colorBorder),(w+2*colorBorder),CV_32FC3);    // mOut will be a color image, 3 layers
	//cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
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
	float *imgIn = new float[(size_t)w*h*nc];
#else
	// Allocate space for an arbitary amount of images
	float **imgIn = new float*[numImgs];
	for (int i = 0; i < numImgs; i++){
		imgIn[i] = new float[(size_t)w*h*nc];
	}
#endif
	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t)(w+2*colorBorder)*(h+2*colorBorder)*mOut.channels()];

	float *v1 = new float[(size_t)w*h*mV1.channels()];
	float *v2 = new float[(size_t)w*h*mV2.channels()];


	// For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
	// Read a camera image frame every 30 milliseconds:
	// cv::waitKey(30) waits 30 milliseconds for a keyboard input,
	// returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
	while (cv::waitKey(30) < 0)
	{
		// Get camera image
		camera >> mIn;
		// convert to float representation (opencv loads image values as single bytes by default)
		mIn.convertTo(mIn, CV_32F);
		// convert range of each channel to [0,1] (opencv default is [0,255])
		mIn /= 255.f;


		// Init raw input image array
		// opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
		// But for CUDA it's better to work with layered images: rrr... ggg... bbb...
		// So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
		convert_mat_to_layered(imgIn, mIn);
#endif
		// Convert all images
		for (int i = 0; i < numImgs; i++) {
			convert_mat_to_layered(imgIn[i], mIn[i]);
		}

		Timer timer; timer.start();
		// Call the CUDA computation
		calculateFlow(imgIn[0], imgIn[1], v1, v2, imgOut, gamma, iterations, w, h, nc, colorBorder);

		timer.end();  float t = timer.get();  // elapsed time in seconds
		cout << "time: " << t * 1000 << " ms" << endl;






		// show input image
#ifdef CAMERA
		showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
#else
		showImage("Input", mIn[0], 100, 100);
		showImage("Input", mIn[1], 100, 100);
#endif
		
		// show output image: first convert to interleaved opencv format from the layered raw array
		convert_layered_to_mat(mOut, imgOut);
		showImage("ColorCoded", mOut, 100 + w + 40, 100);

		// ### Display your own output images here as needed
		convert_layered_to_mat(mV1, v1);
		showImage("V1", mV1, 100 +  2 * w + 80, 100);
		convert_layered_to_mat(mV2, v2);
		showImage("V2", mV2, 100 + 3 * w + 120, 100);

#ifdef CAMERA
		// end of camera loop
	}
#else
		// wait for key inputs
		cv::waitKey(0);
#endif



#ifdef SAVE
		// save input and result
		cv::imwrite("image_input.png", mIn*255.f);  // "imwrite" assumes channel range [0,255]
		cv::imwrite("image_V1.png", v1*255.f);
		cv::imwrite("image_V2.png", v2*255.f);
#endif
		// free allocated arrays
		delete[] imgIn;
		delete[] imgOut;

		// close all opencv windows
		cvDestroyAllWindows();
		return 0;
}



