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


#ifndef HELPER_H
#define HELPER_H

#include <cuda_runtime.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif

// parameter processing
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv)
{
	const char *c_param = param.c_str();
	for (int i = argc - 1; i >= 1; i--)
	{
		if (argv[i][0] != '-') continue;
		if (strcmp(argv[i] + 1, c_param) == 0)
		{
			if (!(i + 1<argc)) continue;
			std::stringstream ss;
			ss << argv[i + 1];
			ss >> var;
			return (bool)ss;
		}
	}
	return false;
}




// opencv helpers
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn);
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn);
void showImage(std::string title, const cv::Mat &mat, int x, int y);
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY);



// adding Gaussian noise
void addNoise(cv::Mat &m, float sigma);




// measuring time
class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
		cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};




// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);



#endif  // HELPER_H
