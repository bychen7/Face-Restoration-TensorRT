#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>


using namespace nvinfer1;
namespace py = pybind11;

class FaceRestoration {
    public:
        FaceRestoration();
        FaceRestoration(const std::string engine_file_path);
        void imagePreProcess(cv::Mat& img, cv::Mat& img_resized);
        void imagesPostProcess(float* output, std::vector<cv::Mat>& img);
        void blobFromImages(std::vector<cv::Mat>& img, float* blob);
        void doInference(IExecutionContext& context, float* input, float* output);
	py::array_t<uint8_t> infer(py::array_t<uint8_t>& img);
        ~FaceRestoration();

    private:
        static const int INPUT_H = 512;
        static const int INPUT_W = 512;
	    static const int BATCH_SIZE = 12;
        static const int CHANNELS = 3;
        static const int INPUT_SIZE = BATCH_SIZE * CHANNELS * INPUT_H * INPUT_W;
        static const int OUTPUT_SIZE = BATCH_SIZE * CHANNELS * INPUT_H * INPUT_W;

        const char *INPUT_BLOB_NAME = "input";
        const char *OUTPUT_BLOB_NAME = "output";

        int inputIndex;
        int outputIndex;

        IRuntime* runtime = nullptr;
        ICudaEngine* engine = nullptr;
        IExecutionContext* context = nullptr;

        float* input = new float[INPUT_SIZE];
        float* output = new float[OUTPUT_SIZE];
};
