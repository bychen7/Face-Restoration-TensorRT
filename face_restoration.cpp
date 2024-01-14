#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "face_restoration.hpp"


using namespace nvinfer1;
namespace py = pybind11;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


FaceRestoration::FaceRestoration() {}

FaceRestoration::FaceRestoration(const std::string engine_file_path) {
    char *trtModelStream = nullptr;
    size_t size = 0;

    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    } else {
        std::cerr << "could not open engine!" << std::endl;
        return;
    }

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
}


FaceRestoration::~FaceRestoration() {
    std::cout << "Calling des" << std::endl;
    delete context;
    delete engine;
    delete runtime;

    delete input;
    delete output;
}


void FaceRestoration::imagePreProcess(cv::Mat& img, cv::Mat& img_resized) {
    cv::cvtColor(img, img_resized, cv::COLOR_BGR2RGB);
    cv::Size dsize = cv::Size(INPUT_W, INPUT_H);
    cv::resize(img_resized, img_resized, dsize);
}


void FaceRestoration::imagePostProcess(float* output, cv::Mat& img) {
    img.create(cv::Size(INPUT_H, INPUT_W), CV_8UC3);
    for (int i = 0; i < OUTPUT_SIZE; i ++) {
        int w = i % INPUT_H;
        int h = (i / INPUT_W) % INPUT_H;
        int c = i / INPUT_H / INPUT_W;

        float pixel = output[i] * 0.5 + 0.5;
        if (pixel < 0) pixel = 0;
        if (pixel > 1) pixel = 1;
        pixel *= 255;

        img.at<cv::Vec3b>(h, w)[c] = pixel;
    }
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
}


void FaceRestoration::blobFromImages(std::vector<cv::Mat>& imgs, float* input) {
    int channels = 3;
    for (int b = 0; b < BATCH_SIZE; b++) {
	    cv::Mat img = imgs[b];
	    for (int c = 0; c < channels; c++) {
	        for (int h = 0; h < INPUT_H; h++) {
	            for (int w = 0; w < INPUT_W; w++) {
	                   float val = ((float)img.at<cv::Vec3b>(h, w)[c] / 255.0 - 0.5) / 0.5;
	                   try {
			     input[b * channels * INPUT_W * INPUT_H + c * INPUT_W * INPUT_H + h * INPUT_W + w] = val;
			   }
			   catch (const std::exception& e) {
			     std::cout << "Exception: " << e.what() << std::endl;
			   }
	            }
	        }
	    }
    }
}


void FaceRestoration::doInference(IExecutionContext& context, float* input, float* output) {
    void* buffers[2];
    CHECK(cudaMalloc(&buffers[inputIndex], INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

py::array_t<uint8_t> FaceRestoration::infer(py::array_t<uint8_t>& imgs)
    {
    auto batch_size = imgs.shape(0);
    auto rows = imgs.shape(1);
    auto cols = imgs.shape(2);
    auto channels = imgs.shape(3);
    auto type = CV_8UC3;

	    
    std::vector<cv::Mat> cvimgs;
    for (size_t index = 0; index < batch_size; ++index) {
        py::array_t<uint8_t> img = imgs[index];
        cv::Mat cvimg(rows, cols, type, (unsigned char*)img.data());
        cv::Mat img_resized;
        imagePreProcess(cvimg, img_resized);
        cvimgs.push_back(img_resized);
    }

    blobFromImages(cvimgs, input);
    doInference(*context, input, output);
    cv::Mat res;
    imagePostProcess(output, res);

    py::array_t<uint8_t> output(
                                py::buffer_info(
                                res.data,
                                sizeof(uint8_t), //itemsize
                                py::format_descriptor<uint8_t>::format(),
                                3, // ndim
                                std::vector<size_t> {batch_size, rows, cols , 3}, // shape
                                std::vector<size_t> { sizeof(uint8_t) * batch_size * cols * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)}
    )
    );
    return output;
    };
