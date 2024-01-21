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

    
    CHECK(cudaMalloc(&buffers[inputIndex], INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaMallocHost((void **)&input, INPUT_SIZE * sizeof(float)));
    CHECK(cudaMallocHost((void **)&output, OUTPUT_SIZE * sizeof(float)));
    CHECK(cudaStreamCreate(&stream));
}


FaceRestoration::~FaceRestoration() {
    std::cout << "Calling des" << std::endl;
    delete context;
    delete engine;
    delete runtime;

    // delete input;
    // delete output;
    CHECK(cudaFreeHost(input));
    CHECK(cudaFreeHost(output));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    cudaStreamDestroy(stream);
}


void FaceRestoration::imagePreProcess(cv::Mat& img, cv::Mat& img_resized) {
    cv::cvtColor(img, img_resized, cv::COLOR_BGR2RGB);
    cv::Size dsize = cv::Size(INPUT_W, INPUT_H);
    cv::resize(img_resized, img_resized, dsize);
}


void FaceRestoration::imagesPostProcess(float* output, std::vector<cv::Mat>& cvimgs) {
    const int step = INPUT_H * INPUT_W * CHANNELS;
    for (size_t index = 0; index < OUTPUT_SIZE; index += step) {
        // Create an OpenCV Mat using the array data
        cv::Mat img(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
        
        for (int i = 0; i < step; i++) {
            int w = i % INPUT_W;
            int h = (i / INPUT_W) % INPUT_H;
            int c = i / (INPUT_W * INPUT_H);

            float pixel = output[index + i] * 0.5 + 0.5;
            pixel = std::min(1.0f, std::max(0.0f, pixel));
            pixel *= 255;

            img.at<cv::Vec3b>(h, w)[c] = static_cast<uint8_t>(pixel);
        }

        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        cvimgs.push_back(img);
    }
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

    // buffers[0] = input;
    // buffers[1] = output;
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    

}

py::array_t<uint8_t> FaceRestoration::infer(py::array_t<uint8_t>& imgs)
    {
    auto batch_size = imgs.shape(0);
    auto rows = imgs.shape(1);
    auto cols = imgs.shape(2);
    auto channels = imgs.shape(3);
    auto type = CV_8UC3;

	    
    auto buf = imgs.request();
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);

    std::vector<cv::Mat> cvimgs;
    auto startTime = std::chrono::high_resolution_clock::now();
    for (py::ssize_t index = 0; index < batch_size; ++index) {
        // Create an OpenCV Mat using the array data
        cv::Mat image(rows, cols, CV_8UC3, ptr + index * rows * cols * channels);
        cv::Mat img_resized;
        imagePreProcess(image, img_resized);
        cvimgs.push_back(img_resized);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    std::cout << "preprocess " << totalTime << " ms" << '\n';

    startTime = std::chrono::high_resolution_clock::now();
    blobFromImages(cvimgs, input);
    endTime = std::chrono::high_resolution_clock::now();
    totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    std::cout << "blobbing " << totalTime << " ms" << '\n';

    startTime = std::chrono::high_resolution_clock::now();
    doInference(*context, input, output);
    endTime = std::chrono::high_resolution_clock::now();
    totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    std::cout << "inference " <<  totalTime << " ms" << '\n';

    std::vector<cv::Mat> res;
    startTime = std::chrono::high_resolution_clock::now();
    imagesPostProcess(output, res);
    endTime = std::chrono::high_resolution_clock::now();
    totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    std::cout << "post " << totalTime << " ms" << '\n' << '\n';

    // Loop through each image in the vector and save it
    // for (int i = 0; i < res.size(); ++i) {
    //     // Create a filename for the image (you can customize the filename as needed)
    //     std::string filename = "image_" + std::to_string(i) + ".jpg";

    //     // Save the image
    //     cv::imwrite(filename, res[i]);

    //     // Optionally, you can print the filename to confirm the saving process
    //     std::cout << "Saved: " << filename << std::endl;
    // }

    // Concatenate the images in the batch
    cv::Mat concatenated;
    cv::vconcat(res, concatenated);

    // Reshape the concatenated image to match the desired output shape
    cv::Mat reshaped(concatenated.rows, batch_size * cols, concatenated.type());
    reshaped.convertTo(reshaped, CV_8U);

    py::array_t<uint8_t> py_output(
        py::buffer_info(
            concatenated.data,
            sizeof(uint8_t), // itemsize
            py::format_descriptor<uint8_t>::format(),
            4, // ndim
            std::vector<size_t>{batch_size, rows, cols, 3}, // shape
            std::vector<size_t>{sizeof(uint8_t) * rows * cols * 3, sizeof(uint8_t) * cols * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)}
        )
    );
    return py_output;
}
