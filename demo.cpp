#include <chrono>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_restoration.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define DEVICE 0  // GPU id

namespace py = pybind11;


class FaceRestore {
public:
    FaceRestoration sample;  // Declare a FaceRestoration object

    FaceRestore() {
        // Constructor: You can initialize things here if needed
    }

    void setup() {
        // Set the CUDA device (assuming you have multiple GPUs)
        cudaSetDevice(DEVICE);

        // Initialize the FaceRestoration object with the model file path
        sample = FaceRestoration("./models/model.engine");
    }

    py::array_t<uint8_t> inference(py::array_t<uint8_t>& img)
    {
    auto rows = img.shape(0);
    auto cols = img.shape(1);
    auto channels = img.shape(2);
    std::cout << "rows: " << rows << " cols: " << cols << " channels: " << channels << std::endl;
    auto type = CV_8UC3;

    cv::Mat cvimg(rows, cols, type, (unsigned char*)img.data());

    cv::imwrite("test.png", cvimg); // OK

    cv::Mat res;
    sample.infer(img, res);

    cv::imwrite("test2.png", res); // OK

    py::array_t<uint8_t> output(
                                py::buffer_info(
                                res.data,
                                sizeof(uint8_t), //itemsize
                                py::format_descriptor<uint8_t>::format(),
                                3, // ndim
                                std::vector<size_t> {rows, cols , 3}, // shape
                                std::vector<size_t> { sizeof(uint8_t) * cols * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)}
    )
    );
    return output;
    }
};

PYBIND11_MODULE(FaceRestore, m) {
    m.doc() = R"pbdoc(
        Pybind11 face restoration
    )pbdoc";
    
    // Define the FaceRestore class and its methods
    py::class_<FaceRestore>(m, "FaceRestore")
        .def(py::init<>())
        .def("setup", &FaceRestore::setup)
        .def("inference", &FaceRestore::inference);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

// Example usage:
// int main() {
//     FaceRestore faceRestore;
//     faceRestore.setup();
//     cv::Mat inputImage = cv::imread("input_image.jpg");
//     cv::Mat outputImage = faceRestore.inference(inputImage);
//     cv::imwrite("output_image.jpg", outputImage);
//     return 0;
// }
