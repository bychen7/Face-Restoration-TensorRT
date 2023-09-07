#include <chrono>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_restoration.hpp"
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define DEVICE 0  // GPU id

namespace py = pybind11;

// Forward declaration for the FaceRestore class
class FaceRestore;

PYBIND11_MODULE(python_face_restore, m) {
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

// Define the FaceRestore class
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

    cv::Mat inference(cv::Mat img) {
        cv::Mat res;
        sample.infer(img, res);
        return res;
    }
};

// Other functions (call_go, FaceRestoration, etc.) should be defined outside the FaceRestore class.

// Example usage:
// int main() {
//     FaceRestore faceRestore;
//     faceRestore.setup();
//     cv::Mat inputImage = cv::imread("input_image.jpg");
//     cv::Mat outputImage = faceRestore.inference(inputImage);
//     cv::imwrite("output_image.jpg", outputImage);
//     return 0;
// }
