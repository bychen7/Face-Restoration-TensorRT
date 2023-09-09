#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "face_restoration.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FaceRestoration, m) {
    m.doc() = "Face Restoration Module";

    py::class_<FaceRestoration>(m, "FaceRestoration")
        .def(py::init<const std::string>())
        .def("infer", &FaceRestoration::infer);
}

