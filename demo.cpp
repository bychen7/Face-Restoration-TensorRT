#include <chrono>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>


#include "face_restoration.hpp"


#define DEVICE 0  // GPU id


int main(int argc, char **argv) {
    cudaSetDevice(DEVICE);

    if (argc != 4 || std::string(argv[2]) != "-i") {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "For Example:" << std::endl;
        std::cerr << "./demo ../models/model.engine -i ../images/test.png" << std::endl;
        return -1;
    }

    const std::string engine_file_path = argv[1];
    const std::string input_image_path = argv[3];

    cv::Mat img = cv::imread(input_image_path);
    FaceRestoration sample = FaceRestoration(engine_file_path);
    cv::Mat res;

    // warm up
    for (int i = 0; i < 10; i++) {
        sample.infer(img, res);
    }

    float times = 0.0;
    int count = 0;
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        sample.infer(img, res);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "count: " << count << std::endl;
        times += elapsed;
        count++;
    }
    std::cout << "times: " << times / count / 1000 << " ms" << std::endl;

    cv::imwrite("res.jpg", res);

    return 0;
}
