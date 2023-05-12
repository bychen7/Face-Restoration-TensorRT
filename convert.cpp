#include <fstream>
#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"


using namespace nvinfer1;


#define DEVICE 0  // GPU id


class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


int genEngine(const std::string onnx_file_path, const std::string engine_file_path) {
    IBuilder* builder = createInferBuilder(gLogger);

    uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(onnx_file_path.c_str(), int(ILogger::Severity::kWARNING));

    IBuilderConfig* config = builder->createBuilderConfig();
    // config->setMaxWorkspaceSize(1U << 31);
    config->setMaxWorkspaceSize(8 * (1L << 31));
    // config->setFlag(BuilderFlag::kFP16);

    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

    std::ofstream outEngine(engine_file_path.c_str(), std::ios::binary);
    if (!outEngine) {
        std::cerr << "could not open output file" << std::endl;
        return false;
    }
    outEngine.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

    delete parser;
    delete network;
    delete config;
    delete builder;
    delete serializedModel;

    return 0;
}


int main(int argc, char **argv) {
    cudaSetDevice(DEVICE);

    if (argc != 4 || std::string(argv[2]) != "-s") {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "For Example:" << std::endl;
        std::cerr << "./convert ../models/model.onnx -s ../models/model.engine" << std::endl;
        return -1;
    }

    const std::string onnx_file_path = argv[1];
    const std::string engine_file_path = argv[3];

    genEngine(onnx_file_path, engine_file_path);

    return 0;
}
