#include "trt_wrapper.h"

#include <NvOnnxParser.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>

#define KEY_ONNX "Onnx"
#define KEY_ENGINE "Engine"
#define KEY_WORKSPACE_SIZE "WorkspaceSize"
#define KEY_BATCH_SIZE "BatchSize"
#define KEY_PRECISION "Precision"

class TRTLogger: public nvinfer1::ILogger {
  public:
	void log(Severity severity, const char* msg) noexcept override {
		// Comment Block Msg
		if (severity == Severity::kINFO) {
			std::cout << msg << std::endl;
		} else if (severity == Severity::kWARNING) {
			std::cout << msg << std::endl;
		} else if (severity == Severity::kERROR) {
			std::cout << msg << std::endl;
		} else if (severity == Severity::kVERBOSE) {
			std::cout << msg << std::endl;
		}
	}

	nvinfer1::ILogger& getTRTLogger() {
		return *this;
	}
} g_logger;

TRTWrapper::TRTWrapper() {
}

TRTWrapper::~TRTWrapper() {
	releaseResource();
}

void TRTWrapper::setConfig(const char* cfg_file) {
	config_ = YAML::LoadFile(cfg_file);
}

void TRTWrapper::releaseResource() {
	for (auto iter = cuda_in_.begin(); iter != cuda_in_.end(); ++iter) {
		cudaFree(*iter);
	}

	cuda_in_.clear();

	for (auto iter = cuda_out_.begin(); iter != cuda_out_.end(); ++iter) {
		cudaFree(*iter);
	}

	cuda_out_.clear();

	if (context_) {
		context_ = NULL;
	}

	if (engine_) {
		engine_ = NULL;
	}
}

void TRTWrapper::allocResource() {
	try {
		if (engine_ == NULL) {
			throw 0;
		}

		int num_binding = engine_->getNbBindings();

		for (int i = 0; i < num_binding; ++i) {
			nvinfer1::Dims dim = engine_->getBindingDimensions(i);
			void* buff_ptr;

			nvinfer1::DataType dt = engine_->getBindingDataType(i);

			int tensor_size = sizeof(dt);
			for (int j = 0; j < dim.nbDims; ++j) {
				tensor_size *= dim.d[j];
			}

			cudaMalloc(&buff_ptr, tensor_size);

			if (engine_->bindingIsInput(i)) {
				cuda_in_.push_back(buff_ptr);
				cuda_buff_.push_back(buff_ptr);

				dim_in_.push_back(dim);
				size_in_.push_back(tensor_size);
			} else {
				cuda_out_.push_back(buff_ptr);
				cuda_buff_.push_back(buff_ptr);

				dim_out_.push_back(dim);
				size_out_.push_back(tensor_size);
			}
		}
	} catch (int code) {
		switch (code) {
		case 0:
			std::cout << "TRT Engine Error" << std::endl;

			break;
		}

		releaseResource();
	}
}

bool TRTWrapper::buildEngine() {
	releaseResource();

	try {
		std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_logger.getTRTLogger()));

		const auto explicit_batch = 1U << static_cast<unsigned int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

		std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(explicit_batch));
		std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, g_logger.getTRTLogger()));

		// Onnx Path
		std::string onnx_path = config_[KEY_ONNX].as<std::string>();

		// Parse Onnx and Verbose
		if (!parser->parseFromFile(onnx_path.c_str(), 3)) {
			throw 1;
		}

		// Batch Size
		batch_size_ = config_[KEY_BATCH_SIZE].as<int>();

		std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

		// Workspace Size
		// Default Workspace Size is 512Mb
		int workspace_size = config_[KEY_WORKSPACE_SIZE].as<int>();

		workspace_size = 2 << (20 + ((int)std::log2((double)workspace_size)));

		config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);

		// Precision
		unsigned int flags = 0x01 << (int)(nvinfer1::BuilderFlag::kGPU_FALLBACK);

		// Only Support FP16 Now
		std::string precision = config_[KEY_PRECISION].as<std::string>();

		if (precision == "FP16") {
			flags |= 0x01 << (int)(nvinfer1::BuilderFlag::kFP16);
		}

		config->setFlags(flags);

		// Build Engine
		std::unique_ptr<nvinfer1::IHostMemory> serialized_model(builder->buildSerializedNetwork(*net, *config));

		std::string engine_path;

		engine_path = config_[KEY_ENGINE].as<std::string>();

		int bin_size = serialized_model->size();

		std::ofstream ofs(engine_path.c_str(), std::ios::binary);

		ofs.write((char*)serialized_model->data(), bin_size);

		ofs.close();

		std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(g_logger.getTRTLogger()));

		engine_ = runtime->deserializeCudaEngine(serialized_model->data(), bin_size);
		context_ = engine_->createExecutionContext();
	} catch (int code) {
		switch (code) {
		case 0:
			std::cout << "Onnx file is not found" << std::endl;

			break;
		case 1:
			std::cout << "Onnx parsing error is occurred" << std::endl;

			break;
		case 2:
			std::cout << "Batch size is not found" << std::endl;

			break;
		}

		return false;
	}

	allocResource();

	return true;
}

bool TRTWrapper::loadEngine() {
	try {
		std::string engine_path = config_[KEY_ENGINE].as<std::string>();

		std::ifstream ifs(engine_path.c_str(), std::ios::binary);

		if (!ifs.good()) {
			throw 0;
		}

		ifs.seekg(0, ifs.end);

		int bin_length = ifs.tellg();

		ifs.seekg(0, ifs.beg);

		auto buff = std::make_unique<char[]>(bin_length);

		ifs.read(buff.get(), bin_length);

		ifs.close();

		std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(g_logger.getTRTLogger()));

		engine_ = runtime->deserializeCudaEngine(buff.get(), bin_length);
		context_ = engine_->createExecutionContext();

		batch_size_ = config_[KEY_BATCH_SIZE].as<int>();

	} catch (int code) {
		return false;
	}

	allocResource();

	return true;
}