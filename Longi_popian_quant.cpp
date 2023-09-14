#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <io.h>
#include <sstream>
#include <filesystem>
#include <sys/stat.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line)
{
    if (code != cudaSuccess) {
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t) 
{
    switch (t) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:   return "error";
    case nvinfer1::ILogger::Severity::kWARNING: return "warning";
    case nvinfer1::ILogger::Severity::kINFO:    return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
    default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger 
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING) {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if (severity <= Severity::kERROR) {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

typedef std::function<void(int current, int count, const std::vector<std::string>& files, nvinfer1::Dims dims, float* ptensor)> Int8Process;

class MyInt8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    MyInt8EntropyCalibrator2(const std::vector<std::string>& imagefiles, nvinfer1::Dims dims, const Int8Process& preprocess)
	{
        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->allimgs_ = imagefiles;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = false;
        this->files_.resize(dims.d[0]);
	}
    MyInt8EntropyCalibrator2(const std::vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess)
    {
        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->entropyCalibratorData_ = entropyCalibratorData;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = true;
        this->files_.resize(dims.d[0]);
    }
    virtual ~MyInt8EntropyCalibrator2()
	{
        if (this->tensor_host_ != nullptr) {
            //checkRuntime(cudaFreeHost(this->tensor_host_));
            free(this->tensor_host_);
            checkRuntime(cudaFree(this->tensor_device_));
            this->tensor_host_ = nullptr;
            this->tensor_device_ = nullptr;
        }
	}

	int32_t getBatchSize() const noexcept override
	{
        return this->dims_.d[0];
	}

	bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept override
	{
        if (!next()) return false;
        bindings[0] = this->tensor_device_;
        return true;
	}

    const std::vector<uint8_t>& getEntropyCalibratorData() noexcept
    {
        return this->entropyCalibratorData_;
    }

	void const* readCalibrationCache(std::size_t& length) noexcept override
	{
        if (this->fromCalibratorData_) {
            length = this->entropyCalibratorData_.size();
            return this->entropyCalibratorData_.data();
        }

        length = 0;
        return nullptr;
	}

	void writeCalibrationCache(void const* cache, std::size_t length) noexcept override
	{
        this->entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache + length);
	}

private:
    bool next()
    {
        int& batch_size = this->dims_.d[0];
        if (this->cursor_ + batch_size > this->allimgs_.size())
            return false;

        for (int i = 0; i < batch_size; ++i)
            this->files_[i] = this->allimgs_[this->cursor_++];

        if (this->tensor_host_ == nullptr) 
        {
            size_t volumn = 1;
            for (int i = 0; i < this->dims_.nbDims; ++i)
                volumn *= this->dims_.d[i];

            this->bytes_ = volumn * sizeof(float);
            this->tensor_host_ = (float*)malloc(this->bytes_);
            //checkRuntime(cudaMallocHost((void**)&this->tensor_host_, this->bytes_));
            checkRuntime(cudaMalloc((void**)&this->tensor_device_, this->bytes_));
        }

        this->preprocess_(this->cursor_, this->allimgs_.size(), this->files_, this->dims_, this->tensor_host_);
        checkRuntime(cudaMemcpy(this->tensor_device_, this->tensor_host_, this->bytes_, cudaMemcpyHostToDevice));
        return true;
    }

private:
    std::function<void(int current, int count, const std::vector<std::string>& files, nvinfer1::Dims dims, float* ptensor)> preprocess_;
    std::vector<std::string> allimgs_;
    int cursor_ = 0;
    size_t bytes_ = 0;
    nvinfer1::Dims dims_;
    std::vector<std::string> files_;
    float* tensor_host_ = nullptr;
    float* tensor_device_ = nullptr;
    std::vector<uint8_t> entropyCalibratorData_;
    bool fromCalibratorData_ = false;
};

//static bool exists(const std::string& path)
//{
//#ifdef _WIN32
//    return ::PathFileExistsA(path.c_str());
//#else
//    return access(path.c_str(), R_OK) == 0;
//#endif
//}
bool isFileExists_ifstream(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}
bool build_i8_model(const std::string& onnxPath, const std::vector<std::string>& vctImageFile, int maxBatchSize, size_t workspace, const std::string& enginePath, const std::string& calibFilePath)
{
    if (isFileExists_ifstream(enginePath)) {
        printf("Info: %s has exists.\n", enginePath.c_str());
        return true;
    }

    TRTLogger logger;

    // 这是基本需要的组件
    nvinfer1::IBuilder* pBuilder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* pConfig = pBuilder->createBuilderConfig();

    // createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    nvinfer1::INetworkDefinition* pNetwork = pBuilder->createNetworkV2(1);

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser* pIParser = nvonnxparser::createParser(*pNetwork, logger);
    if (!pIParser->parseFromFile(onnxPath.c_str(), 1)) {
        printf("ERROR: Failed to parse %s!\n", onnxPath.c_str());

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        pIParser->destroy();
        pNetwork->destroy();
        pConfig->destroy();
        pBuilder->destroy();
        return false;
    }

    printf("Workspace Size = %.2f MB\n", (size_t)(1 << 30) / 1024.0f / 1024.0f);
    //pConfig->setMaxWorkspaceSize(1 << 33);
    pConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

    // 如果模型有多个执行上下文，则必须多个profile
    // 多个输入共用一个profile
    // nvinfer1::IOptimizationProfile* pProfile = pIBuilder->createOptimizationProfile();
    nvinfer1::ITensor* input_tensor = pNetwork->getInput(0);
    nvinfer1::Dims input_dims = input_tensor->getDimensions();

    //input_dims.d[0] = 1;
    pConfig->setFlag(nvinfer1::BuilderFlag::kINT8);

    auto preprocess = [](int currentIndex, int totalCount, const std::vector<std::string>& batch_files, nvinfer1::Dims dims, float* ptensor) -> void
    {
        printf("Preprocess %d / %d\n", currentIndex, totalCount);

        // 标定所采用的数据预处理必须与推理时一样
        int& modelInputWidth = dims.d[3];
        int& modelInputHeight = dims.d[2];
        int image_area = modelInputWidth * modelInputHeight;
        for (int i = 0; i < batch_files.size(); ++i)
        {
            cv::Mat image = cv::imread(batch_files[i], 1);
            cv::resize(image, image, cv::Size(modelInputWidth, modelInputHeight));
            if (3 != image.channels())
            {
                printf("ERROR: image channel is not 3, but is %d.\n", image.channels());
            }
            image.convertTo(image, CV_32FC3);
            image = image / 255.0f;

            // HWC2CHW
            unsigned char* pimage = image.data;
            float* phost_r = ptensor;
            float* phost_g = ptensor + image_area * 1;
            float* phost_b = ptensor + image_area * 2;
            for (int i = 0; i < image_area; ++i, pimage += 3) {
                // 注意这里的顺序rgb调换了
                *phost_r++ = pimage[2];
                *phost_g++ = pimage[1];
                *phost_b++ = pimage[0];
            }
            ptensor += image_area * 3;
        }
    };

    // 配置int8标定数据读取工具
    MyInt8EntropyCalibrator2* calibrator = new MyInt8EntropyCalibrator2(vctImageFile, input_dims, preprocess);
    pConfig->setInt8Calibrator(calibrator);

    //// 配置最小允许batch
    //input_dims.d[0] = 1;
    //pProfile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    //pProfile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    //// 配置最大允许batch  if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
    //input_dims.d[0] = maxBatchSize;
    //pProfile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    //pConfig->addOptimizationProfile(pProfile);
    
    nvinfer1::ICudaEngine* pEngine = pBuilder->buildEngineWithConfig(*pNetwork, *pConfig);
    if (nullptr == pEngine)
    {
        printf("ERROR: Build engine failed!\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的
        delete calibrator;
        input_tensor;
        //pProfile;
        pIParser->destroy();
        pNetwork->destroy();
        pConfig->destroy();
        pBuilder->destroy();
        return false;
    }

    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory* pModelData = pEngine->serialize();
    //nvinfer1::IHostMemory* pModelData = pIBuilder->buildSerializedNetwork(*pNetwork, *pConfig);
    FILE* f = fopen(enginePath.c_str(), "wb");
    fwrite(pModelData->data(), 1, pModelData->size(), f);
    fclose(f);

    f = fopen(calibFilePath.c_str(), "wb");
    std::vector<uint8_t> calib_data = calibrator->getEntropyCalibratorData();
    fwrite(calib_data.data(), 1, calib_data.size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    delete calibrator;
    pModelData->destroy();
    pEngine->destroy();
    input_tensor;
    //pProfile;
    pIParser->destroy();
    pNetwork->destroy();
    pConfig->destroy();
    pBuilder->destroy();
    printf("Build Model Done.\n");

    return true;
}

std::vector<unsigned char> load_file(const std::string& file) 
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

//获取某个文件夹中特定格式(后缀)的文件名 [不要修改]
void GetAllFormatFiles(std::string path, std::vector<std::string>& files, std::string format)
{
    //文件句柄  
    long long hFile = 0;
    //文件信息  
    struct _finddata_t fileinfo;
    std::string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1)
    {
        do
        {
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    //files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
                    GetAllFormatFiles(p.assign(path).append("\\").append(fileinfo.name), files, format);
                }
            }
            else
            {
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);

        _findclose(hFile);
    }
}

int main()
{
    const std::string onnxPath = "../../../int8Demo/2023-8-19-popian_15.onnx";
    std::vector<std::string> vctImageFile;
    int maxBatchSize = 1;
    size_t workspace = 1 << 30;
    const std::string enginePath = "../../../int8Demo/2023-8-19-popian_i8_15.engine";
    const std::string calibFilePath = "../../../int8Demo/calib.data";

    GetAllFormatFiles("../../../int8Demo/popian_calib_pics", vctImageFile, ".jpg");
    build_i8_model(onnxPath, vctImageFile, maxBatchSize, workspace, enginePath, calibFilePath);

    return 0;
}
