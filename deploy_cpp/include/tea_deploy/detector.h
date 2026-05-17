#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "tea_deploy/config.h"

namespace tea {

struct TeaDetectorImpl;

class TeaDetector {
public:
    TeaDetector();
    ~TeaDetector();

    TeaDetector(const TeaDetector&) = delete;
    TeaDetector& operator=(const TeaDetector&) = delete;

    bool Init(const ModelConfig& model_cfg, const InferenceConfig& infer_cfg, std::string* err = nullptr);

    /// 从 export_onnx 生成的 .meta.json 初始化（推荐）
    bool InitFromMeta(const std::string& meta_json_path, const InferenceConfig& infer_cfg, std::string* err = nullptr);

    std::vector<Detection> Detect(const cv::Mat& bgr) const;

    const ModelConfig& model_config() const;
    const InferenceConfig& inference_config() const;

private:
    std::unique_ptr<TeaDetectorImpl> impl_;
};

}  // namespace tea
