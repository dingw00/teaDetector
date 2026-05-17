#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace tea {

/// 与 export_onnx 导出的 .meta.json / preprocessor_config 对齐
struct PreprocessConfig {
    int input_size = 640;
    bool do_rescale = true;    // ÷255
    bool do_normalize = true;  // (x-mean)/std
    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> std{0.229f, 0.224f, 0.225f};
};

struct ModelConfig {
    std::string onnx_path;
    int input_size = 640;
    int num_queries = 300;
    int num_classes = 2;
    bool use_focal_loss = true;
    PreprocessConfig preprocess{};
};

enum class PreprocessMode {
    /// 等比缩放 + 黑边（letterbox），与 eval_tea.preprocess_bgr 一致
    Letterbox,
    /// 直接拉伸到 input_size×input_size，与 HF RTDetrImageProcessor（do_pad=false）一致
    /// 当前 checkpoint 未重训时请用此模式
    Stretch,
};

enum class BoxMapMode {
    /// HF postprocess：target_sizes = 原图 (H,W)，适用于 Stretch 预处理
    OrigTargetSize,
    /// letterbox 反变换：先按网络输入尺寸还原，再减 pad / ratio
    LetterboxInverse,
};

struct Detection {
    int label = 0;
    float score = 0.f;
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
};

struct InferenceConfig {
    float conf_threshold = 0.2f;
    float nms_threshold = 0.3f;
    PreprocessMode preprocess_mode = PreprocessMode::Stretch;
    BoxMapMode box_map_mode = BoxMapMode::OrigTargetSize;
};

/// 从 export_onnx 生成的 .meta.json 读取关键字段（无第三方 JSON 依赖）
bool LoadMetaJson(const std::string& meta_path, ModelConfig* out, std::string* err = nullptr);

}  // namespace tea
