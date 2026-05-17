#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "tea_deploy/config.h"

namespace tea {

struct LetterboxMeta {
    float ratio = 1.f;
    int pad_w = 0;
    int pad_h = 0;
    int new_w = 0;
    int new_h = 0;
};

/// BGR uint8 → NCHW float32（CHW 连续），写入 out（长度 3*H*W）
bool PreprocessBgr(
    const cv::Mat& bgr,
    const PreprocessConfig& cfg,
    PreprocessMode mode,
    std::vector<float>* out,
    LetterboxMeta* letterbox = nullptr);

}  // namespace tea
