#pragma once

#include <vector>

#include "tea_deploy/config.h"
#include "tea_deploy/preprocess.h"

namespace tea {

/// logits [num_queries*num_classes], pred_boxes [num_queries*4]（cxcywh 归一化）
std::vector<Detection> PostprocessDetections(
    const float* logits,
    const float* pred_boxes,
    int num_queries,
    int num_classes,
    bool use_focal_loss,
    int orig_w,
    int orig_h,
    int network_input_size,
    BoxMapMode box_map,
    const LetterboxMeta& letterbox,
    float conf_threshold,
    float nms_threshold);

}  // namespace tea
