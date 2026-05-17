#include "tea_deploy/postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace tea {
namespace {

inline float Sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

void CenterToCorners(const float* cxcywh, float* xyxy) {
    const float cx = cxcywh[0];
    const float cy = cxcywh[1];
    const float w = cxcywh[2];
    const float h = cxcywh[3];
    xyxy[0] = cx - 0.5f * w;
    xyxy[1] = cy - 0.5f * h;
    xyxy[2] = cx + 0.5f * w;
    xyxy[3] = cy + 0.5f * h;
}

float IoU(const Detection& a, const Detection& b) {
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.f, xx2 - xx1);
    const float h = std::max(0.f, yy2 - yy1);
    const float inter = w * h;
    const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}

std::vector<Detection> NmsPerClass(std::vector<Detection> dets, float nms_thres) {
    if (dets.empty() || nms_thres >= 1.f) {
        std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
            return a.score > b.score;
        });
        return dets;
    }
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<Detection> out;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (dets[i].label != dets[j].label) continue;
            if (IoU(dets[i], dets[j]) > nms_thres) removed[j] = true;
        }
    }
    return out;
}

struct ScoredIndex {
    float score = 0.f;
    int flat_index = 0;
};

void MapBoxToOrig(
    float* xyxy_norm,
    int orig_w,
    int orig_h,
    int network_input_size,
    BoxMapMode box_map,
    const LetterboxMeta& lb) {
    float xyxy[4];
    CenterToCorners(xyxy_norm, xyxy);

    if (box_map == BoxMapMode::OrigTargetSize) {
        const float sx = static_cast<float>(orig_w);
        const float sy = static_cast<float>(orig_h);
        xyxy_norm[0] = xyxy[0] * sx;
        xyxy_norm[1] = xyxy[1] * sy;
        xyxy_norm[2] = xyxy[2] * sx;
        xyxy_norm[3] = xyxy[3] * sy;
        return;
    }

    const float s = static_cast<float>(network_input_size);
    float x1 = xyxy[0] * s;
    float y1 = xyxy[1] * s;
    float x2 = xyxy[2] * s;
    float y2 = xyxy[3] * s;
    const float inv_r = lb.ratio > 1e-6f ? (1.f / lb.ratio) : 1.f;
    x1 = (x1 - static_cast<float>(lb.pad_w)) * inv_r;
    y1 = (y1 - static_cast<float>(lb.pad_h)) * inv_r;
    x2 = (x2 - static_cast<float>(lb.pad_w)) * inv_r;
    y2 = (y2 - static_cast<float>(lb.pad_h)) * inv_r;
    xyxy_norm[0] = std::max(0.f, std::min(x1, static_cast<float>(orig_w - 1)));
    xyxy_norm[1] = std::max(0.f, std::min(y1, static_cast<float>(orig_h - 1)));
    xyxy_norm[2] = std::max(0.f, std::min(x2, static_cast<float>(orig_w)));
    xyxy_norm[3] = std::max(0.f, std::min(y2, static_cast<float>(orig_h)));
}

}  // namespace

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
    float nms_threshold) {
    const int num_top = num_queries;
    std::vector<Detection> candidates;
    candidates.reserve(static_cast<size_t>(num_top));

    if (use_focal_loss) {
        const int flat_size = num_queries * num_classes;
        std::vector<ScoredIndex> flat(static_cast<size_t>(flat_size));
        for (int i = 0; i < flat_size; ++i) {
            const int col = i;
            const float sc = Sigmoid(logits[i]);
            flat[static_cast<size_t>(i)] = {sc - static_cast<float>(col) * 1e-6f, i};
        }
        std::partial_sort(
            flat.begin(),
            flat.begin() + std::min(num_top, flat_size),
            flat.end(),
            [](const ScoredIndex& a, const ScoredIndex& b) { return a.score > b.score; });

        const int k = std::min(num_top, flat_size);
        for (int rank = 0; rank < k; ++rank) {
            const int idx = flat[static_cast<size_t>(rank)].flat_index;
            const float score = Sigmoid(logits[idx]);
            const int label = idx % num_classes;
            const int q = idx / num_classes;
            const float* box = pred_boxes + q * 4;
            float xyxy[4] = {box[0], box[1], box[2], box[3]};
            MapBoxToOrig(xyxy, orig_w, orig_h, network_input_size, box_map, letterbox);
            if (score < conf_threshold) continue;
            if (xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1]) continue;
            candidates.push_back(
                {label, score, xyxy[0], xyxy[1], xyxy[2], xyxy[3]});
        }
    } else {
        for (int q = 0; q < num_queries; ++q) {
            const float* logit_row = logits + q * num_classes;
            int best_label = 0;
            float best_score = -1.f;
            for (int c = 0; c < num_classes; ++c) {
                const float sc = logit_row[c];
                if (sc > best_score) {
                    best_score = sc;
                    best_label = c;
                }
            }
            const float* box = pred_boxes + q * 4;
            float xyxy[4] = {box[0], box[1], box[2], box[3]};
            MapBoxToOrig(xyxy, orig_w, orig_h, network_input_size, box_map, letterbox);
            if (best_score < conf_threshold) continue;
            if (xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1]) continue;
            candidates.push_back(
                {best_label, best_score, xyxy[0], xyxy[1], xyxy[2], xyxy[3]});
        }
    }

    return NmsPerClass(std::move(candidates), nms_threshold);
}

}  // namespace tea
