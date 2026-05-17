#include "tea_deploy/preprocess.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace tea {
namespace {

void BgrToNchw(
    const cv::Mat& rgb_f32_hwc,
    const PreprocessConfig& cfg,
    std::vector<float>* out) {
    const int h = rgb_f32_hwc.rows;
    const int w = rgb_f32_hwc.cols;
    out->resize(static_cast<size_t>(3) * h * w);
    std::vector<cv::Mat> ch(3);
    for (int c = 0; c < 3; ++c) {
        ch[c] = cv::Mat(h, w, CV_32F, out->data() + static_cast<size_t>(c) * h * w);
    }
    cv::split(rgb_f32_hwc, ch);
    if (!cfg.do_normalize) return;
    for (int c = 0; c < 3; ++c) {
        ch[c] = (ch[c] - cfg.mean[static_cast<size_t>(c)]) / cfg.std[static_cast<size_t>(c)];
    }
}

}  // namespace

bool PreprocessBgr(
    const cv::Mat& bgr,
    const PreprocessConfig& cfg,
    PreprocessMode mode,
    std::vector<float>* out,
    LetterboxMeta* letterbox) {
    if (bgr.empty() || bgr.type() != CV_8UC3 || !out) return false;

    const int input_size = cfg.input_size;
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat canvas(input_size, input_size, CV_8UC3, cv::Scalar(0, 0, 0));
    LetterboxMeta lb{};

    if (mode == PreprocessMode::Letterbox) {
        lb.ratio = std::min(
            static_cast<float>(input_size) / static_cast<float>(rgb.cols),
            static_cast<float>(input_size) / static_cast<float>(rgb.rows));
        lb.new_w = static_cast<int>(rgb.cols * lb.ratio);
        lb.new_h = static_cast<int>(rgb.rows * lb.ratio);
        lb.pad_w = (input_size - lb.new_w) / 2;
        lb.pad_h = (input_size - lb.new_h) / 2;
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(lb.new_w, lb.new_h), 0, 0, cv::INTER_LINEAR);
        resized.copyTo(canvas(cv::Rect(lb.pad_w, lb.pad_h, lb.new_w, lb.new_h)));
    } else {
        cv::resize(rgb, canvas, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);
        lb.ratio = 1.f;
        lb.pad_w = lb.pad_h = 0;
        lb.new_w = lb.new_h = input_size;
    }

    if (letterbox) *letterbox = lb;

    cv::Mat blob;
    canvas.convertTo(blob, CV_32FC3, cfg.do_rescale ? (1.0 / 255.0) : 1.0);
    BgrToNchw(blob, cfg, out);
    return true;
}

}  // namespace tea
