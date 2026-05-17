/**
 * 茶叶检测 C++ 部署示例（ONNX Runtime + OpenCV）
 *
 * 构建见 deploy_cpp/CMakeLists.txt
 *
 * 当前仓库 checkpoint（拉伸 640 + ÷255 + ImageNet mean/std）：
 *   tea_detector_demo meta.json image.jpg --mode stretch
 *
 * Letterbox（与 eval_tea.preprocess_bgr 一致，未按此方式训练的权重精度会下降）：
 *   tea_detector_demo meta.json image.jpg --mode letterbox
 */

#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tea_deploy/detector.h"

namespace {

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage:\n  " << argv0
        << " <model.meta.json> <image_path> [--mode stretch|letterbox]\n"
        << "  [--conf 0.2] [--nms 0.3]\n";
}

tea::InferenceConfig ParseArgs(int argc, char** argv) {
    tea::InferenceConfig cfg;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m == "letterbox") {
                cfg.preprocess_mode = tea::PreprocessMode::Letterbox;
                cfg.box_map_mode = tea::BoxMapMode::LetterboxInverse;
            } else {
                cfg.preprocess_mode = tea::PreprocessMode::Stretch;
                cfg.box_map_mode = tea::BoxMapMode::OrigTargetSize;
            }
        } else if (arg == "--conf" && i + 1 < argc) {
            cfg.conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            cfg.nms_threshold = std::stof(argv[++i]);
        }
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        PrintUsage(argv[0]);
        return 1;
    }

    const std::string meta_path = argv[1];
    const std::string image_path = argv[2];
    tea::InferenceConfig infer_cfg = ParseArgs(argc, argv);

    tea::TeaDetector detector;
    std::string err;
    if (!detector.InitFromMeta(meta_path, infer_cfg, &err)) {
        std::cerr << "Init failed: " << err << "\n";
        return 1;
    }

    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) {
        std::cerr << "Cannot read image: " << image_path << "\n";
        return 1;
    }

    const auto dets = detector.Detect(bgr);
    std::cout << "detections=" << dets.size() << "\n";
    for (const auto& d : dets) {
        std::cout << "label=" << d.label << " score=" << d.score << " box=[" << d.x1 << "," << d.y1
                  << "," << d.x2 << "," << d.y2 << "]\n";
    }

    static const char* kNames[] = {"I", "Y"};
    for (const auto& d : dets) {
        const int lid = std::max(0, std::min(d.label, 1));
        cv::rectangle(
            bgr,
            cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
            cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)),
            cv::Scalar(0, 255, 0),
            2);
        const std::string tag =
            std::string(kNames[lid]) + " " + std::to_string(static_cast<int>(d.score * 100) / 100.f);
        cv::putText(
            bgr,
            tag,
            cv::Point(static_cast<int>(d.x1), std::max(0, static_cast<int>(d.y1) - 4)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 0),
            2);
    }
    cv::imwrite("out_cpp.jpg", bgr);
    std::cout << "saved out_cpp.jpg\n";
    return 0;
}
