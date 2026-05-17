#include "tea_deploy/detector.h"

#include <onnxruntime_cxx_api.h>

#include "tea_deploy/postprocess.h"
#include "tea_deploy/preprocess.h"

namespace tea {

struct TeaDetectorImpl {
    ModelConfig model_cfg{};
    InferenceConfig infer_cfg{};
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "tea_detector"};
    std::unique_ptr<Ort::Session> session;
};

TeaDetector::TeaDetector() : impl_(std::make_unique<TeaDetectorImpl>()) {}

TeaDetector::~TeaDetector() = default;

const ModelConfig& TeaDetector::model_config() const { return impl_->model_cfg; }

const InferenceConfig& TeaDetector::inference_config() const { return impl_->infer_cfg; }

bool TeaDetector::Init(
    const ModelConfig& model_cfg,
    const InferenceConfig& infer_cfg,
    std::string* err) {
    impl_->model_cfg = model_cfg;
    impl_->infer_cfg = infer_cfg;
    if (infer_cfg.preprocess_mode == PreprocessMode::Letterbox &&
        infer_cfg.box_map_mode == BoxMapMode::OrigTargetSize) {
        impl_->infer_cfg.box_map_mode = BoxMapMode::LetterboxInverse;
    }

    try {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, model_cfg.onnx_path.c_str(), opts);
    } catch (const Ort::Exception& e) {
        if (err) *err = e.what();
        impl_->session.reset();
        return false;
    }
    return true;
}

bool TeaDetector::InitFromMeta(
    const std::string& meta_json_path,
    const InferenceConfig& infer_cfg,
    std::string* err) {
    ModelConfig mc;
    if (!LoadMetaJson(meta_json_path, &mc, err)) return false;
    return Init(mc, infer_cfg, err);
}

std::vector<Detection> TeaDetector::Detect(const cv::Mat& bgr) const {
    if (!impl_->session || bgr.empty()) return {};

    LetterboxMeta lb{};
    std::vector<float> input;
    if (!PreprocessBgr(
            bgr,
            impl_->model_cfg.preprocess,
            impl_->infer_cfg.preprocess_mode,
            &input,
            &lb)) {
        return {};
    }

    const int h = impl_->model_cfg.input_size;
    const int w = impl_->model_cfg.input_size;
    const int64_t input_shape[4] = {1, 3, h, w};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input.data(), input.size(), input_shape, 4);

    const char* input_names[] = {"pixel_values"};
    const char* output_names[] = {"logits", "pred_boxes"};

    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        2);

    const float* logits = outputs[0].GetTensorData<float>();
    const float* boxes = outputs[1].GetTensorData<float>();

    BoxMapMode box_map = impl_->infer_cfg.box_map_mode;
    if (impl_->infer_cfg.preprocess_mode == PreprocessMode::Letterbox &&
        box_map == BoxMapMode::OrigTargetSize) {
        box_map = BoxMapMode::LetterboxInverse;
    }

    return PostprocessDetections(
        logits,
        boxes,
        impl_->model_cfg.num_queries,
        impl_->model_cfg.num_classes,
        impl_->model_cfg.use_focal_loss,
        bgr.cols,
        bgr.rows,
        impl_->model_cfg.input_size,
        box_map,
        lb,
        impl_->infer_cfg.conf_threshold,
        impl_->infer_cfg.nms_threshold);
}

}  // namespace tea
