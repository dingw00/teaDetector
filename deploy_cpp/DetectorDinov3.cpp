/****  Copyright (c) 2026-2026 HongYun Tech. All rights reserved.  ****
 * \文件名    DetectorDinov3.cpp
 * \功能描述  DINOv3 茶叶检测器实现
 *
 * \创建者    dingw
 * \日期      May 2026
 *
 * \TODO
 * \修改日志
 *  20260528   dingw  Moved detector parameters and modelpath, model
 *					  settings to SharedData.
 *  20260524   dingw  Renamed cp to teabud.
 *  07May2026  dingw  Initial creation.
 *********************************************************************/
#include "TeaDetector.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct HfDet {
    int label = 0;
    float score = 0.f;
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
};

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

void MapBoxToOrigStretch(float* xyxy, int orig_w, int orig_h) {
    float corners[4];
    CenterToCorners(xyxy, corners);
    const float sx = static_cast<float>(orig_w);
    const float sy = static_cast<float>(orig_h);
    xyxy[0] = corners[0] * sx;
    xyxy[1] = corners[1] * sy;
    xyxy[2] = corners[2] * sx;
    xyxy[3] = corners[3] * sy;
}

bool PreprocessStretchBgr(
    const cv::Mat& bgr,
    int input_size,
    bool do_rescale,
    bool do_normalize,
    const float mean[3],
    const float std[3],
    std::vector<float>* out) {
    if (bgr.empty() || bgr.type() != CV_8UC3 || !out) return false;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat canvas;
    cv::resize(rgb, canvas, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);

    cv::Mat blob;
    canvas.convertTo(blob, CV_32FC3, do_rescale ? (1.0 / 255.0) : 1.0);
    out->resize(static_cast<size_t>(3) * input_size * input_size);
    std::vector<cv::Mat> ch(3);
    for (int c = 0; c < 3; ++c) {
        ch[c] = cv::Mat(input_size, input_size, CV_32F,
            out->data() + static_cast<size_t>(c) * input_size * input_size);
    }
    cv::split(blob, ch);
    if (!do_normalize) return true;
    for (int c = 0; c < 3; ++c) {
        ch[c] = (ch[c] - mean[c]) / std[c];
    }
    return true;
}

std::vector<HfDet> PostprocessHfDetections(
    const float* logits,
    const float* pred_boxes,
    int num_queries,
    int num_classes,
    bool use_focal_loss,
    int orig_w,
    int orig_h,
    float conf_threshold) {
    const int num_top = num_queries;
    std::vector<HfDet> candidates;
    candidates.reserve(static_cast<size_t>(num_top));

    if (use_focal_loss) {
        const int flat_size = num_queries * num_classes;
        struct ScoredIndex {
            float score = 0.f;
            int flat_index = 0;
        };
        std::vector<ScoredIndex> flat(static_cast<size_t>(flat_size));
        for (int i = 0; i < flat_size; ++i) {
            const float sc = Sigmoid(logits[i]);
            flat[static_cast<size_t>(i)] = {sc - static_cast<float>(i) * 1e-6f, i};
        }
        const int k = std::min(num_top, flat_size);
        std::partial_sort(
            flat.begin(), flat.begin() + k, flat.end(),
            [](const ScoredIndex& a, const ScoredIndex& b) { return a.score > b.score; });

        for (int rank = 0; rank < k; ++rank) {
            const int idx = flat[static_cast<size_t>(rank)].flat_index;
            const float score = Sigmoid(logits[idx]);
            const int label = idx % num_classes;
            const int q = idx / num_classes;
            const float* box = pred_boxes + q * 4;
            float xyxy[4] = {box[0], box[1], box[2], box[3]};
            MapBoxToOrigStretch(xyxy, orig_w, orig_h);
            if (score < conf_threshold) continue;
            if (xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1]) continue;
            candidates.push_back({label, score, xyxy[0], xyxy[1], xyxy[2], xyxy[3]});
        }
    } else {
        for (int q = 0; q < num_queries; ++q) {
            const float* logit_row = logits + q * num_classes;
            int best_label = 0;
            float best_score = -1.f;
            for (int c = 0; c < num_classes; ++c) {
                if (logit_row[c] > best_score) {
                    best_score = logit_row[c];
                    best_label = c;
                }
            }
            const float* box = pred_boxes + q * 4;
            float xyxy[4] = {box[0], box[1], box[2], box[3]};
            MapBoxToOrigStretch(xyxy, orig_w, orig_h);
            if (best_score < conf_threshold) continue;
            if (xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1]) continue;
            candidates.push_back({best_label, best_score, xyxy[0], xyxy[1], xyxy[2], xyxy[3]});
        }
    }
    return candidates;
}

std::vector<HfDet> NmsPerClassHf(std::vector<HfDet> dets, float nms_thres) {
    if (dets.empty() || nms_thres >= 1.f) {
        std::sort(dets.begin(), dets.end(), [](const HfDet& a, const HfDet& b) {
            return a.score > b.score;
        });
        return dets;
    }
    std::sort(dets.begin(), dets.end(), [](const HfDet& a, const HfDet& b) {
        return a.score > b.score;
    });
    std::vector<HfDet> out;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (dets[i].label != dets[j].label) continue;
            const float xx1 = std::max(dets[i].x1, dets[j].x1);
            const float yy1 = std::max(dets[i].y1, dets[j].y1);
            const float xx2 = std::min(dets[i].x2, dets[j].x2);
            const float yy2 = std::min(dets[i].y2, dets[j].y2);
            const float w = std::max(0.f, xx2 - xx1);
            const float h = std::max(0.f, yy2 - yy1);
            const float inter = w * h;
            const float area_a = std::max(0.f, dets[i].x2 - dets[i].x1) * std::max(0.f, dets[i].y2 - dets[i].y1);
            const float area_b = std::max(0.f, dets[j].x2 - dets[j].x1) * std::max(0.f, dets[j].y2 - dets[j].y1);
            const float uni = area_a + area_b - inter;
            const float iou = uni <= 0.f ? 0.f : inter / uni;
            if (iou > nms_thres) removed[j] = true;
        }
    }
    return out;
}

std::string MetaPathForOnnx(const std::string& onnx_path) {
    const auto pos = onnx_path.find_last_of('.');
    if (pos == std::string::npos) return onnx_path + ".meta.json";
    return onnx_path.substr(0, pos) + ".meta.json";
}

bool FindJsonBool(const std::string& text, const std::string& key, bool* out) {
    const std::string pat = "\"" + key + "\"";
    auto pos = text.find(pat);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos = text.find_first_not_of(" \t\n\r", pos + 1);
    if (pos == std::string::npos) return false;
    if (text.compare(pos, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (text.compare(pos, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

bool FindJsonInt(const std::string& text, const std::string& key, int* out) {
    const std::string pat = "\"" + key + "\"";
    auto pos = text.find(pat);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    std::istringstream iss(text.substr(pos + 1));
    iss >> *out;
    return !iss.fail();
}

bool FindJsonFloatArray3(const std::string& text, const std::string& key, float out[3]) {
    const std::string pat = "\"" + key + "\"";
    auto pos = text.find(pat);
    if (pos == std::string::npos) return false;
    pos = text.find('[', pos);
    if (pos == std::string::npos) return false;
    const auto end = text.find(']', pos);
    if (end == std::string::npos) return false;
    std::istringstream iss(text.substr(pos + 1, end - pos - 1));
    char comma = 0;
    for (int i = 0; i < 3; ++i) {
        iss >> out[i];
        if (i < 2) iss >> comma;
    }
    return !iss.fail();
}

}  // namespace

TeaDetectorDINOV3::TeaDetectorDINOV3(SharedData* sharedData) : CDetector(sharedData) {
    // 扫描文件夹下的可用模型文件
    std::vector<std::string> modelPaths = camUtils::scanFilesEndWithX(readConfig("projDir").toStdString() +
        "resource/detect_models/DINOV3/", ".onnx");
    if (modelPaths.empty()) {
        qDebug() << "No Tea Detector Model Files Found.";
    }
    std::vector<QString> qModelPaths;
    for (const auto& path : modelPaths) {
        qModelPaths.push_back(QString::fromStdString(path));
    }
    sharedData->setDetectorModelChoices(qModelPaths);
}

void TeaDetectorDINOV3::detectOnnxBackendFromSession() {
    onnx_backend_ = Dinov3OnnxBackend::Legacy;
    if (!session) return;

    bool has_logits = false;
    bool has_pred_boxes = false;
    bool has_labels = false;
    bool has_boxes = false;
    bool has_scores = false;

    const size_t n_out = session->GetOutputCount();
    for (size_t i = 0; i < n_out; ++i) {
        const auto name = session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        const std::string out_name = name.get();
        if (out_name == "logits") has_logits = true;
        if (out_name == "pred_boxes") has_pred_boxes = true;
        if (out_name == "labels") has_labels = true;
        if (out_name == "boxes") has_boxes = true;
        if (out_name == "scores") has_scores = true;
    }

    if (has_logits && has_pred_boxes) {
        onnx_backend_ = Dinov3OnnxBackend::Hf;
        return;
    }
    if (has_labels && has_boxes && has_scores) {
        onnx_backend_ = Dinov3OnnxBackend::Legacy;
        return;
    }

    std::cerr << "Warning: unknown ONNX outputs; defaulting to Legacy backend." << std::endl;
}

bool TeaDetectorDINOV3::tryLoadMetaJson(const std::string& onnx_path) {
    const std::string meta_path = MetaPathForOnnx(onnx_path);
    std::ifstream ifs(meta_path);
    if (!ifs) {
        qDebug() << "HF ONNX: no meta.json at" << QString::fromStdString(meta_path)
                 << ", using defaults.";
        return false;
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    const std::string text = oss.str();

    int v = 0;
    if (FindJsonInt(text, "input_size", &v) && v > 0) input_size = v;
    if (FindJsonInt(text, "num_queries", &v) && v > 0) num_queries_ = v;
    if (FindJsonInt(text, "num_classes", &v) && v > 0) num_classes_ = v;
    if (FindJsonBool(text, "use_focal_loss", &use_focal_loss_)) { /* ok */ }

    auto proc_pos = text.find("\"preprocessor_config\"");
    if (proc_pos != std::string::npos) {
        const std::string sub = text.substr(proc_pos);
        bool b = do_rescale_;
        if (FindJsonBool(sub, "do_rescale", &b)) do_rescale_ = b;
        b = do_normalize_;
        if (FindJsonBool(sub, "do_normalize", &b)) do_normalize_ = b;
        FindJsonFloatArray3(sub, "image_mean", image_mean_);
        FindJsonFloatArray3(sub, "image_std", image_std_);
    }

    auto prep_pos = text.find("\"preprocess_config\"");
    if (prep_pos != std::string::npos) {
        const std::string sub = text.substr(prep_pos);
        bool b = do_rescale_;
        if (FindJsonBool(sub, "do_rescale", &b)) do_rescale_ = b;
        b = do_normalize_;
        if (FindJsonBool(sub, "do_normalize", &b)) do_normalize_ = b;
        FindJsonFloatArray3(sub, "image_mean", image_mean_);
        FindJsonFloatArray3(sub, "image_std", image_std_);
    }

    qDebug() << "HF ONNX meta:" << QString::fromStdString(meta_path)
             << "classes=" << num_classes_ << "queries=" << num_queries_
             << "normalize=" << do_normalize_;
    return true;
}

bool TeaDetectorDINOV3::loadModel(std::string modelPath, bool useGPU) {
    (void)useGPU;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (sharedData->camStatus.teaDetectorStatus->detectorModelLoaded) {
        if (modelPath == sharedData->getDetectorModelPath().toStdString()) {
            qDebug() << "Tea Detector DINOV3 model already loaded.";
            return true;
        }
        _unloadModel();
    }

    std::vector<QString> modelPaths = sharedData->getDetectorModelChoices();
    const QString qPath = QString::fromStdString(modelPath);
    if (std::find(modelPaths.begin(), modelPaths.end(), qPath) == modelPaths.end()) {
        qDebug() << "Model file not found in available model files.";
        return false;
    }

    qDebug() << "<< Loading Tea Detector DINOV3 model:" << QString::fromStdString(modelPath);
    try {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        const std::wstring w_path(modelPath.begin(), modelPath.end());
        session = std::make_unique<Ort::Session>(env, w_path.c_str(), options);
#else
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
#endif

        const auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() >= 4 && input_shape[2] > 0) {
            input_size = static_cast<int>(input_shape[2]);
        }

        detectOnnxBackendFromSession();

        if (onnx_backend_ == Dinov3OnnxBackend::Hf) {
            tryLoadMetaJson(modelPath);
            const size_t n_out = session->GetOutputCount();
            for (size_t i = 0; i < n_out; ++i) {
                const auto out_name =
                    session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                if (out_name.get() != std::string("logits")) continue;
                const auto logits_shape =
                    session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
                if (logits_shape.size() >= 3) {
                    if (logits_shape[1] > 0) num_queries_ = static_cast<int>(logits_shape[1]);
                    if (logits_shape[2] > 0) num_classes_ = static_cast<int>(logits_shape[2]);
                }
                break;
            }
            qDebug() << ">> Tea Detector DINOV3 HF backend (pixel_values -> logits/pred_boxes).";
        } else {
            qDebug() << ">> Tea Detector DINOV3 Legacy backend (letterbox + labels/boxes/scores).";
        }

        sharedData->camStatus.teaDetectorStatus->detectorModelLoaded = true;
        sharedData->setDetectorModelPath(qPath);
        is_model_loaded = true;
        qDebug() << ">> Tea Detector DINOV3 model loaded.";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        sharedData->camStatus.teaDetectorStatus->detectorModelLoaded = false;
        is_model_loaded = false;
        return false;
    }
}

void TeaDetectorDINOV3::unloadModel() {
    std::lock_guard<std::mutex> lock(m_mutex);
    _unloadModel();
}

void TeaDetectorDINOV3::_unloadModel() {
    session.reset();
    onnx_backend_ = Dinov3OnnxBackend::Legacy;
    sharedData->camStatus.teaDetectorStatus->detectorModelLoaded = false;
    is_model_loaded = false;
}

int TeaDetectorDINOV3::detect(const cv::Mat& rgb, std::vector<std::shared_ptr<TeaBud>>& teabuds) {
    std::vector<cv::Rect2f> bboxes;
    std::vector<float> confs;
    const int n = detect(rgb, bboxes, confs);
    teabuds.clear();
    for (int i = 0; i < n; ++i) {
        teabuds.push_back(std::make_shared<TeaBud>(i, bboxes[static_cast<size_t>(i)], confs[static_cast<size_t>(i)]));
    }
    return n;
}

int TeaDetectorDINOV3::detect(const cv::Mat& rgb, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs) {
    if (!sharedData->camStatus.teaDetectorStatus->detectorModelLoaded || !session) {
        std::cerr << "Error: Model not loaded!" << std::endl;
        bboxes.clear();
        confs.clear();
        return 0;
    }
    if (onnx_backend_ == Dinov3OnnxBackend::Hf) {
        return detectHf(rgb, bboxes, confs);
    }
    return detectLegacy(rgb, bboxes, confs);
}

int TeaDetectorDINOV3::detectLegacy(const cv::Mat& rgb, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs) {
    const float ratio = std::min(static_cast<float>(input_size) / rgb.cols,
        static_cast<float>(input_size) / rgb.rows);
    const int new_w = static_cast<int>(rgb.cols * ratio);
    const int new_h = static_cast<int>(rgb.rows * ratio);
    const int pad_w = (input_size - new_w) / 2;
    const int pad_h = (input_size - new_h) / 2;

    cv::Mat resized, canvas(input_size, input_size, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::resize(rgb, resized, cv::Size(new_w, new_h));
    resized.copyTo(canvas(cv::Rect(pad_w, pad_h, new_w, new_h)));

    cv::Mat blob;
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(blob, CV_32FC3, 1.0 / 255.0);

    std::vector<float> input_tensor_values(static_cast<size_t>(input_size) * input_size * 3);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(input_size, input_size, CV_32F,
            input_tensor_values.data() + static_cast<size_t>(i) * input_size * input_size);
    }
    cv::split(blob, channels);

    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const int64_t img_shape[] = {1, 3, input_size, input_size};
    const int64_t tgt_shape[] = {1, 2};
    int64_t tgt_data[] = {static_cast<int64_t>(input_size), static_cast<int64_t>(input_size)};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, input_tensor_values.data(), input_tensor_values.size(), img_shape, 4));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem, tgt_data, 2, tgt_shape, 2));

    const char* in_names[] = {"images", "orig_target_sizes"};
    const char* out_names[] = {"labels", "boxes", "scores"};

    auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), inputs.size(), out_names, 3);

    float* boxes_ptr = outputs[1].GetTensorMutableData<float>();
    float* scores_ptr = outputs[2].GetTensorMutableData<float>();
    const size_t count = outputs[1].GetTensorTypeAndShapeInfo().GetShape()[1];

    bboxes.clear();
    confs.clear();

    const int orig_w = rgb.cols;
    const int orig_h = rgb.rows;
    const float conf_thres = sharedData->camStatus.teaDetectorStatus->confThres;

    for (size_t i = 0; i < count; i++) {
        if (scores_ptr[i] < conf_thres) continue;

        float x1 = (boxes_ptr[i * 4 + 0] - pad_w) / ratio;
        float y1 = (boxes_ptr[i * 4 + 1] - pad_h) / ratio;
        float x2 = (boxes_ptr[i * 4 + 2] - pad_w) / ratio;
        float y2 = (boxes_ptr[i * 4 + 3] - pad_h) / ratio;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        const float w = std::max(0.0f, x2 - x1);
        const float h = std::max(0.0f, y2 - y1);
        if (w > 0 && h > 0) {
            bboxes.emplace_back(x1, y1, w, h);
            confs.push_back(scores_ptr[i]);
        }
    }

    const std::vector<int> keep_idxs =
        apply_nms(bboxes, confs, sharedData->camStatus.teaDetectorStatus->nmsThres);
    std::vector<cv::Rect2f> nms_bboxes;
    std::vector<float> nms_confs;
    nms_bboxes.reserve(keep_idxs.size());
    nms_confs.reserve(keep_idxs.size());
    for (int idx : keep_idxs) {
        nms_bboxes.push_back(bboxes[static_cast<size_t>(idx)]);
        nms_confs.push_back(confs[static_cast<size_t>(idx)]);
    }
    bboxes.swap(nms_bboxes);
    confs.swap(nms_confs);
    return static_cast<int>(confs.size());
}

int TeaDetectorDINOV3::detectHf(const cv::Mat& rgb, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs) {
    const int orig_w = rgb.cols;
    const int orig_h = rgb.rows;
    const float conf_thres = sharedData->camStatus.teaDetectorStatus->confThres;
    const float nms_thres = sharedData->camStatus.teaDetectorStatus->nmsThres;

    std::vector<float> input_tensor_values;
    if (!PreprocessStretchBgr(rgb, input_size, do_rescale_, do_normalize_, image_mean_, image_std_,
            &input_tensor_values)) {
        bboxes.clear();
        confs.clear();
        return 0;
    }

    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const int64_t img_shape[] = {1, 3, input_size, input_size};

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, input_tensor_values.data(), input_tensor_values.size(), img_shape, 4));

    const char* in_names[] = {"pixel_values"};
    const char* out_names[] = {"logits", "pred_boxes"};

    auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 1, out_names, 2);

    const float* logits = outputs[0].GetTensorData<float>();
    const float* pred_boxes = outputs[1].GetTensorData<float>();

    std::vector<HfDet> dets = PostprocessHfDetections(
        logits, pred_boxes, num_queries_, num_classes_, use_focal_loss_, orig_w, orig_h, conf_thres);
    dets = NmsPerClassHf(std::move(dets), nms_thres);

    bboxes.clear();
    confs.clear();
    bboxes.reserve(dets.size());
    confs.reserve(dets.size());
    for (const auto& d : dets) {
        const float w = std::max(0.f, d.x2 - d.x1);
        const float h = std::max(0.f, d.y2 - d.y1);
        if (w <= 0.f || h <= 0.f) continue;
        bboxes.emplace_back(d.x1, d.y1, w, h);
        confs.push_back(d.score);
    }
    return static_cast<int>(confs.size());
}

float TeaDetectorDINOV3::calculate_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    const float uni = a.area() + b.area() - inter;
    return (uni <= 0) ? 0 : inter / uni;
}

std::vector<int> TeaDetectorDINOV3::apply_nms(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float iou_thresh) {
    std::vector<int> idxs(boxes.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    while (!idxs.empty()) {
        const int cur = idxs[0];
        keep.push_back(cur);
        std::vector<int> rest;
        for (size_t i = 1; i < idxs.size(); ++i) {
            if (calculate_iou(boxes[cur], boxes[idxs[i]]) <= iou_thresh) {
                rest.push_back(idxs[i]);
            }
        }
        idxs.swap(rest);
    }
    return keep;
}
