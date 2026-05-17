#include "tea_deploy/config.h"

#include <cmath>
#include <fstream>
#include <sstream>

namespace tea {
namespace {

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

bool FindJsonFloatArray3(const std::string& text, const std::string& key, std::array<float, 3>* out) {
    const std::string pat = "\"" + key + "\"";
    auto pos = text.find(pat);
    if (pos == std::string::npos) return false;
    pos = text.find('[', pos);
    if (pos == std::string::npos) return false;
    auto end = text.find(']', pos);
    if (end == std::string::npos) return false;
    std::istringstream iss(text.substr(pos + 1, end - pos - 1));
    char comma = 0;
    for (int i = 0; i < 3; ++i) {
        iss >> (*out)[i];
        if (i < 2) iss >> comma;
    }
    return !iss.fail();
}

bool FindJsonString(const std::string& text, const std::string& key, std::string* out) {
    const std::string pat = "\"" + key + "\"";
    auto pos = text.find(pat);
    if (pos == std::string::npos) return false;
    pos = text.find('"', pos + pat.size());
    if (pos == std::string::npos) return false;
    auto end = text.find('"', pos + 1);
    if (end == std::string::npos) return false;
    *out = text.substr(pos + 1, end - pos - 1);
    return true;
}

std::string UnescapeJsonPath(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            out.push_back(s[i + 1]);
            ++i;
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

}  // namespace

bool LoadMetaJson(const std::string& meta_path, ModelConfig* out, std::string* err) {
    if (!out) {
        if (err) *err = "out is null";
        return false;
    }
    std::ifstream ifs(meta_path);
    if (!ifs) {
        if (err) *err = "cannot open meta: " + meta_path;
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    const std::string text = oss.str();

    ModelConfig cfg;
    std::string onnx_path;
    if (FindJsonString(text, "onnx", &onnx_path)) {
        cfg.onnx_path = UnescapeJsonPath(onnx_path);
    }
    FindJsonInt(text, "input_size", &cfg.input_size);
    FindJsonInt(text, "num_queries", &cfg.num_queries);
    FindJsonInt(text, "num_classes", &cfg.num_classes);
    FindJsonBool(text, "use_focal_loss", &cfg.use_focal_loss);

    cfg.preprocess.input_size = cfg.input_size;

    auto proc_pos = text.find("\"preprocessor_config\"");
    if (proc_pos != std::string::npos) {
        const std::string sub = text.substr(proc_pos);
        bool do_norm = cfg.preprocess.do_normalize;
        if (FindJsonBool(sub, "do_normalize", &do_norm)) cfg.preprocess.do_normalize = do_norm;
        bool do_rescale = cfg.preprocess.do_rescale;
        if (FindJsonBool(sub, "do_rescale", &do_rescale)) cfg.preprocess.do_rescale = do_rescale;
        FindJsonFloatArray3(sub, "image_mean", &cfg.preprocess.mean);
        FindJsonFloatArray3(sub, "image_std", &cfg.preprocess.std);
    }

    if (cfg.onnx_path.empty()) {
        if (err) *err = "meta.json missing onnx path";
        return false;
    }
    *out = cfg;
    return true;
}

}  // namespace tea
