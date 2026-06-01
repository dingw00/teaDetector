/****  Copyright (c) 2026-2026 HongYun Tech. All rights reserved.  ****
 * \文件名    TeaDetector.h
 * \功能描述  TeaDetector 类声明，茶叶检测模块
 *
 * \创建者    dingw
 * \日期      May 2026
 *
 * \TODO
 * \修改日志
 *  20260528   dingw  Moved detector parameters and modelpath, model 
 *					  settings to SharedData.
 *  07May2026  dingw  Initial creation.
 *********************************************************************/
#pragma once
#include <string>
#include <onnxruntime_cxx_api.h>    // ONNX Runtime C++ API,用于加载和运行onnx模型
#include <vector>
#include <memory>
#include "../Common/SharedData.h"
#include "CamUtils.h"

//#include "CamUtils.h"
//#include "Setting.h"

//class SharedData;
// 将静态成员声明移到类外进行初始化
class CDetector {
protected:
	std::mutex m_mutex;
	SharedData* sharedData;

public:
	CDetector(SharedData* sharedData) : sharedData(sharedData) {};
	virtual ~CDetector() {};
	/**
	 * \brief 导入检测模型和权重文件
	 *
	 * \param modelPath，模型文件路径
	 * \param classFilePath，目标检测类别文件路径
	 */
	virtual bool loadModel(std::string modelPath, bool useGPU) = 0;
	/**
	 * \brief 检测茶叶嫩梢目标
	 *
	 * \param srcImg [in] 输入的源图像,要求8UC3格式的彩色图像
	 * \param output [out] 检测到的茶叶嫩梢对象集合,每个对象包含检测框、类型、置信度等信息
	 * \return 检测到的目标数量
	 */
	virtual void unloadModel() = 0;
	virtual int detect(const cv::Mat& rgb, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs) = 0;
	virtual int detect(const cv::Mat& rgb, std::vector<std::shared_ptr<TeaBud>>& teabuds) = 0;

};

class TeaDetectorYOLOX : public CDetector {
public:
	TeaDetectorYOLOX(SharedData* sharedData);
	~TeaDetectorYOLOX();

	/**
	 * \brief 导入检测模型和权重文件
	 *
	 * \param modelPath [in] 模型文件路径
	 * \param classFilePath [in] 目标检测类别文件路径
	 * \return =0，成功；< 0,错误代码
	 */
	bool loadModel(std::string modelPath, bool useGPU = false);
	void unloadModel();
	void _unloadModel();
	/**
	 * \brief 检测茶叶嫩梢目标
	 *
	 * \param rgb [in/out] 输入的源图像,要求8UC3格式的彩色图像
	 * \param teabuds [out] 检测到的茶叶嫩梢对象集合，检测框和置信度信息存储在TeaBud中
	 * \param display [in] 是否在输入图像上绘制检测框
	 * \return 检测到的目标数量
	 */
	int detect(const cv::Mat& rgb, std::vector<std::shared_ptr<TeaBud>>& teabuds);

	/**
	 * \brief 检测茶叶嫩梢目标
	 *
	 * \param rgb [in] 输入的源图像,要求8UC3格式的彩色图像
	 * \param bboxes [out] 检测到的目标检测框集合
	 * \param confs [out] 检测到的目标置信度集合
	 *
	 * \return 检测到的目标数量
	 */
	int detect(const cv::Mat& rgb, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs);

	/**
	 * \brief Setter for object detection parameters.
	 *
	 * \param confThres [in] 置信度阈值
	 * \param nmsThres [in] NMS非最大抑制阈值
	 */
	void setParams(float confThres = 0.2, float nmsThres = 0.3);

	/**
	 * \brief Getter for object detection parameters.
	 *
	 * \return A map containing the current values of confThres and nmsThres.
	 */
	std::map<std::string, float> getParams();

private:
	/**
	 * \brief 将图像按短边缩放并填充灰色到指定大小(INP_HEIGHT, INP_WIDTH)。注意缩放后的图片被放置在左上角（而不是居中）。
	 *
	 * \param srcimg [in] 源图片,类型8UC3
	 * \param scale [out] 计算出的缩放比例系数
	 * \return 缩放后的图片
	 */
	cv::Mat reImage(cv::Mat srcimg, float* scale);

	/**
	 * \brief BGR2RGB通道转化，将图片转化为浮点型，并进行归一化处理。(mean={0.485f, 0.456f, 0.406f}, std={0.229f, 0.224f, 0.225f})
	 *
	 * \param img [in/out] 转化前/后的图像
	 */
	void normalize(cv::Mat& img);

private:

	cv::dnn::Net m_nets;					//模型网络
	std::vector<std::string> classes;		//目标检测类名
};

struct BoundingBox {
	int x;
	int y;
	int width;
	int height;

	BoundingBox() : x(0), y(0), width(0), height(0) {}
	BoundingBox(int x_, int y_, int width_, int height_)
		: x(x_), y(y_), width(width_), height(height_) {
	}
};

struct DetectedObject {
	BoundingBox box;
	float conf{};
	int classId{};
};

#ifdef USE_ONNX
class TeaDetectorYOLO12 : public CDetector {
public:
	/**
	 * @brief Constructor to initialize the YOLO detector with model and label paths.       构造函数使用模型和标签路径初始化YOLO检测器。
	 *
	 * @param modelPath Path to the ONNX model file.                                        ONNX模型文件的路径
	 * @param labelsPath Path to the file containing class labels.                          包含类标签的文件的路径
	 * @param useGPU Whether to use GPU for inference (default is false).                   GPU是否使用GPU进行推理（默认值为false）
	 */
	TeaDetectorYOLO12(SharedData* sharedData);
	~TeaDetectorYOLO12();

	/**
	 * \brief 导入检测模型和权重文件
	 *
	 * \param modelPath [in] 模型文件路径
	 * \param classFilePath [in] 目标检测类别文件路径
	 * \return =0，成功；< 0,错误代码
	 */
	bool loadModel(std::string modelPath, bool useGPU = true);

	void unloadModel();

	void _unloadModel();

	/**
	 * \brief 检测茶叶嫩梢目标
	 *
	 * \param rgb [in] 输入的源图像,要求8UC3格式的彩色图像
	 * \param bboxes [out] 检测到的目标检测框集合
	 * \param confs [out] 检测到的目标置信度集合
	 *
	 * \return 检测到的目标数量
	 */
	int detect(const cv::Mat& rgb, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs);

	/**
	 * \brief 检测茶叶嫩梢目标
	 *
	 * \param rgb [in/out] 输入的源图像,要求8UC3格式的彩色图像
	 * \param teabuds [out] 检测到的茶叶嫩梢对象集合，检测框和置信度信息存储在TeaBud中
	 * \param display [in] 是否在输入图像上绘制检测框
	 * \return 检测到的目标数量
	 */
	int detect(const cv::Mat& rgb, std::vector<std::shared_ptr<TeaBud>>& teabuds);

private:

	Ort::Env env{ nullptr };                         // ONNX Runtime environment   
	Ort::SessionOptions sessionOptions{ nullptr };   // Session options for ONNX Runtime                ONNX运行时的会话选项
	Ort::Session session{ nullptr };                 // ONNX Runtime session for running inference      用于运行推理的ONNX运行时会话
	bool isDynamicInputShape{};                    // Flag indicating if input shape is dynamic         指示输入形状是否为动态的标志
	cv::Size inputImageShape;                      // Expected input image shape for the model          模型的预期输入图像形状

	// Vectors to hold allocated input and output node names                    用于保存分配的输入和输出节点名称的向量
	std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
	std::vector<const char*> inputNames;
	std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
	std::vector<const char*> outputNames;

	size_t numInputNodes, numOutputNodes;          // Number of input and output nodes in the model     模型中的输入和输出节点数量

	std::vector<std::string> classNames;            // Vector of class names loaded from file           从文件加载的类名向量
	std::vector<cv::Scalar> classColors;            // Vector of colors for each class                  每个类别的颜色矢量


private:
	/**
	 * @brief Preprocesses the input image for model inference.                         对输入图像进行预处理以进行模型推理
	 *
	 * @param image Input image.                                                        图像输入图像
	 * @param blob Reference to pointer where preprocessed data will be stored.         param blob指向存储预处理数据的指针的引用
	 * @param inputTensorShape Reference to vector representing input tensor shape.     param inputTensorShape对表示输入张量形状的向量的引用
	 * @return cv::Mat Resized image after preprocessing.                               预处理后调整大小的图像
	 */
	cv::Mat preprocess(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);

	/**
	 * @brief Postprocesses the model output to extract detections.         对模型输出进行后处理以提取检测结果
	 *
	 * @param originalImageSize Size of the original input image.           originalImageSize原始输入图像的大小
	 * @param resizedImageShape Size of the image after preprocessing.      resizedImageShape预处理后图像的大小
	 * @param outputTensors Vector of output tensors from the model.        outputTensors模型输出张量的向量
	 * @return std::vector<Detection> Vector of detections.                 std:：vector<Detection>检测向量
	 */
	std::vector<DetectedObject> postprocess(const cv::Size& originalImageSize, const cv::Size& resizedImageShape,
		const std::vector<Ort::Value>& outputTensors);
};


/// Legacy ONNX（如 dino_0329_30）：images + orig_target_sizes → labels/boxes/scores（图内含后处理）
/// HF ONNX（export_onnx 导出）：pixel_values → logits/pred_boxes（后处理在 C++ 完成）
enum class Dinov3OnnxBackend { Legacy, Hf };

class TeaDetectorDINOV3 : public CDetector {
public:
	// 默认构造函数：仅初始化 Ort 环境
	TeaDetectorDINOV3(SharedData* sharedData);

	/**
	 * @brief 加载/重新加载 ONNX 模型
	 * @param model_path 模型文件路径
	 * @return bool 是否加载成功
	 * useGPU 该函数中未使用，添加该变量仅为与CteaFinder类的loadmodel函数接口对齐
	 */
	bool loadModel(std::string model_path, bool useGPU = false);
	void unloadModel();
	void _unloadModel();
	
	/**
	 *  对提供的图像进行检测
	 *
	 *  img 输入图像进行检测
	 *  m_curTargets检测向量
	 *  threshold用于过滤检测的置信阈值（默认值为0.4）
	 *  非最大抑制的IoU阈值（默认值为0.45）未添加
	 */
	int detect(const cv::Mat& rgb, std::vector<std::shared_ptr<TeaBud>>& teabuds);

	int detect(const cv::Mat& rgb, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs);

private:
	Ort::Env env;
	std::unique_ptr<Ort::Session> session;
	int input_size = 640;
	bool is_model_loaded = false;

	Dinov3OnnxBackend onnx_backend_ = Dinov3OnnxBackend::Legacy;
	int num_queries_ = 300;
	int num_classes_ = 1;
	bool use_focal_loss_ = true;
	bool do_rescale_ = true;
	bool do_normalize_ = false;
	float image_mean_[3] = {0.485f, 0.456f, 0.406f};
	float image_std_[3] = {0.229f, 0.224f, 0.225f};

	void detectOnnxBackendFromSession();
	bool tryLoadMetaJson(const std::string& onnx_path);

	int detectLegacy(const cv::Mat& bgr, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs);
	int detectHf(const cv::Mat& bgr, std::vector<cv::Rect2f>& bboxes, std::vector<float>& confs);

	static float calculate_iou(const cv::Rect2f& a, const cv::Rect2f& b);
	std::vector<int> apply_nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, float iou_thresh);

};

#endif

