#pragma once

#include "trt_wrapper.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <string>
#include <vector>

template<typename PointT>
class TRTLMNet: public TRTWrapper {
  public:
	TRTLMNet();
	virtual ~TRTLMNet();

	// Configure headless output: KITTI .label predictions and/or range-view PNGs.
	void setOutput(const std::string& label_dir, const std::string& png_dir, bool save_png);

	// void inference(std::vector<PointT> scans, Eigen::Matrix4d pose);
	void inference(pcl::PointCloud<PointT>& scans, Eigen::Matrix4d pose);
	// void inference(std::vector<pcl::PointXYZI> scans, Eigen::Matrix4d pose, cv::Mat img);
	// void inference(std::vector<pcl::PointXYZI> scans, Eigen::Matrix4d pose, std::vector<cv::Mat> imgs);

  protected:
	// bool preprocess(std::vector<PointT> scans, Eigen::Matrix4d pose);
	bool preprocess(pcl::PointCloud<PointT>& scans, Eigen::Matrix4d pose);
	void rangeProjection(std::vector<Eigen::Vector4d> cur_vertex);
	// void postprocess(std::vector<PointT> scans);
	void postprocess(pcl::PointCloud<PointT>& scans);
	// void postprocess(std::vector<pcl::PointXYZI> scans, cv::Mat img);
	// void postprocess(std::vector<pcl::PointXYZI> scans, std::vector<cv::Mat> imgs);

	const int img_h_;
	const int img_w_;
	const int max_point_;
	const float proj_fov_up_;
	const float proj_fov_down_;
	const float min_range_;
	const float max_range_;

	bool init_;

	float img_means_[5];
	float img_stds_[5];

	float fov_;
	float fov_up_;
	float fov_down_;

	std::vector<int> proj_xs_;
	std::vector<int> proj_ys_;
	std::vector<int> pred_np_;

	std::vector<float> proj_range_;
	std::vector<float> proj_x_;
	std::vector<float> proj_y_;
	std::vector<float> proj_z_;
	std::vector<float> proj_remission_;
	std::vector<int> proj_idx_;
	std::vector<float> diff_image_;

	std::vector<float> cur_range_;
	std::vector<float> last_range_;

	std::vector<float> proj_output_;

	// std::vector<PointT> last_scans_;
	pcl::PointCloud<PointT> last_scans_;
	Eigen::Matrix4d last_pose_;

	cudaStream_t stream_;

	// Headless output state.
	void writeLabel(const pcl::PointCloud<PointT>& scans);

	std::string label_dir_;
	std::string png_dir_;
	bool save_png_ = false;
	int frame_idx_ = 0;
};

template class TRTLMNet<pcl::PointXYZI>;