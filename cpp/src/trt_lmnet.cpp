#include "trt_lmnet.h"

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

template<typename PointT>
TRTLMNet<PointT>::TRTLMNet()
    : img_h_(64)
    , img_w_(2048)
    , max_point_(300000)
    // KITTI HDL-64 sensor / residual params (see config/data_preparing.yaml and
    // the model's arch_cfg.yaml). min/max range bound the residual valid mask.
    , proj_fov_up_(3)
    , proj_fov_down_(-25)
    , min_range_(2.0f)
    , max_range_(50.0f) {
	init_ = false;

	// img_means_[0] = 12.12f;     img_means_[1] = 10.88f;     img_means_[2] = 0.23f;      img_means_[3] = -1.04f;     img_means_[4] = 0.21f;
	// img_stds_[0] = 12.32f;      img_stds_[1] = 11.47f;      img_stds_[2] = 6.91f;       img_stds_[3] = 0.86f;       img_stds_[4] = 0.16f;

	// img_means_[0] = 15.79f;     img_means_[1] = 0.27f;     img_means_[2] = 0.08f;      img_means_[3] = 3.12f;     img_means_[4] = 33.76f;
	// img_stds_[0] = 11.02f;      img_stds_[1] = 12.88f;      img_stds_[2] = 9.79f;       img_stds_[3] = 7.03f;       img_stds_[4] = 23.09f;

	// img_means_[0] = 6.15f;     img_means_[1] = 0.10f;     img_means_[2] = 0.87f;      img_means_[3] = 1.35f;     img_means_[4] = 36.40f;
	// img_stds_[0] = 3.04f;      img_stds_[1] = 3.52f;      img_stds_[2] = 4.13f;       img_stds_[3] = 2.12f;       img_stds_[4] = 24.91f;

	// img_means_[0] = 10.04f;
	// img_means_[1] = -0.28f;
	// img_means_[2] = 0.06f;
	// img_means_[3] = -0.49f;
	// img_means_[4] = 19.21f;
	// img_stds_[0] = 12.66f;
	// img_stds_[1] = 12.16f;
	// img_stds_[2] = 10.32f;
	// img_stds_[3] = 1.86f;
	// img_stds_[4] = 21.82f;

	// img_means_[0] = 2.35f;     img_means_[1] = 0.22f;     img_means_[2] = 0.072f;      img_means_[3] = 1.07f;     img_means_[4] = 25.65f;
	// img_stds_[0] = 2.09f;      img_stds_[1] = 1.84f;      img_stds_[2] = 1.93f;       img_stds_[3] = 1.19f;       img_stds_[4] = 27.24f;


	// KITTI normalization (range, x, y, z, remission) from arch_cfg.yaml.
	img_means_[0] = 12.12f;    img_means_[1] = 10.88f;    img_means_[2] = 0.23f;      img_means_[3] = -1.04f;     img_means_[4] = 0.21f;
	img_stds_[0] = 12.32f;     img_stds_[1] = 11.47f;     img_stds_[2] = 6.91f;       img_stds_[3] = 0.86f;       img_stds_[4] = 0.16f;

	// Standard spherical projection (matches utils/utils.py range_projection).
	fov_up_ = proj_fov_up_ / 180.0f * M_PI;
	fov_down_ = proj_fov_down_ / 180.0f * M_PI;
	fov_ = std::abs(fov_down_) + std::abs(fov_up_);

	proj_range_.resize(img_h_ * img_w_);

	proj_x_.resize(img_h_ * img_w_);
	proj_y_.resize(img_h_ * img_w_);
	proj_z_.resize(img_h_ * img_w_);

	proj_remission_.resize(img_h_ * img_w_);
	proj_idx_.resize(img_h_ * img_w_);
	diff_image_.resize(img_h_ * img_w_);

	cur_range_.resize(img_h_ * img_w_);
	last_range_.resize(img_h_ * img_w_);

	cudaStreamCreate(&stream_);
}

template<typename PointT>
TRTLMNet<PointT>::~TRTLMNet() {
	proj_range_.clear();

	proj_x_.clear();
	proj_y_.clear();
	proj_z_.clear();

	proj_remission_.clear();
	proj_idx_.clear();
	diff_image_.clear();

	cur_range_.clear();
	last_range_.clear();

	cudaStreamDestroy(stream_);
}

template<typename PointT>
void TRTLMNet<PointT>::setOutput(const std::string& label_dir, const std::string& png_dir, bool save_png) {
	label_dir_ = label_dir;
	png_dir_ = png_dir;
	save_png_ = save_png;

	if (!label_dir_.empty()) {
		std::filesystem::create_directories(label_dir_);
	}

	if (save_png_ && !png_dir_.empty()) {
		std::filesystem::create_directories(png_dir_);
	}
}

template<typename PointT>
void TRTLMNet<PointT>::writeLabel(const pcl::PointCloud<PointT>& scans) {
	if (label_dir_.empty()) {
		return;
	}

	// learning_map_inv: class 0 -> 0 (ignored), 1 -> 9 (static), 2 -> 251 (moving).
	static const uint32_t kLabelMap[3] = {0u, 9u, 251u};

	std::vector<uint32_t> labels(scans.points.size(), 0u);

	for (size_t i = 0; i < scans.points.size(); ++i) {
		int c = pred_np_[i];
		labels[i] = (c >= 0 && c < 3) ? kLabelMap[c] : 0u;
	}

	std::stringstream ss;
	ss << label_dir_ << "/" << std::setw(6) << std::setfill('0') << frame_idx_ << ".label";

	std::ofstream ofs(ss.str(), std::ios::binary);
	ofs.write(reinterpret_cast<const char*>(labels.data()), labels.size() * sizeof(uint32_t));
	ofs.close();
}

inline bool cmp(std::pair<int, float>& a, std::pair<int, float>& b) {
	if (a.second == b.second) {
		return a.first > b.first;
	}

	return a.second > b.second;
}

template<typename PointT>
void TRTLMNet<PointT>::rangeProjection(std::vector<Eigen::Vector4d> cur_vertex) {
	std::fill(last_range_.begin(), last_range_.end(), -1.0f);

	std::vector<std::pair<int, float>> depths;
	std::vector<int> proj_xs;
	std::vector<int> proj_ys;

	depths.resize(cur_vertex.size());
	proj_xs.resize(cur_vertex.size());
	proj_ys.resize(cur_vertex.size());

	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < cur_vertex.size(); ++i) {
		float depth = std::sqrt(cur_vertex[i](0) * cur_vertex[i](0) + cur_vertex[i](1) * cur_vertex[i](1) + cur_vertex[i](2) * cur_vertex[i](2));

		float yaw = -std::atan2(cur_vertex[i](1), cur_vertex[i](0));
		float pitch = std::asin(cur_vertex[i](2) / depth);

		int proj_x = std::floor((0.5f * (yaw / M_PI + 1.0)) * img_w_);
		int proj_y = std::floor((1.0f - (pitch + std::abs(fov_down_)) / fov_) * img_h_);

		proj_x = std::max(0, std::min(img_w_ - 1, proj_x));
		proj_y = std::max(0, std::min(img_h_ - 1, proj_y));

		// depths.push_back({i, depth});
		// proj_xs.push_back(proj_x);
		// proj_ys.push_back(proj_y);

		depths[i] = std::make_pair(i, depth);
		proj_xs[i] = proj_x;
		proj_ys[i] = proj_y;
	}

	std::sort(depths.begin(), depths.end(), cmp);

	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < depths.size(); ++i) {
		if (depths[i].second <= min_range_ || depths[i].second >= max_range_) {
			continue;
		}

		int order = depths[i].first;

		last_range_[proj_ys[order] * img_w_ + proj_xs[order]] = depths[i].second;
	}
}

template<typename PointT>
// bool TRTLMNet<PointT>::preprocess(std::vector<PointT> scans, Eigen::Matrix4d pose) {
bool TRTLMNet<PointT>::preprocess(pcl::PointCloud<PointT>& scans, Eigen::Matrix4d pose) {
	std::fill(proj_range_.begin(), proj_range_.end(), 0.0f);

	std::fill(proj_x_.begin(), proj_x_.end(), 0.0f);
	std::fill(proj_y_.begin(), proj_y_.end(), 0.0f);
	std::fill(proj_z_.begin(), proj_z_.end(), 0.0f);

	std::fill(proj_remission_.begin(), proj_remission_.end(), 0.0f);

	std::fill(proj_idx_.begin(), proj_idx_.end(), 0);
	std::fill(diff_image_.begin(), diff_image_.end(), 0.0f);

	std::fill(cur_range_.begin(), cur_range_.end(), -1.0f);

	std::vector<std::pair<int, float>> depths;

	proj_xs_.clear();
	proj_ys_.clear();

	depths.resize(scans.points.size());
	proj_xs_.resize(scans.points.size());
	proj_ys_.resize(scans.points.size());
	
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < scans.points.size(); ++i) {
		float depth = std::sqrt(scans.points[i].x * scans.points[i].x + scans.points[i].y * scans.points[i].y + scans.points[i].z * scans.points[i].z);

		float yaw = -std::atan2(scans.points[i].y, scans.points[i].x);
		float pitch = std::asin(scans.points[i].z / depth);

		int proj_x = std::floor((0.5f * (yaw / M_PI + 1.0)) * img_w_);
		int proj_y = std::floor((1.0f - (pitch + std::abs(fov_down_)) / fov_) * img_h_);

		proj_x = std::max(0, std::min(img_w_ - 1, proj_x));
		proj_y = std::max(0, std::min(img_h_ - 1, proj_y));

		// depths.push_back({i, depth});
		// proj_xs_.push_back(proj_x);
		// proj_ys_.push_back(proj_y);

		depths[i] = std::make_pair(i, depth);
		proj_xs_[i] = proj_x;
		proj_ys_[i] = proj_y;
	}

	std::sort(depths.begin(), depths.end(), cmp);

	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < depths.size(); ++i) {
		// if(depths[i].second <= LMNET_MIN_RANGE || depths[i].second >= LMNET_MAX_RANGE) {
		//     continue;
		// }

		int order = depths[i].first;

		cur_range_[proj_ys_[order] * img_w_ + proj_xs_[order]] = depths[i].second;

		proj_range_[proj_ys_[order] * img_w_ + proj_xs_[order]] = (depths[i].second - img_means_[0]) / img_stds_[0];

		proj_x_[proj_ys_[order] * img_w_ + proj_xs_[order]] = (scans.points[order].x - img_means_[1]) / img_stds_[1];
		proj_y_[proj_ys_[order] * img_w_ + proj_xs_[order]] = (scans.points[order].y - img_means_[2]) / img_stds_[2];
		proj_z_[proj_ys_[order] * img_w_ + proj_xs_[order]] = (scans.points[order].z - img_means_[3]) / img_stds_[3];

		proj_remission_[proj_ys_[order] * img_w_ + proj_xs_[order]] = (scans.points[order].intensity - img_means_[4]) / img_stds_[4];

		proj_idx_[proj_ys_[order] * img_w_ + proj_xs_[order]] = 1;
	}

	Eigen::Matrix4d pose_dot = pose.inverse() * last_pose_;

	std::vector<Eigen::Vector4d> last_scan_transformed;

	last_scan_transformed.resize(last_scans_.size());

	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < last_scans_.size(); ++i) {
		Eigen::Vector4d last_vec(last_scans_[i].x, last_scans_[i].y, last_scans_[i].z, 1.f);

		// Transform the last scan into the current frame:
		// inv(current_pose) * last_pose * last_point (matches gen_residual_images.py).
		last_scan_transformed[i] = pose_dot * last_vec;
	}

	rangeProjection(last_scan_transformed);

	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < diff_image_.size(); ++i) {
		if (cur_range_[i] <= min_range_ || cur_range_[i] >= max_range_ || last_range_[i] <= min_range_ || last_range_[i] >= max_range_) {
			continue;
		}

		if (proj_idx_[i] == 0) {
			continue;
		}

		diff_image_[i] = std::abs(cur_range_[i] - last_range_[i]) / cur_range_[i];
	}

	return true;
}

template<typename PointT>
// void TRTLMNet<PointT>::postprocess(std::vector<PointT> scans) {
void TRTLMNet<PointT>::postprocess(pcl::PointCloud<PointT>& scans) {
// void TRTLMNet::postprocess(std::vector<pcl::PointXYZI> scans, cv::Mat img)
// void TRTLMNet::postprocess(std::vector<pcl::PointXYZI> scans, std::vector<cv::Mat> imgs) {
	pred_np_.clear();

	pred_np_.resize(scans.points.size());

	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < scans.points.size(); ++i) {
		int y_idx = proj_ys_[i];
		int x_idx = proj_xs_[i];

		std::vector<float> pred_val;

		pred_val.push_back(proj_output_[y_idx * img_w_ + x_idx]);
		pred_val.push_back(proj_output_[y_idx * img_w_ + x_idx + img_h_ * img_w_]);
		pred_val.push_back(proj_output_[y_idx * img_w_ + x_idx + img_h_ * img_w_ * 2]);

		float max_val = pred_val[0];
		int max_idx = 0;

		for (int i = 1; i < pred_val.size(); ++i) {
			if (pred_val[i] > max_val) {
				max_val = pred_val[i];
				max_idx = i;
			}
		}

		// pred_np_.push_back(max_idx);

		pred_np_[i] = max_idx;
	}

	// Always write KITTI-format predictions for evaluation.
	writeLabel(scans);

	// Optional range-view visualization (top: range image, bottom: moving mask).
	if (save_png_) {
	const int proj_h = img_h_;
	const int proj_w = 1024;
	const float proj_fov_up = proj_fov_up_;
	const float proj_fov_down = proj_fov_down_;
	const int power = 16;

	std::vector<float> proj_range;
	proj_range.resize(proj_h * proj_w);

	std::fill(proj_range.begin(), proj_range.end(), -1);

	std::vector<int> proj_idx;
	proj_idx.resize(proj_h * proj_w);
	std::fill(proj_idx.begin(), proj_idx.end(), -1);

	std::vector<std::pair<int, float>> depths;
	std::vector<int> proj_xs;
	std::vector<int> proj_ys;

	float fov_up = proj_fov_up / 180.0f * M_PI;
	float fov_down = proj_fov_down / 180.0f * M_PI;
	float fov = std::abs(fov_down) + std::abs(fov_up);

	for (int i = 0; i < scans.points.size(); ++i) {
		float depth = std::sqrt(scans.points[i].x * scans.points[i].x + scans.points[i].y * scans.points[i].y + scans.points[i].z * scans.points[i].z);
		float yaw = -std::atan2(scans.points[i].y, scans.points[i].x);
		float pitch = std::asin(scans.points[i].z / depth);

		int proj_x = std::floor((0.5 * (yaw / M_PI + 1.0)) * proj_w);
		int proj_y = std::floor((1.0 - (pitch + std::abs(fov_down)) / fov) * proj_h);

		proj_x = std::max(0, std::min(proj_w - 1, proj_x));
		proj_y = std::max(0, std::min(proj_h - 1, proj_y));

		depths.push_back({i, depth});
		proj_xs.push_back(proj_x);
		proj_ys.push_back(proj_y);
	}

	std::sort(depths.begin(), depths.end(), cmp);

	float min_depth = depths[0].second;

	for (int i = 0; i < depths.size(); ++i) {
		int order = depths[i].first;

		proj_idx[proj_ys[order] * proj_w + proj_xs[order]] = order;

		proj_range[proj_ys[order] * proj_w + proj_xs[order]] = depths[i].second;

		if (depths[i].second > 0) {
			proj_range[proj_ys[order] * proj_w + proj_xs[order]] = std::pow(depths[i].second, 1.0 / power);
		}

		if (proj_range[proj_ys[order] * proj_w + proj_xs[order]] > 0 && proj_range[proj_ys[order] * proj_w + proj_xs[order]] < min_depth) {
			min_depth = proj_range[proj_ys[order] * proj_w + proj_xs[order]];
		}
	}

	float max_depth = min_depth;

	for (int i = 0; i < proj_range.size(); ++i) {
		if (proj_range[i] < 0) {
			proj_range[i] = min_depth;
		}

		if (proj_range[i] >= max_depth) {
			max_depth = proj_range[i];
		}
	}

	// float scale_factor = (float)imgs[0].cols / 170;
	// int resize_h = (int)((float)imgs[0].rows / scale_factor);

	// for (int i = 0; i < 6; ++i) {
	// 	cv::resize(imgs[i], imgs[i], cv::Size(170, resize_h));
	// }

	cv::Mat result = cv::Mat::zeros(proj_h * 2, proj_w, CV_8UC3);
	// cv::Mat result = cv::Mat::zeros(proj_h * 2 + resize_h, proj_w, CV_8UC3);

	// imgs[0].copyTo(result.rowRange(0, resize_h).colRange(proj_w / 2 - 170 / 2, proj_w / 2 + 170 / 2));
	// imgs[1].copyTo(result.rowRange(0, resize_h).colRange(proj_w / 2 - 170 / 2 - 140, proj_w / 2 + 170 / 2 - 140));
	// imgs[2].copyTo(result.rowRange(0, resize_h).colRange(proj_w / 2 + 170 / 2 - 20, proj_w / 2 + 170 / 2 + 150));
	// imgs[4].copyTo(result.rowRange(0, resize_h).colRange(proj_w / 2 - 170 / 2 - 280, proj_w / 2 - 170 / 2 - 110));
	// imgs[5].copyTo(result.rowRange(0, resize_h).colRange(proj_w / 2 + 170 / 2 + 130, proj_w / 2 + 170 / 2 + 300));

	for (int i = 0; i < proj_h; ++i) {
		for (int j = 0; j < proj_w; ++j) {
			result.at<cv::Vec3b>(i, j)[0] = (unsigned char)(((proj_range[i * proj_w + j] - min_depth) / (max_depth - min_depth)) * 255);
			result.at<cv::Vec3b>(i, j)[1] = (unsigned char)(((proj_range[i * proj_w + j] - min_depth) / (max_depth - min_depth)) * 255);
			result.at<cv::Vec3b>(i, j)[2] = (unsigned char)(((proj_range[i * proj_w + j] - min_depth) / (max_depth - min_depth)) * 255);

			if (proj_idx[i * proj_w + j] >= 0 && pred_np_[proj_idx[i * proj_w + j]] == 2) {
				result.at<cv::Vec3b>(i + proj_h, j)[0] = 0;
				result.at<cv::Vec3b>(i + proj_h, j)[1] = 0;
				result.at<cv::Vec3b>(i + proj_h, j)[2] = 255;
			}
		}
	}

	// for (int i = 0; i < proj_h; ++i) {
	// 	for (int j = 0; j < proj_w; ++j) {
	// 		result.at<cv::Vec3b>(i + resize_h, j)[0] = (unsigned char)(((proj_range[i * proj_w + j] - min_depth) / (max_depth - min_depth)) * 255);
	// 		result.at<cv::Vec3b>(i + resize_h, j)[1] = (unsigned char)(((proj_range[i * proj_w + j] - min_depth) / (max_depth - min_depth)) * 255);
	// 		result.at<cv::Vec3b>(i + resize_h, j)[2] = (unsigned char)(((proj_range[i * proj_w + j] - min_depth) / (max_depth - min_depth)) * 255);

	// 		if (proj_idx[i * proj_w + j] >= 0 && pred_np_[proj_idx[i * proj_w + j]] == 2) {
	// 			result.at<cv::Vec3b>(i + proj_h + resize_h, j)[0] = 0;
	// 			result.at<cv::Vec3b>(i + proj_h + resize_h, j)[1] = 0;
	// 			result.at<cv::Vec3b>(i + proj_h + resize_h, j)[2] = 255;
	// 		}
	// 	}
	// }

	std::stringstream ss;
	ss << png_dir_ << "/" << std::setw(6) << std::setfill('0') << frame_idx_ << ".png";

	cv::imwrite(ss.str(), result);
	}  // if (save_png_)

	++frame_idx_;
}

template<typename PointT>
// void TRTLMNet<PointT>::inference(std::vector<PointT> scans, Eigen::Matrix4d pose) {
void TRTLMNet<PointT>::inference(pcl::PointCloud<PointT>& scans, Eigen::Matrix4d pose) {
// void TRTLMNet::inference(std::vector<pcl::PointXYZI> scans, Eigen::Matrix4d pose, cv::Mat img)
// void TRTLMNet::inference(std::vector<pcl::PointXYZI> scans, Eigen::Matrix4d pose, std::vector<cv::Mat> imgs) {
	if (proj_output_.empty()) {
		proj_output_.resize(dim_out_[0].d[0] * dim_out_[0].d[1] * dim_out_[0].d[2] * dim_out_[0].d[3]);
	}

	// Frame 0 has no previous scan, so last_scans_ is empty and the residual
	// channel is all zeros -- matching the Python pipeline's first frame.
	if (!preprocess(scans, pose)) {
		return;
	}

	cuda_buff_[0] = cuda_in_[0];
	cuda_buff_[1] = cuda_out_[0];

	// Input Host to Device Copy
	int ch_len = dim_in_[0].d[2] * dim_in_[0].d[3];  // * sizeof(float);

	// Diagnostic: dump the assembled 6-channel network input for one frame so it
	// can be compared against the Python parser. Enable with LMNET_DUMP=<frame>.
	if (const char* dump_env = std::getenv("LMNET_DUMP")) {
		if (frame_idx_ == std::atoi(dump_env)) {
			std::ofstream dbg("work_dir/cpp_input_f" + std::to_string(frame_idx_) + ".bin", std::ios::binary);
			for (const std::vector<float>* ch : {&proj_range_, &proj_x_, &proj_y_, &proj_z_, &proj_remission_, &diff_image_}) {
				dbg.write(reinterpret_cast<const char*>(ch->data()), ch_len * sizeof(float));
			}
			dbg.close();
			std::cout << "Dumped cpp_input_f" << frame_idx_ << ".bin" << std::endl;
		}
	}

	cudaMemcpyAsync((float*)cuda_in_[0], proj_range_.data(), ch_len * sizeof(float), cudaMemcpyHostToDevice, stream_);
	cudaMemcpyAsync((float*)cuda_in_[0] + ch_len, proj_x_.data(), ch_len * sizeof(float), cudaMemcpyHostToDevice, stream_);
	cudaMemcpyAsync((float*)cuda_in_[0] + ch_len * 2, proj_y_.data(), ch_len * sizeof(float), cudaMemcpyHostToDevice, stream_);
	cudaMemcpyAsync((float*)cuda_in_[0] + ch_len * 3, proj_z_.data(), ch_len * sizeof(float), cudaMemcpyHostToDevice, stream_);
	cudaMemcpyAsync((float*)cuda_in_[0] + ch_len * 4, proj_remission_.data(), ch_len * sizeof(float), cudaMemcpyHostToDevice, stream_);
	cudaMemcpyAsync((float*)cuda_in_[0] + ch_len * 5, diff_image_.data(), ch_len * sizeof(float), cudaMemcpyHostToDevice, stream_);

	// Inference
	context_->enqueueV2(cuda_buff_.data(), stream_, nullptr);

	// Copy Output Deivce to Host
	cudaMemcpyAsync(proj_output_.data(), (float*)cuda_out_[0], size_out_[0], cudaMemcpyDeviceToHost, stream_);

	cudaStreamSynchronize(stream_);

	postprocess(scans);
	// postprocess(scans, img);
	// postprocess(scans, imgs);

	last_pose_ = pose;
	last_scans_ = scans;
}