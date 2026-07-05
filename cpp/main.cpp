// LiDAR-MOS TensorRT C++ inference on the KITTI-Odometry toy dataset (seq 08).
//
// Usage:
//   lmnet [seq_dir] [cfg_path] [out_label_dir] [png_dir]
//
// Defaults (run from the repository root):
//   seq_dir       = data/sequences/08
//   cfg_path      = cpp/cfg/lmnet.yaml
//   out_label_dir = data/predictions_cpp_trt/sequences/08/predictions
//   png_dir       = "" (empty -> no PNGs; pass a dir to also dump range-view PNGs)
//
// Predictions are written as KITTI .label files (uint32 per point: 9 = static,
// 251 = moving) so they can be scored with utils/evaluate_mos.py.

#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>

#include "trt_lmnet.h"

void readFileLists(const std::string& dir_path, std::vector<std::string>& out_filelists, const std::string& type)
{
    DIR* dir = opendir(dir_path.c_str());

    if (dir == nullptr) {
        std::cerr << "Could not open directory: " << dir_path << std::endl;
        exit(EXIT_FAILURE);
    }

    struct dirent* ptr;

    while ((ptr = readdir(dir)) != nullptr) {
        std::string file_path = ptr->d_name;

        if (file_path[0] == '.') {
            continue;
        }

        if (type.size() <= 0) {
            out_filelists.push_back(ptr->d_name);
        } else {
            if (file_path.size() < type.size()) {
                continue;
            }

            std::string file_type = file_path.substr(file_path.size() - type.size(), type.size());

            if (file_type == type) {
                out_filelists.push_back(ptr->d_name);
            }
        }
    }

    closedir(dir);
}

void sortFileLists(std::vector<std::string>& file_lists)
{
    std::sort(file_lists.begin(), file_lists.end());
}

// Read a KITTI velodyne .bin file (x, y, z, intensity as float32) into a cloud.
void readKITTIData(const std::string& file_path, pcl::PointCloud<pcl::PointXYZI>::Ptr& scans)
{
    std::fstream f_in(file_path.c_str(), std::ios::in | std::ios::binary);

    if (!f_in.good()) {
        std::cerr << "Could not read file: " << file_path << std::endl;
        exit(EXIT_FAILURE);
    }

    pcl::PointXYZI point;

    // Guard both reads so the trailing EOF read does not push a bogus point.
    while (f_in.read((char*)&point.x, 3 * sizeof(float)) &&
           f_in.read((char*)&point.intensity, sizeof(float))) {
        scans->points.push_back(point);
    }

    f_in.close();
}

// Load KITTI camera-frame poses (T_w_cam0) and convert them to the LiDAR frame
// relative to the first frame, matching utils/gen_residual_images.py:
//   new_pose = inv(T_cam_velo) * inv(pose_0) * pose_i * T_cam_velo
void loadPoses(const std::string& pose_path, const Eigen::Matrix4d& calib, std::vector<Eigen::Matrix4d>& poses)
{
    std::fstream f_in(pose_path.c_str(), std::ios::in);

    if (!f_in.good()) {
        std::cerr << "Could not read file: " << pose_path << std::endl;
        exit(EXIT_FAILURE);
    }

    const Eigen::Matrix4d velo_to_cam = calib.inverse();
    Eigen::Matrix4d inv_frame0 = Eigen::Matrix4d::Identity();

    bool init_pose = false;

    double f_val[12];

    while (f_in >> f_val[0]) {
        for (int i = 1; i < 12; ++i) {
            f_in >> f_val[i];
        }

        Eigen::Matrix4d pose;
        pose << f_val[0], f_val[1], f_val[2],  f_val[3],
                f_val[4], f_val[5], f_val[6],  f_val[7],
                f_val[8], f_val[9], f_val[10], f_val[11],
                0.0,      0.0,      0.0,       1.0;

        if (!init_pose) {
            inv_frame0 = pose.inverse();
            init_pose = true;
        }

        poses.push_back(velo_to_cam * inv_frame0 * pose * calib);
    }

    f_in.close();
}

// Load the KITTI calibration Tr (T_cam_velo). The odometry calib.txt lists
// P0..P3 then "Tr: ..." on the last line; we keep the last 3x4 matrix read.
void loadCalib(const std::string& file_path, Eigen::Matrix4d& calib)
{
    std::fstream f_in(file_path.c_str(), std::ios::in);

    if (!f_in.good()) {
        std::cerr << "Could not read file: " << file_path << std::endl;
        exit(EXIT_FAILURE);
    }

    double f_val[12] = {0};

    std::string line;
    while (std::getline(f_in, line)) {
        if (line.empty()) {
            continue;
        }

        // Drop the leading token (e.g. "Tr:" or "P0:") and parse 12 doubles.
        std::size_t colon = line.find(':');
        std::string body = (colon == std::string::npos) ? line : line.substr(colon + 1);

        std::stringstream ss(body);
        double vals[12];
        int n = 0;
        while (n < 12 && (ss >> vals[n])) {
            ++n;
        }

        if (n == 12) {
            for (int i = 0; i < 12; ++i) {
                f_val[i] = vals[i];
            }
        }
    }

    calib << f_val[0], f_val[1], f_val[2],  f_val[3],
             f_val[4], f_val[5], f_val[6],  f_val[7],
             f_val[8], f_val[9], f_val[10], f_val[11],
             0.0,      0.0,      0.0,       1.0;

    f_in.close();
}

int main(int argc, char** argv)
{
    std::string seq_dir       = (argc > 1) ? argv[1] : "data/sequences/08";
    std::string cfg_path      = (argc > 2) ? argv[2] : "cpp/cfg/lmnet.yaml";
    std::string out_label_dir = (argc > 3) ? argv[3] : "data/predictions_cpp_trt/sequences/08/predictions";
    std::string png_dir       = (argc > 4) ? argv[4] : "";

    const bool save_png = !png_dir.empty();

    std::string bin_folder_path = seq_dir + "/velodyne";
    std::string pose_file_path  = seq_dir + "/poses.txt";
    std::string calib_file_path = seq_dir + "/calib.txt";

    std::vector<std::string> bin_file_list;
    readFileLists(bin_folder_path, bin_file_list, "bin");
    sortFileLists(bin_file_list);

    Eigen::Matrix4d calib;
    loadCalib(calib_file_path, calib);

    std::vector<Eigen::Matrix4d> poses;
    loadPoses(pose_file_path, calib, poses);

    std::cout << "Frames: " << bin_file_list.size() << ", poses: " << poses.size() << std::endl;

    if (poses.size() < bin_file_list.size()) {
        std::cerr << "Not enough poses for the available scans." << std::endl;
        return EXIT_FAILURE;
    }

    // LMNet TensorRT
    auto trt_lmnet = std::make_shared<TRTLMNet<pcl::PointXYZI>>();

    trt_lmnet->setConfig(cfg_path.c_str());

    if (!trt_lmnet->loadEngine()) {
        std::cout << "No cached engine; building from ONNX..." << std::endl;
        trt_lmnet->buildEngine();
    }

    trt_lmnet->setOutput(out_label_dir, png_dir, save_png);

    double total_ms = 0.0;

    for (std::size_t i = 0; i < bin_file_list.size(); ++i) {
        std::string bin_file_path = bin_folder_path + "/" + bin_file_list[i];

        pcl::PointCloud<pcl::PointXYZI>::Ptr scans = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        readKITTIData(bin_file_path, scans);

        auto start = std::chrono::steady_clock::now();
        trt_lmnet->inference(*scans, poses[i]);
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start).count() / 1000.0;
        total_ms += ms;

        if (i % 200 == 0) {
            std::cout << "[" << i << "/" << bin_file_list.size() << "] "
                      << bin_file_list[i] << "  " << ms << " ms" << std::endl;
        }
    }

    std::cout << "Done. Mean inference time: " << (total_ms / bin_file_list.size())
              << " ms over " << bin_file_list.size() << " frames." << std::endl;
    std::cout << "Predictions written to: " << out_label_dir << std::endl;

    return 0;
}
