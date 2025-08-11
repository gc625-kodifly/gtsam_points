/* ──────────────────────────────────────────────────────────────────── *\
 *  MatchingCostFactorDemo.cpp  (dynamic N, root dir via argv)        *
\* ──────────────────────────────────────────────────────────────────── */

#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/format.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_points/config.hpp>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>

#ifdef GTSAM_POINTS_USE_CUDA
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>
#endif

#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/optimizers/isam2_ext.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>

#include <glk/thin_lines.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

/* =================================================================== */
class MatchingCostFactorDemo {
public:
  explicit MatchingCostFactorDemo(const std::string& root_dir)
      : data_path(root_dir) {

    auto viewer = guik::LightViewer::instance();
    viewer->enable_vsync();

    /* -------- read graph ---------------------------------------------------- */
    std::ifstream ifs(data_path + "/graph.txt");
    if (!ifs) {
      std::cerr << "error: failed to open " << data_path + "/graph.txt" << std::endl;
      std::exit(1);
    }

    std::vector<std::string> graph_lines;
    std::string line;
    while (std::getline(ifs, line)) {
      if (!line.empty()) graph_lines.push_back(line);
    }

    num_frames = graph_lines.size();
    if (num_frames == 0) {
      std::cerr << "error: graph.txt has no lines" << std::endl;
      std::exit(1);
    }
    std::cout << "[INFO] found " << num_frames << " frames" << std::endl;

#ifdef GTSAM_POINTS_USE_CUDA
    std::cout << "Register GPU linearization hook" << std::endl;
    gtsam_points::LinearizationHook::register_hook(
        [] { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif

    /* -------- containers ---------------------------------------------------- */
    frames.resize(num_frames);
    voxelmaps.resize(num_frames);
    voxelmaps_gpu.resize(num_frames);
    ids.resize(num_frames);

    /* -------- load every frame --------------------------------------------- */
    for (size_t i = 0; i < num_frames; ++i) {
      /* parse id and pose -------------------------------------------------- */
      std::istringstream iss(graph_lines[i]);
      std::string id;            // e.g. CWLPH_2
      gtsam::Vector3 t;
      gtsam::Quaternion q;
      iss >> id >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >> q.w();

      ids[i] = id;

      gtsam::Pose3 P(gtsam::Rot3(q), t);
      poses.insert(i, P);
      poses_gt.insert(i, P);

      /* points file path ---------------------------------------------------- */
      std::string points_path = data_path + "/" + id + "/" + id + ".bin";
      std::ifstream test(points_path, std::ios::binary);
      if (!test) {
        points_path = data_path + "/" + id + "/points.bin";
        test.open(points_path, std::ios::binary);
      }
      if (!test) {
        std::cerr << "error: failed to read points for " << id << std::endl;
        std::exit(1);
      }

      std::cout << "loading " << points_path << std::endl;
      auto pts_f = gtsam_points::read_points(points_path);
      std::cout << "  -> loaded " << pts_f.size() << " points" << std::endl;
      if (!pts_f.empty())
        std::cout << "     first point: " << pts_f[0].transpose() << std::endl;

      std::vector<Eigen::Vector4d> pts_d(pts_f.size());
      std::transform(pts_f.begin(), pts_f.end(), pts_d.begin(),
                     [](const Eigen::Vector3f& p) {
                       return (Eigen::Vector4d() << p.cast<double>(), 1.0).finished();
                     });

      auto covs = gtsam_points::estimate_covariances(pts_d);

#ifndef GTSAM_POINTS_USE_CUDA
      auto frame = std::make_shared<gtsam_points::PointCloudCPU>();
#else
      auto frame = std::make_shared<gtsam_points::PointCloudGPU>();
#endif
      frame->add_points(pts_d);
      frame->add_covs(covs);
      frame->add_normals(
          gtsam_points::estimate_normals(frame->points, frame->size()));
      frames[i] = frame;

      auto vmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(2.0);
      vmap->insert(*frame);
      voxelmaps[i] = vmap;

#ifdef GTSAM_POINTS_USE_CUDA
      auto vmap_gpu = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(2.0);
      vmap_gpu->insert(*frame);
      voxelmaps_gpu[i] = vmap_gpu;
#endif

      viewer->update_drawable(
          "frame_" + id,
          std::make_shared<glk::PointCloudBuffer>(frame->points, frame->size()),
          guik::Rainbow());
    }
    update_viewer(poses);

    /* --------------- UI defaults & enums ----------------------------------- */
    pose_noise_scale = 0.1f;
    optimizer_types  = {"LM", "ISAM2"};
    factor_types     = {"ICP", "ICP_PLANE", "GICP", "VGICP"
#ifdef GTSAM_POINTS_USE_CUDA
                        , "VGICP_GPU"
#endif
    };
    optimizer_type = 0;
    factor_type    = 0;
    full_connection = true;
    num_threads     = 1;
    correspondence_update_tolerance_rot   = 0.0f;
    correspondence_update_tolerance_trans = 0.0f;

    /* --------------- UI panel ---------------------------------------------- */
    viewer->register_ui_callback("control", [this] {
      ImGui::DragFloat("noise_scale", &pose_noise_scale, 0.01f, 0.0f);
      if (ImGui::Button("add noise")) {
        for (size_t i = 1; i < num_frames; ++i) {
          gtsam::Pose3 n = gtsam::Pose3::Expmap(gtsam::Vector6::Random() * pose_noise_scale);
          poses.update<gtsam::Pose3>(i, poses_gt.at<gtsam::Pose3>(i) * n);
        }
        update_viewer(poses);
      }

      ImGui::Separator();
      ImGui::Checkbox("full connection", &full_connection);
      ImGui::DragInt("num threads", &num_threads, 1, 1, 128);
      ImGui::Combo("factor type",  &factor_type,
                   factor_types.data(), factor_types.size());
      ImGui::Combo("optimizer type", &optimizer_type,
                   optimizer_types.data(), optimizer_types.size());
      ImGui::DragFloat("corr tol rot",
                       &correspondence_update_tolerance_rot,
                       0.001f, 0.f, 0.1f);
      ImGui::DragFloat("corr tol trans",
                       &correspondence_update_tolerance_trans,
                       0.01f, 0.f, 1.f);

      if (ImGui::Button("optimize")) {
        if (optimization_thread.joinable()) optimization_thread.join();
        optimization_thread = std::thread([this] { run_optimization(); });
      }
    });
  }

  ~MatchingCostFactorDemo() {
    if (optimization_thread.joinable()) optimization_thread.join();
  }

private:
  /* --------------------- helpers ------------------------------------------- */
  void save_values(const gtsam::Values& vals, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) { std::cerr << "cannot open " << path << '\n'; return; }
    for (size_t i = 0; i < num_frames; ++i) {
      const auto& P = vals.at<gtsam::Pose3>(i);
      const auto& t = P.translation();
      gtsam::Quaternion q = P.rotation().toQuaternion();
      ofs << ids[i] << ' '
          << t.x() << ' ' << t.y() << ' ' << t.z() << ' '
          << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
    std::cout << "[INFO] wrote " << path << '\n';
  }

  void update_viewer(const gtsam::Values& vals) {
    guik::LightViewer::instance()->invoke([=] {
      auto vwr = guik::LightViewer::instance();

      std::vector<Eigen::Vector3f> lines;
      for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Isometry3f pose(vals.at<gtsam::Pose3>(i).matrix().cast<float>());

        const std::string& id = ids[i];

        vwr->find_drawable("frame_" + id)
            .first->add("model_matrix", pose);
        vwr->update_drawable(
            "coord_" + id,
            glk::Primitives::coordinate_system(),
            guik::VertexColor(pose * Eigen::UniformScaling<float>(5.0f)));

        size_t j_end = full_connection ? num_frames
                                       : std::min(i + 2, num_frames);
        for (size_t j = i + 1; j < j_end; ++j) {
          lines.push_back(
              vals.at<gtsam::Pose3>(i).translation().cast<float>());
          lines.push_back(
              vals.at<gtsam::Pose3>(j).translation().cast<float>());
        }
      }
      vwr->update_drawable(
          "factors",
          std::make_shared<glk::ThinLines>(lines),
          guik::FlatColor(0.f, 1.f, 0.f, 1.f));
    });
  }

  gtsam::NonlinearFactor::shared_ptr create_factor(
      gtsam::Key tgt, gtsam::Key src,
      const gtsam_points::PointCloud::ConstPtr& tgt_pc,
      const gtsam_points::GaussianVoxelMap::ConstPtr& tgt_vm,
      const gtsam_points::GaussianVoxelMap::ConstPtr& tgt_vm_gpu,
      const gtsam_points::PointCloud::ConstPtr& src_pc) {

    const std::string ft = factor_types[factor_type];
    if (ft == "ICP") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(
          tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(
          correspondence_update_tolerance_rot,
          correspondence_update_tolerance_trans);
      f->set_num_threads(num_threads);
      return f;
    }
    if (ft == "ICP_PLANE") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedPointToPlaneICPFactor>(
          tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(
          correspondence_update_tolerance_rot,
          correspondence_update_tolerance_trans);
      f->set_num_threads(num_threads);
      return f;
    }
    if (ft == "GICP") {
      auto f =
          gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(tgt, src,
                                                                 tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(
          correspondence_update_tolerance_rot,
          correspondence_update_tolerance_trans);
      f->set_num_threads(num_threads);
      return f;
    }
    if (ft == "VGICP") {
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(
          tgt, src, tgt_vm, src_pc);
    }
    if (ft == "VGICP_GPU") {
#ifdef GTSAM_POINTS_USE_CUDA
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(
          tgt, src, tgt_vm_gpu, src_pc);
#endif
    }
    std::cerr << "unknown factor type " << ft << '\n';
    return nullptr;
  }

  /* --------------------- optimisation --------------------------------------- */
  void run_optimization() {
    gtsam::NonlinearFactorGraph graph;
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        0, poses.at<gtsam::Pose3>(0),
        gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

    for (size_t i = 0; i < num_frames; ++i) {
      size_t j_end = full_connection ? num_frames
                                     : std::min(i + 2, num_frames);
      for (size_t j = i + 1; j < j_end; ++j)
        graph.add(create_factor(i, j, frames[i], voxelmaps[i],
                                voxelmaps_gpu[i], frames[j]));
    }

      /* ---------- LM --------------------------------------------------------- */
    if (optimizer_types[optimizer_type] == std::string("LM")) {
      gtsam_points::LevenbergMarquardtExtParams prm;
      prm.maxIterations = 300; 
      prm.callback = [this](auto& st, const gtsam::Values& v) {
        guik::LightViewer::instance()->append_text(st.to_string());
        update_viewer(v);
      };
      gtsam_points::LevenbergMarquardtOptimizerExt opt(graph, poses, prm);
      opt.optimize();

      //  ✨ write to <root>/optimized.txt instead of CWD
      save_values(opt.values(), data_path + "/optimized.txt");
    }

    /* ---------- ISAM2 ------------------------------------------------------ */
    else {
      gtsam::ISAM2Params p;
      p.relinearizeSkip = 1;
      p.setRelinearizeThreshold(0.f);
      gtsam_points::ISAM2Ext isam2(p);

      isam2.update(graph, poses);
      update_viewer(isam2.calculateEstimate());
      for (size_t i = 0; i < num_frames; ++i) {
        isam2.update();
        update_viewer(isam2.calculateEstimate());
      }

      //  ✨ write to <root>/optimized.txt instead of CWD
      save_values(isam2.calculateEstimate(), data_path + "/optimized.txt");
    } 
  }

  /* ----------------- data --------------------------------------------------- */
  std::string data_path;
  size_t      num_frames{0};

  float pose_noise_scale;

  std::vector<const char*> optimizer_types;
  std::vector<const char*> factor_types;
  int  optimizer_type, factor_type;
  bool full_connection;
  int  num_threads;
  float correspondence_update_tolerance_rot;
  float correspondence_update_tolerance_trans;

  std::thread optimization_thread;

  gtsam::Values poses, poses_gt;
  std::vector<gtsam_points::PointCloud::Ptr>       frames;
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps, voxelmaps_gpu;
  std::vector<std::string> ids;  // frame identifiers (e.g., CWLPH_3)
};

/* =================================================================== */
int main(int argc, char** argv) {
  std::string root = (argc > 1) ? argv[1] : "data/preprocessed_dir";
  MatchingCostFactorDemo demo(root);
  guik::LightViewer::instance()->spin();
  return 0;
}

/* =================================================================== */
