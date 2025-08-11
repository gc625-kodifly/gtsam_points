/* ──────────────────────────────────────────────────────────────────── *\
 *  MatchingCostFactorDemo (GUI + auto-run + parallel loading)         *
\* ──────────────────────────────────────────────────────────────────── */

#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <mutex>
#include <cctype>
#include <limits>

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

// Viewer (skipped in headless mode)
#include <glk/thin_lines.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

// LASlib + small_gicp
#include <LASlib/lasreader.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

struct Args {
  std::string root;                     // positional
  std::string file_format = "bin";
  std::string points_name = "points";
  double      downsample_resolution = 0.25;
  int         downsample_threads    = 4;
  std::string optimizer_type = "LM";
  std::string factor_type    = "GICP";
  bool        headless       = false;
  int         load_threads   = 0;
  std::string graph_path;               // NEW: optional path to graph.txt
};

static inline std::string lower_copy(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

static bool parse_cli(int argc, char** argv, Args& out) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <root_dir>"
              << " [--file_format las|laz|bin]"
              << " [--points_name NAME]"
              << " [--downsample_resolution 0.25]"
              << " [--downsample_threads 4]"
              << " [--optimizer_type LM|ISAM2]"
              << " [--factor_type ICP|ICP_PLANE|GICP|VGICP|VGICP_GPU]"
              << " [--load_threads N]"
              << " [--graph /abs/or/rel/path/to/graph.txt]"   // << added
              << " [--headless]\n";
    return false;
  }
  out.root = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string key = argv[i];
    if (key == std::string("--headless")) { out.headless = true; continue; }
    if (key.rfind("--", 0) == 0 && i + 1 < argc) {
      key = key.substr(2);
      std::string val = argv[++i];

      if (key == "file_format") out.file_format = lower_copy(val);
      else if (key == "points_name") out.points_name = val;
      else if (key == "downsample_resolution") out.downsample_resolution = std::stod(val);
      else if (key == "downsample_threads") out.downsample_threads = std::stoi(val);
      else if (key == "optimizer_type") out.optimizer_type = val;
      else if (key == "factor_type") out.factor_type = val;
      else if (key == "load_threads") out.load_threads = std::stoi(val);
      else if (key == "graph") out.graph_path = val;                   // << added
      else std::cerr << "[WARN] unknown flag --" << key << '\n';
    }
  }

  out.optimizer_type = lower_copy(out.optimizer_type);
  out.factor_type    = lower_copy(out.factor_type);
  if (out.file_format != "las" && out.file_format != "laz" && out.file_format != "bin") {
    std::cerr << "[WARN] file_format '" << out.file_format << "' not recognized; using 'bin'\n";
    out.file_format = "bin";
  }
  if (out.load_threads <= 0) {
    out.load_threads = out.downsample_threads > 0 ? out.downsample_threads
                                                  : (int)std::thread::hardware_concurrency();
    if (out.load_threads <= 0) out.load_threads = 4;
  }

  // Default graph path if not supplied
  if (out.graph_path.empty()) out.graph_path = out.root + "/graph.txt";  // << added
  return true;
}

/* ---------- IO helpers ---------------------------------------------------- */
static std::vector<Eigen::Vector3f> read_points_from_las(const std::string& path) {
  LASreadOpener opener;
  opener.set_file_name(path.c_str());
  LASreader* reader = opener.open();
  if (!reader) {
    std::cerr << "error: failed to open " << path << " with LASlib\n";
    return {};
  }
  uint64_t n = (reader->header.version_minor >= 4 && reader->header.extended_number_of_point_records)
                ? reader->header.extended_number_of_point_records
                : reader->header.number_of_point_records;
  std::vector<Eigen::Vector3f> points;
  if (n > 0 && n < static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    points.reserve(static_cast<size_t>(n));

  while (reader->read_point()) {
    const LASpoint& p = reader->point;
    points.emplace_back(static_cast<float>(p.get_x()),
                        static_cast<float>(p.get_y()),
                        static_cast<float>(p.get_z()));
  }
  reader->close();
  delete reader;
  return points;
}

static std::string compose_points_path(const Args& a, const std::string& id) {
  // <root>/<id>/<points_name>.<file_format>
  std::ostringstream oss;
  oss << a.root << "/" << id << "/" << a.points_name << "." << a.file_format;
  std::string path = oss.str();

  if (a.file_format == "bin") {
    std::ifstream test(path, std::ios::binary);
    if (!test) {
      std::ostringstream alt;
      alt << a.root << "/" << id << ".bin";
      return alt.str();
    }
  }
  return path;
}

/* =================================================================== */
class MatchingCostFactorDemo {
public:
  explicit MatchingCostFactorDemo(const Args& a)
      : args(a) {

#ifdef GTSAM_POINTS_USE_CUDA
    gtsam_points::LinearizationHook::register_hook(
        [] { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif

    if (!args.headless) {
      auto viewer = guik::LightViewer::instance();
      viewer->enable_vsync();
    }

    /* -------- read graph (sequential) ------------------------------------- */
    std::ifstream ifs(args.graph_path);
    if (!ifs) {
    std::cerr << "error: failed to open " << args.graph_path << std::endl;
    std::exit(1);
    }
    std::cout << "[INFO] using graph: " << args.graph_path << std::endl;

    struct Line { std::string id; gtsam::Pose3 pose; };
    std::vector<Line> entries;
    {
      std::string line;
      while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string id;
        gtsam::Vector3 t;
        gtsam::Quaternion q;
        iss >> id >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >> q.w();
        entries.push_back(Line{id, gtsam::Pose3(gtsam::Rot3(q), t)});
      }
    }

    num_frames = entries.size();
    if (num_frames == 0) {
      std::cerr << "error: graph.txt has no lines" << std::endl;
      std::exit(1);
    }
    std::cout << "[INFO] found " << num_frames << " frames" << std::endl;

    /* -------- containers --------------------------------------------------- */
    frames.resize(num_frames);
    voxelmaps.resize(num_frames);
    voxelmaps_gpu.resize(num_frames);
    ids.resize(num_frames);
    init_poses.resize(num_frames);

    // Fill ids & initial poses (sequential)
    for (size_t i = 0; i < num_frames; ++i) {
      ids[i] = entries[i].id;
      init_poses[i] = entries[i].pose;
    }

    /* -------- parallel load ------------------------------------------------ */
    std::mutex print_mu;

    auto load_one = [&](size_t i) {
      const std::string& id = ids[i];
      const std::string points_path = compose_points_path(args, id);

      { std::lock_guard<std::mutex> lk(print_mu);
        std::cout << "loading " << points_path << std::endl;
      }

      std::vector<Eigen::Vector3f> pts_f;
      if (args.file_format == "las" || args.file_format == "laz") {
        pts_f = read_points_from_las(points_path);
      } else {
        pts_f = gtsam_points::read_points(points_path);
      }

      {
        std::lock_guard<std::mutex> lk(print_mu);
        std::cout << "  -> loaded " << pts_f.size() << " points" << std::endl;
      }
      if (pts_f.empty()) {
        std::lock_guard<std::mutex> lk(print_mu);
        std::cerr << "error: failed to read " << points_path << std::endl;
        return; // leave this index empty; optimization will likely fail
      }

      // Downsample (optional)
      std::shared_ptr<small_gicp::PointCloud> cloud(new small_gicp::PointCloud(pts_f));
      if (args.downsample_resolution > 0.0) {
        cloud = small_gicp::voxelgrid_sampling_omp(*cloud, args.downsample_resolution, args.downsample_threads);
        std::lock_guard<std::mutex> lk(print_mu);
        std::cout << "  -> after downsampling: " << cloud->size() << " points" << std::endl;
      }

      // Convert to Eigen::Vector4d (x,y,z,1) and build gtsam_points objects
      const std::vector<Eigen::Vector4d>& pts_d = cloud->points;

#ifndef GTSAM_POINTS_USE_CUDA
      auto frame = std::make_shared<gtsam_points::PointCloudCPU>();
#else
      auto frame = std::make_shared<gtsam_points::PointCloudGPU>();
#endif
      frame->add_points(pts_d);
      frame->add_covs(gtsam_points::estimate_covariances(pts_d));
      frame->add_normals(gtsam_points::estimate_normals(frame->points, frame->size()));
      frames[i] = frame;

      auto vmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(2.0);
      vmap->insert(*frame);
      voxelmaps[i] = vmap;

#ifdef GTSAM_POINTS_USE_CUDA
      auto vmap_gpu = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(2.0);
      vmap_gpu->insert(*frame);
      voxelmaps_gpu[i] = vmap_gpu;
#endif
    };

    // OpenMP parallel for (if available)
#if defined(_OPENMP)
    #pragma omp parallel for num_threads(args.load_threads) schedule(dynamic)
    for (int i = 0; i < static_cast<int>(num_frames); ++i) {
      load_one(static_cast<size_t>(i));
    }
#else
    // Fallback: tiny thread pool splitting the range
    {
      int T = std::max(1, args.load_threads);
      std::vector<std::thread> workers;
      workers.reserve(T);
      auto chunk = (num_frames + (size_t)T - 1) / (size_t)T;
      for (int t = 0; t < T; ++t) {
        size_t begin = (size_t)t * chunk;
        size_t end   = std::min(begin + chunk, num_frames);
        if (begin >= end) break;
        workers.emplace_back([&, begin, end] {
          for (size_t i = begin; i < end; ++i) load_one(i);
        });
      }
      for (auto& th : workers) th.join();
    }
#endif

    // Insert poses into gtsam::Values (sequential; Values is not thread-safe)
    for (size_t i = 0; i < num_frames; ++i) {
      poses.insert(i, init_poses[i]);
      poses_gt.insert(i, init_poses[i]);
    }

    // Add viewer drawables after threads complete
    if (!args.headless) {
      auto viewer = guik::LightViewer::instance();
      for (size_t i = 0; i < num_frames; ++i) {
        if (!frames[i]) continue;
        viewer->update_drawable(
          "frame_" + ids[i],
          std::make_shared<glk::PointCloudBuffer>(frames[i]->points, frames[i]->size()),
          guik::Rainbow());
      }
      update_viewer(poses);
    }

    /* --------------- enums/indexing ---------------------------------------- */
    optimizer_types  = {"LM", "ISAM2"};
    factor_types     = {"ICP", "ICP_PLANE", "GICP", "VGICP"
#ifdef GTSAM_POINTS_USE_CUDA
                        , "VGICP_GPU"
#endif
    };
    optimizer_type = index_of(optimizer_types, args.optimizer_type);
    factor_type    = index_of(factor_types, args.factor_type);
#ifndef GTSAM_POINTS_USE_CUDA
    if (factor_types[factor_type] == std::string("VGICP_GPU")) {
      std::cerr << "[WARN] VGICP_GPU requested but CUDA not enabled; falling back to VGICP\n";
      factor_type = index_of(factor_types, "VGICP");
    }
#endif
    full_connection = true;
    num_threads     = std::max(1, args.downsample_threads);
    correspondence_update_tolerance_rot   = 0.0f;
    correspondence_update_tolerance_trans = 0.0f;

    /* ---------- auto-run optimization -------------------------------------- */
    if (!args.headless) {
      if (optimization_thread.joinable()) optimization_thread.join();
      optimization_thread = std::thread([this] { run_optimization(); });
    } else {
      run_optimization();
    }

    // Optional small UI to re-run (unchanged)
    if (!args.headless) {
      guik::LightViewer::instance()->register_ui_callback("control", [this] {
        ImGui::TextUnformatted("MatchingCostFactorDemo");
        ImGui::Separator();
        ImGui::Text("root: %s", args.root.c_str());
        ImGui::Text("src:  %s.%s", args.points_name.c_str(), args.file_format.c_str());
        ImGui::Text("ds:   res=%.3f, threads=%d", args.downsample_resolution, args.downsample_threads);

        ImGui::Checkbox("full connection", &full_connection);
        ImGui::DragInt("num threads", &num_threads, 1, 1, 128);
        ImGui::Combo("factor type",  &factor_type,
                     vec_cstr(factor_types).data(), static_cast<int>(factor_types.size()));
        ImGui::Combo("optimizer type", &optimizer_type,
                     vec_cstr(optimizer_types).data(), static_cast<int>(optimizer_types.size()));
        ImGui::DragFloat("corr tol rot",
                         &correspondence_update_tolerance_rot,
                         0.001f, 0.f, 0.1f);
        ImGui::DragFloat("corr tol trans",
                         &correspondence_update_tolerance_trans,
                         0.01f, 0.f, 1.f);

        if (ImGui::Button("Re-run optimization")) {
          if (optimization_thread.joinable()) optimization_thread.join();
          optimization_thread = std::thread([this] { run_optimization(); });
        }
      });
    }
  }

  ~MatchingCostFactorDemo() {
    if (optimization_thread.joinable()) optimization_thread.join();
  }

private:
  static int index_of(const std::vector<std::string>& vec, const std::string& key) {
    auto key_l = lower_copy(key);
    for (size_t i = 0; i < vec.size(); ++i) {
      if (lower_copy(vec[i]) == key_l) return static_cast<int>(i);
    }
    return 0;
  }
  static std::vector<const char*> vec_cstr(const std::vector<std::string>& v) {
    std::vector<const char*> out; out.reserve(v.size());
    for (auto& s : v) out.push_back(s.c_str());
    return out;
  }

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
    if (args.headless) return;
    guik::LightViewer::instance()->invoke([=] {
      auto vwr = guik::LightViewer::instance();
      std::vector<Eigen::Vector3f> lines;
      for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Isometry3f pose(vals.at<gtsam::Pose3>(i).matrix().cast<float>());

        const std::string& id = ids[i];

        vwr->find_drawable("frame_" + id).first->add("model_matrix", pose);
        vwr->update_drawable(
            "coord_" + id,
            glk::Primitives::coordinate_system(),
            guik::VertexColor(pose * Eigen::UniformScaling<float>(5.0f)));

        size_t j_end = full_connection ? num_frames
                                       : std::min(i + 2, num_frames);
        for (size_t j = i + 1; j < j_end; ++j) {
          lines.push_back(vals.at<gtsam::Pose3>(i).translation().cast<float>());
          lines.push_back(vals.at<gtsam::Pose3>(j).translation().cast<float>());
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

    const std::string& ft = factor_types[factor_type];
    if (ft == "ICP") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(correspondence_update_tolerance_rot,
                                             correspondence_update_tolerance_trans);
      f->set_num_threads(num_threads);
      return f;
    }
    if (ft == "ICP_PLANE") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedPointToPlaneICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(correspondence_update_tolerance_rot,
                                             correspondence_update_tolerance_trans);
      f->set_num_threads(num_threads);
      return f;
    }
    if (ft == "GICP") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(correspondence_update_tolerance_rot,
                                             correspondence_update_tolerance_trans);
      f->set_num_threads(num_threads);
      return f;
    }
    if (ft == "VGICP") {
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(tgt, src, tgt_vm, src_pc);
    }
    if (ft == "VGICP_GPU") {
#ifdef GTSAM_POINTS_USE_CUDA
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(tgt, src, tgt_vm_gpu, src_pc);
#else
      std::cerr << "VGICP_GPU requested but CUDA build not enabled.\n";
      return nullptr;
#endif
    }
    std::cerr << "unknown factor type " << ft << '\n';
    return nullptr;
  }

  void run_optimization() {
    gtsam::NonlinearFactorGraph graph;
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        0, poses.at<gtsam::Pose3>(0),
        gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

    for (size_t i = 0; i < num_frames; ++i) {
      size_t j_end = full_connection ? num_frames
                                     : std::min(i + 2, num_frames);
      for (size_t j = i + 1; j < j_end; ++j) {
        auto f = create_factor(i, j, frames[i], voxelmaps[i], voxelmaps_gpu[i], frames[j]);
        if (f) graph.add(f);
      }
    }

    if (optimizer_types[optimizer_type] == std::string("LM")) {
      gtsam_points::LevenbergMarquardtExtParams prm;
      prm.maxIterations = 300;
      prm.callback = [this](auto& st, const gtsam::Values& v) {
        if (!args.headless) {
          guik::LightViewer::instance()->append_text(st.to_string());
          update_viewer(v);
        } else {
          std::cout << st.to_string() << std::endl;
        }
      };
      gtsam_points::LevenbergMarquardtOptimizerExt opt(graph, poses, prm);
      opt.optimize();
      save_values(opt.values(), args.root + "/optimized.txt");
    } else {
      gtsam::ISAM2Params p;
      p.relinearizeSkip = 1;
      p.setRelinearizeThreshold(0.f);
      gtsam_points::ISAM2Ext isam2(p);

      isam2.update(graph, poses);
      if (!args.headless) update_viewer(isam2.calculateEstimate());
      for (size_t i = 0; i < num_frames; ++i) {
        isam2.update();
        if (!args.headless) update_viewer(isam2.calculateEstimate());
      }
      save_values(isam2.calculateEstimate(), args.root + "/optimized.txt");
    }
  }

private:
  Args args;
  size_t      num_frames{0};

  std::vector<std::string> optimizer_types;
  std::vector<std::string> factor_types;
  int  optimizer_type{0}, factor_type{0};
  bool full_connection{true};
  int  num_threads{1};
  float correspondence_update_tolerance_rot{0.0f};
  float correspondence_update_tolerance_trans{0.0f};

  std::thread optimization_thread;

  std::vector<std::string> ids;
  std::vector<gtsam::Pose3> init_poses;

  gtsam::Values poses, poses_gt;
  std::vector<gtsam_points::PointCloud::Ptr>       frames;
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps, voxelmaps_gpu;
};

/* =================================================================== */
int main(int argc, char** argv) {
  Args args;
  if (!parse_cli(argc, argv, args)) return 1;

  MatchingCostFactorDemo demo(args);

  if (!args.headless) {
    guik::LightViewer::instance()->spin();
  }
  return 0;
}
