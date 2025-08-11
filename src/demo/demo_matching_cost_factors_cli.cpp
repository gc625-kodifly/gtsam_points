#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <limits>
#include <cctype>

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

#include <LASlib/lasreader.hpp>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

/* -------------------------- tiny CLI parser -------------------------- */
struct Args {
  std::string root;
  std::string file_format = "bin";          // "las", "laz", or "bin"
  std::string points_name = "points";       // basename (/no extension)
  double      downsample_resolution = 0.25; // meters; <=0 disables
  int         downsample_threads    = 4;
  std::string optimizer_type = "LM";        // "LM" or "ISAM2"
  std::string factor_type    = "VGICP_GPU";      // "ICP","ICP_PLANE","GICP","VGICP","VGICP_GPU"
};

static inline std::string lower_copy(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

static void print_usage(const char* prog) {
  std::cerr <<
    "Usage: " << prog << " <directory> [options]\n"
    "Options:\n"
    "  --file_format <las|laz|bin>         (default: bin)\n"
    "  --points_name <basename>            (default: points)\n"
    "  --downsample_resolution <meters>    (default: 0.25)\n"
    "  --downsample_threads <int>          (default: 4)\n"
    "  --optimizer_type <LM|ISAM2>         (default: LM)\n"
    "  --factor_type <ICP|ICP_PLANE|GICP|VGICP|VGICP_GPU> (default: GICP)\n";
}

static bool parse_cli(int argc, char** argv, Args& out) {
  if (argc < 2) { print_usage(argv[0]); return false; }
  out.root = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { print_usage(argv[0]); return false; }
    if (a.rfind("--", 0) != 0) { std::cerr << "Unknown arg: " << a << "\n"; return false; }

    auto key = a.substr(2);
    std::string val;
    auto eq = key.find('=');
    if (eq != std::string::npos) {
      val = key.substr(eq + 1);
      key = key.substr(0, eq);
    } else {
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("-", 0) != 0) {
        val = argv[++i];
      } else {
        std::cerr << "Missing value for --" << key << "\n"; return false;
      }
    }

    if (key == "file_format") out.file_format = lower_copy(val);
    else if (key == "points_name") out.points_name = val;
    else if (key == "downsample_resolution") out.downsample_resolution = std::stod(val);
    else if (key == "downsample_threads") out.downsample_threads = std::stoi(val);
    else if (key == "optimizer_type") out.optimizer_type = val;
    else if (key == "factor_type") out.factor_type = val;
    else { std::cerr << "Unknown option --" << key << "\n"; return false; }
  }

  // normalize
  out.optimizer_type = lower_copy(out.optimizer_type);
  out.factor_type    = lower_copy(out.factor_type);

  // validate
  if (out.file_format != "las" && out.file_format != "laz" && out.file_format != "bin") {
    std::cerr << "Invalid --file_format, expected las|laz|bin\n"; return false;
  }
  if (out.optimizer_type != "lm" && out.optimizer_type != "isam2") {
    std::cerr << "Invalid --optimizer_type, expected LM|ISAM2\n"; return false;
  }
  if (out.factor_type != "icp" && out.factor_type != "icp_plane" &&
      out.factor_type != "gicp" && out.factor_type != "vgicp" &&
      out.factor_type != "vgicp_gpu") {
    std::cerr << "Invalid --factor_type, expected ICP|ICP_PLANE|GICP|VGICP|VGICP_GPU\n"; return false;
  }
  if (out.downsample_threads < 1) out.downsample_threads = 1;
  return true;
}

/* --------------------- LAS/LAZ reader via LASlib --------------------- */
static std::vector<Eigen::Vector3f> read_points_from_las(const std::string& path) {
  LASreadOpener opener;
  opener.set_file_name(path.c_str());
  LASreader* reader = opener.open();
  if (!reader) {
    std::cerr << "error: failed to open " << path << " with LASlib\n";
    return {};
  }

  uint64_t n = 0;
  if (reader->header.version_minor >= 4 && reader->header.extended_number_of_point_records)
    n = reader->header.extended_number_of_point_records;
  else
    n = reader->header.number_of_point_records;

  std::vector<Eigen::Vector3f> points;
  if (n > 0 && n < static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    points.reserve(static_cast<size_t>(n));

  while (reader->read_point()) {
    const LASpoint& p = reader->point;
    points.emplace_back(
      static_cast<float>(p.get_x()),
      static_cast<float>(p.get_y()),
      static_cast<float>(p.get_z())
    );
  }

  reader->close();
  delete reader;
  return points;
}

/* =================================================================== */
class MatchingCostFactorDemo {
public:
  explicit MatchingCostFactorDemo(const Args& args)
      : data_path(args.root),
        cli(args) {

    /* -------- read graph ---------------------------------------------------- */
    std::ifstream ifs(data_path + "/graph.txt");
    if (!ifs) {
      std::cerr << "error: failed to open " << data_path + "/graph.txt" << std::endl;
      std::exit(1);
    }

#ifdef GTSAM_POINTS_USE_CUDA
    gtsam_points::LinearizationHook::register_hook(
        [] { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif

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
    std::cout << "[INFO] found " << num_frames << " frames\n";

    /* -------- containers ---------------------------------------------------- */
    frames.resize(num_frames);
    voxelmaps.resize(num_frames);
    voxelmaps_gpu.resize(num_frames);
    ids.resize(num_frames);

    /* -------- load every frame --------------------------------------------- */
    const std::string ext = cli.file_format; // "las","laz","bin"

    for (size_t i = 0; i < num_frames; ++i) {
      /* parse id and pose -------------------------------------------------- */
      std::istringstream iss(graph_lines[i]);
      std::string id;            // e.g., SKMES_1
      gtsam::Vector3 t;
      gtsam::Quaternion q;
      iss >> id >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >> q.w();

      ids[i] = id;

      gtsam::Pose3 P(gtsam::Rot3(q), t);
      poses.insert(i, P);
      poses_gt.insert(i, P);

      /* points file path ---------------------------------------------------- */
      const std::string points_path = data_path + "/" + id + "/" + cli.points_name + "." + ext;

      std::cout << "loading " << points_path << std::endl;
      std::vector<Eigen::Vector3f> pts_f;
      if (ext == "las" || ext == "laz") {
        pts_f = read_points_from_las(points_path);
      } else { // "bin"
        pts_f = gtsam_points::read_points(points_path);
      }

      std::cout << "  -> loaded " << pts_f.size() << " points\n";
      if (pts_f.empty()) {
        std::cerr << "error: no points read from " << points_path << "\n";
        std::exit(1);
      }

      /* small_gicp voxel downsampling ------------------------------------- */
      auto gicp_cloud = std::make_shared<small_gicp::PointCloud>(pts_f);
      if (cli.downsample_resolution > 0.0) {
        gicp_cloud = small_gicp::voxelgrid_sampling_omp(
            *gicp_cloud, cli.downsample_resolution, cli.downsample_threads);
      }
      std::cout << "num points after downsampling : " << gicp_cloud->size() << std::endl;

      /* to (x,y,z,1) double ----------------------------------------------- */
      std::vector<Eigen::Vector4d> pts_d;
      pts_d.reserve(gicp_cloud->points.size());
      for (const auto& p : gicp_cloud->points) {
        Eigen::Vector4d v;
        v << static_cast<double>(p[0]),
             static_cast<double>(p[1]),
             static_cast<double>(p[2]),
             1.0;
        pts_d.push_back(v);
      }

      auto covs = gtsam_points::estimate_covariances(pts_d);

#ifndef GTSAM_POINTS_USE_CUDA
      auto frame = std::make_shared<gtsam_points::PointCloudCPU>();
#else
      auto frame = std::make_shared<gtsam_points::PointCloudGPU>();
#endif
      frame->add_points(pts_d);
      frame->add_covs(covs);
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
    }

    /* --------------- algorithm choices ------------------------------------ */
    optimizer_types  = {"LM", "ISAM2"};
    factor_types     = {"ICP", "ICP_PLANE", "GICP", "VGICP"
#ifdef GTSAM_POINTS_USE_CUDA
                        , "VGICP_GPU"
#endif
    };

    optimizer_type = (cli.optimizer_type == "lm") ? 0 : 1;

    if      (cli.factor_type == "icp")        factor_type = 0;
    else if (cli.factor_type == "icp_plane")  factor_type = 1;
    else if (cli.factor_type == "gicp")       factor_type = 2;
    else if (cli.factor_type == "vgicp")      factor_type = 3;
    else if (cli.factor_type == "vgicp_gpu") {
#ifdef GTSAM_POINTS_USE_CUDA
      factor_type = 4;
#else
      std::cerr << "[WARN] VGICP_GPU requested but CUDA build is off. Falling back to VGICP.\n";
      factor_type = 3;
#endif
    } else {
      std::cerr << "[WARN] Unknown factor_type; defaulting to GICP.\n";
      factor_type = 2;
    }

    // Validate indices
    if (optimizer_type < 0 || optimizer_type >= static_cast<int>(optimizer_types.size())) {
      std::cerr << "[WARN] invalid optimizer_type; forcing LM\n";
      optimizer_type = 0;
    }
    if (factor_type < 0 || factor_type >= static_cast<int>(factor_types.size())) {
      std::cerr << "[WARN] invalid factor_type; forcing GICP\n";
      factor_type = 2;
    }

    full_connection = true;
    num_threads     = std::max(1, cli.downsample_threads); // reuse for ICP/GICP threads
    correspondence_update_tolerance_rot   = 0.0f;
    correspondence_update_tolerance_trans = 0.0f;

    std::cout << "\n=== Config ===\n"
              << "root: " << data_path << "\n"
              << "format: " << ext << "\n"
              << "points_name: " << cli.points_name << "\n"
              << "downsample_resolution: " << cli.downsample_resolution << "\n"
              << "downsample_threads: " << cli.downsample_threads << "\n"
              << "optimizer: " << optimizer_types[optimizer_type] << "\n"
              << "factor: " << factor_types[factor_type] << "\n\n";

    run_optimization();
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

  gtsam::NonlinearFactor::shared_ptr create_factor(
      gtsam::Key tgt, gtsam::Key src,
      const gtsam_points::PointCloud::ConstPtr& tgt_pc,
      const gtsam_points::GaussianVoxelMap::ConstPtr& tgt_vm,
      const gtsam_points::GaussianVoxelMap::ConstPtr& tgt_vm_gpu,
      const gtsam_points::PointCloud::ConstPtr& src_pc) {

    const std::string ft = factor_types[factor_type];

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
#ifdef GTSAM_POINTS_USE_CUDA
    if (ft == "VGICP_GPU") {
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(tgt, src, tgt_vm_gpu, src_pc);
    }
#endif

    std::cerr << "unknown factor type " << ft << '\n';
    return nullptr;
  }

  /* --------------------- optimization -------------------------------------- */
  void run_optimization() {
    gtsam::NonlinearFactorGraph graph;

    // Strong prior on the first pose
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        0, poses.at<gtsam::Pose3>(0),
        gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

    // Add pairwise factors
    size_t num_added = 0;
    for (size_t i = 0; i < num_frames; ++i) {
      size_t j_end = full_connection ? num_frames : std::min(i + 2, num_frames);
      for (size_t j = i + 1; j < j_end; ++j) {
        auto f = create_factor(i, j, frames[i], voxelmaps[i],
                               voxelmaps_gpu[i], frames[j]);
        if (f) { graph.add(f); ++num_added; }
      }
    }
    if (num_added == 0) {
      std::cerr << "[ERROR] No pairwise factors were created. Check --factor_type.\n";
      std::exit(1);
    }

    /* ---------- LM --------------------------------------------------------- */
    if (optimizer_types[optimizer_type] == std::string("LM")) {
      gtsam_points::LevenbergMarquardtExtParams prm;
      prm.maxIterations = 300;
      prm.callback = [](auto& st, const gtsam::Values&) {
        std::cout << st.to_string() << std::endl;
      };

      gtsam_points::LevenbergMarquardtOptimizerExt opt(graph, poses, prm);
      opt.optimize();
      save_values(opt.values(), data_path + "/optimized.txt");
    }

    /* ---------- ISAM2 ------------------------------------------------------ */
    else {
      gtsam::ISAM2Params p;
      p.relinearizeSkip = 1;
      p.setRelinearizeThreshold(0.f);
      gtsam_points::ISAM2Ext isam2(p);

      isam2.update(graph, poses);
      for (size_t i = 0; i < num_frames; ++i) {
        isam2.update();
      }
      save_values(isam2.calculateEstimate(), data_path + "/optimized.txt");
    }
  }

  /* ----------------- data --------------------------------------------------- */
  std::string data_path;
  size_t      num_frames{0};

  std::vector<const char*> optimizer_types;
  std::vector<const char*> factor_types;
  int  optimizer_type{0};
  int  factor_type{2};
  bool full_connection{true};
  int  num_threads{4};
  float correspondence_update_tolerance_rot{0.0f};
  float correspondence_update_tolerance_trans{0.0f};

  gtsam::Values poses, poses_gt;
  std::vector<gtsam_points::PointCloud::Ptr>       frames;
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps, voxelmaps_gpu;
  std::vector<std::string> ids;

  Args cli;
};

/* =================================================================== */
int main(int argc, char** argv) {
  Args args;
  if (!parse_cli(argc, argv, args)) return 1;
  MatchingCostFactorDemo demo(args);
  return 0;
}
/* =================================================================== */
