// gtsam_points_py.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>
#ifdef GTSAM_POINTS_USE_CUDA
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>
#include <gtsam_points/optimizers/linearization_hook.hpp>
#endif

#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor_gpu.hpp>
#include <gtsam_points/optimizers/isam2_ext.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// ---------------- Options ----------------
struct Options {
  double downsample_resolution = 0.25;   // kept for future; not used if you pre-downsample in Python
  int    downsample_threads    = 4;      // "
  std::string optimizer_type   = "LM";   // "LM" or "ISAM2"
  std::string factor_type      = "GICP"; // "ICP","ICP_PLANE","GICP","VGICP","VGICP_GPU"
  bool   full_connection       = true;   // fully connect frames vs chain
  int    num_threads           = 4;      // factor linearization threads
  float  corr_tol_rot          = 0.0f;
  float  corr_tol_trans        = 0.0f;
  int    root_index            = -1;     // -1 => use first element (index 0)
};

static inline std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
  return s;
}

static void build_frame_from_points(
    const Eigen::Ref<const Eigen::MatrixXd>& pts_xyz,
    std::shared_ptr<gtsam_points::PointCloud>& frame,
    std::shared_ptr<gtsam_points::GaussianVoxelMap>& vmap,
    std::shared_ptr<gtsam_points::GaussianVoxelMap>& vmap_gpu)
{
  if (pts_xyz.cols() != 3 && pts_xyz.cols() != 4)
    throw std::invalid_argument("points must be Nx3 or Nx4");
  std::vector<Eigen::Vector4d> pts4; pts4.reserve((size_t)pts_xyz.rows());
  for (int i = 0; i < pts_xyz.rows(); ++i) {
    Eigen::Vector4d p(0,0,0,1);
    p.head<3>() = pts_xyz.row(i).head<3>();
    if (pts_xyz.cols() == 4) p(3) = pts_xyz(i,3);
    pts4.push_back(p);
  }
  #ifndef GTSAM_POINTS_USE_CUDA
    auto frame_cpu = std::make_shared<gtsam_points::PointCloudCPU>();
    frame_cpu->add_points(pts4);
    frame_cpu->add_covs(gtsam_points::estimate_covariances(pts4));
    frame_cpu->add_normals(gtsam_points::estimate_normals(frame_cpu->points, frame_cpu->size()));
    frame = frame_cpu;
  #else
    auto frame_gpu = std::make_shared<gtsam_points::PointCloudGPU>();
    frame_gpu->add_points(pts4);
    frame_gpu->add_covs(gtsam_points::estimate_covariances(pts4));
    frame_gpu->add_normals(gtsam_points::estimate_normals(frame_gpu->points, frame_gpu->size()));
    frame = frame_gpu;
  #endif

  vmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(2.0);
  vmap->insert(*frame);
#ifdef GTSAM_POINTS_USE_CUDA
  vmap_gpu = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(2.0);
  vmap_gpu->insert(*frame);
#else
  (void)vmap_gpu;
#endif
}

// --------------- Core class ---------------
class MatchingCostCore {
public:
  explicit MatchingCostCore(const Options& o): opt_(o) {
#ifdef GTSAM_POINTS_USE_CUDA
    gtsam_points::LinearizationHook::register_hook(
      [] { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif
  }

  // nodes: [{"CWLPH_1": {"pts": ndarray(N,3/4), "init_pos": ndarray(7)}}, ...]
  void set_data_from_list(const py::list& nodes) {
    ids_.clear(); frames_.clear(); voxelmaps_.clear(); voxelmaps_gpu_.clear(); init_poses_.clear();

    ids_.reserve(nodes.size());
    frames_.resize(nodes.size());
    voxelmaps_.resize(nodes.size());
    voxelmaps_gpu_.resize(nodes.size());
    init_poses_.resize(nodes.size());

    int idx = 0;
    for (py::handle h : nodes) {
      py::dict d = py::cast<py::dict>(h);
      if (d.size() != 1) throw std::invalid_argument("each list item must be a dict with exactly one key");
      // extract the sole (id -> record) pair
      for (auto kv : d) {
        std::string id = py::cast<std::string>(kv.first);
        py::dict rec = py::cast<py::dict>(kv.second);

        if (!rec.contains("pts"))       throw std::invalid_argument("missing 'pts' for node " + id);
        if (!rec.contains("init_pos"))  throw std::invalid_argument("missing 'init_pos' for node " + id);

        Eigen::MatrixXd pts = py::cast<Eigen::MatrixXd>(rec["pts"]);
        Eigen::VectorXd v7  = py::cast<Eigen::VectorXd>(rec["init_pos"]);
        if (v7.size() != 7) throw std::invalid_argument("'init_pos' must be length 7 [tx ty tz qx qy qz qw]");

        std::shared_ptr<gtsam_points::PointCloud> frame;
        std::shared_ptr<gtsam_points::GaussianVoxelMap> vm, vm_gpu;
        build_frame_from_points(pts, frame, vm, vm_gpu);

        ids_.push_back(id);
        frames_[idx] = frame;
        voxelmaps_[idx] = vm;
        voxelmaps_gpu_[idx] = vm_gpu;

        // Pose: row vector [tx,ty,tz,qx,qy,qz,qw]
        Eigen::Vector3d t = v7.head<3>();
        Eigen::Quaterniond q(v7(6), v7(3), v7(4), v7(5)); // qw, qx, qy, qz
        init_poses_[idx] = gtsam::Pose3(gtsam::Rot3(q), t);
        ++idx;
      }
    }
  }

  // Returns: [{"node_id": np.array([tx,ty,tz,qx,qy,qz,qw])}, ...] in the same order
  py::list run_as_list_of_dicts() {
    py::gil_scoped_release release;

    // build initial Values
    gtsam::Values values;
    for (size_t i = 0; i < ids_.size(); ++i) values.insert(i, init_poses_[i]);

    // prior on root (by list order unless root_index is set)
    size_t root = (opt_.root_index >= 0) ? (size_t)opt_.root_index : 0;
    if (root >= ids_.size()) throw std::runtime_error("root_index out of range");
    gtsam::NonlinearFactorGraph graph;
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
      root, values.at<gtsam::Pose3>(root),
      gtsam::noiseModel::Isotropic::Precision(6, 1e6)));

    // connect factors
    for (size_t i = 0; i < ids_.size(); ++i) {
      size_t j_end = opt_.full_connection ? ids_.size() : std::min(ids_.size(), i + 2);
      for (size_t j = i + 1; j < j_end; ++j) {
        auto f = create_factor(i, j, frames_[i], voxelmaps_[i], voxelmaps_gpu_[i], frames_[j]);
        if (f) graph.add(f);
      }
    }

    // optimize
    if (lower_copy(opt_.optimizer_type) == "lm") {
      gtsam_points::LevenbergMarquardtExtParams prm;
      prm.maxIterations = 300;
      gtsam_points::LevenbergMarquardtOptimizerExt opt(graph, values, prm);
      opt.optimize();
      return values_to_list(opt.values());
    } else {
      gtsam::ISAM2Params p; p.relinearizeSkip = 1; p.setRelinearizeThreshold(0.f);
      gtsam_points::ISAM2Ext isam2(p);
      isam2.update(graph, values);
      for (size_t i = 0; i < ids_.size(); ++i) isam2.update();
      return values_to_list(isam2.calculateEstimate());
    }
  }

private:
  gtsam::NonlinearFactor::shared_ptr create_factor(
      gtsam::Key tgt, gtsam::Key src,
      const gtsam_points::PointCloud::ConstPtr& tgt_pc,
      const gtsam_points::GaussianVoxelMap::ConstPtr& tgt_vm,
      const gtsam_points::GaussianVoxelMap::ConstPtr& tgt_vm_gpu,
      const gtsam_points::PointCloud::ConstPtr& src_pc)
  {
    const auto ft = lower_copy(opt_.factor_type);
    if (ft == "icp") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(opt_.corr_tol_rot, opt_.corr_tol_trans);
      f->set_num_threads(opt_.num_threads);
      return f;
    }
    if (ft == "icp_plane") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedPointToPlaneICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(opt_.corr_tol_rot, opt_.corr_tol_trans);
      f->set_num_threads(opt_.num_threads);
      return f;
    }
    if (ft == "gicp") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(opt_.corr_tol_rot, opt_.corr_tol_trans);
      f->set_num_threads(opt_.num_threads);
      return f;
    }
    if (ft == "vgicp") {
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(tgt, src, tgt_vm, src_pc);
    }
    if (ft == "vgicp_gpu") {
#ifdef GTSAM_POINTS_USE_CUDA
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(tgt, src, tgt_vm_gpu, src_pc);
#else
      throw std::runtime_error("VGICP_GPU requested but CUDA not enabled");
#endif
    }
    throw std::runtime_error("Unknown factor_type: " + opt_.factor_type);
  }

  py::list values_to_list(const gtsam::Values& vals) const {
    py::list out;
    for (size_t i = 0; i < ids_.size(); ++i) {
      const auto& P = vals.at<gtsam::Pose3>(i);
      const auto& t = P.translation();
      gtsam::Quaternion q = P.rotation().toQuaternion();
      py::array_t<double> row(7);
      auto r = row.mutable_unchecked<1>();
      r(0)=t.x(); r(1)=t.y(); r(2)=t.z(); r(3)=q.x(); r(4)=q.y(); r(5)=q.z(); r(6)=q.w();
      py::dict item;
      item[py::str(ids_[i])] = std::move(row);
      out.append(std::move(item));
    }
    return out;
  }

private:
  Options opt_;
  std::vector<std::string> ids_;
  std::vector<gtsam::Pose3> init_poses_;
  std::vector<gtsam_points::PointCloud::Ptr>       frames_;
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps_, voxelmaps_gpu_;
};

// ---------------- Module ----------------
PYBIND11_MODULE(gtsam_points_py, m) {
  m.doc() = "GTSAM points pybind wrapper (list-of-dicts IO)";

  py::class_<Options>(m, "Options")
    .def(py::init<>())
    .def_readwrite("downsample_resolution", &Options::downsample_resolution)
    .def_readwrite("downsample_threads",    &Options::downsample_threads)
    .def_readwrite("optimizer_type",        &Options::optimizer_type)
    .def_readwrite("factor_type",           &Options::factor_type)
    .def_readwrite("full_connection",       &Options::full_connection)
    .def_readwrite("num_threads",           &Options::num_threads)
    .def_readwrite("corr_tol_rot",          &Options::corr_tol_rot)
    .def_readwrite("corr_tol_trans",        &Options::corr_tol_trans)
    .def_readwrite("root_index",            &Options::root_index);

  py::class_<MatchingCostCore>(m, "MatchingCostCore")
    .def(py::init<const Options&>(), py::arg("options"))
    .def("set_data_from_list", &MatchingCostCore::set_data_from_list, py::arg("nodes"))
    .def("run_as_list_of_dicts", &MatchingCostCore::run_as_list_of_dicts);
}
