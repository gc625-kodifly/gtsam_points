#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <map>
#include <string>
#include <vector>

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





// so we just do py::sth instead of pybind11::sth
namespace py = pybind11;


struct FrameData {
    std::string id;
    Eigen::VectorXd pose;   // dynamic
    Eigen::MatrixXd points;

    FrameData(const std::string& id_,
              const Eigen::VectorXd& pose_,
              const Eigen::MatrixXd& pts)
        : id(id_), pose(pose_), points(pts)
    {
        if (pose_.size() != 7)
            throw std::runtime_error("pose must have length 7");
    }
};

struct OptimizerParams{
  bool full_connection;
  int num_threads;
  float correspondence_update_tolerance_rot;
  float correspondence_update_tolerance_trans;
  std::string optimizer_type;
  std::string factor_type;
};



class CostFactorMerge {


public: 
  explicit CostFactorMerge(const OptimizerParams& p) : opt_params(p) {
#ifdef GTSAM_POINTS_USE_CUDA
    gtsam_points::LinearizationHook::register_hook(
        [] { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif
    
  }
  void load_frames(const std::vector<FrameData>& frame_data) {

    num_frames = frame_data.size();
    frames.resize(num_frames);
    voxelmaps.resize(num_frames);
    voxelmaps_gpu.resize(num_frames);
    ids.resize(num_frames);

    for (std::size_t i = 0; i < num_frames; ++i) {
      const auto& f = frame_data[i];      
      py::print(
        "loading frame:",f.id,
        "pose shape:", f.pose.rows(), "x", f.pose.cols(),
        "points shape:", f.points.rows(), "x", f.points.cols()
      );
      
      ids[i] = f.id;
      gtsam::Pose3 P(
          gtsam::Rot3::Quaternion(f.pose(6), f.pose(3), f.pose(4), f.pose(5)),
          gtsam::Vector3(f.pose(0), f.pose(1), f.pose(2))
        );
      
      poses.insert(i, P);
      poses_gt.insert(i, P);

      std::vector<Eigen::Vector4d> pts_d; 
      pts_d.reserve(static_cast<size_t>(f.points.rows()));
      for (Eigen::Index r = 0; r < f.points.rows(); ++r){
        pts_d.emplace_back(
          static_cast<float>(f.points(r,0)),
          static_cast<float>(f.points(r,1)),
          static_cast<float>(f.points(r,2)),
          1.0
        );
      }
      // points_d.emplace_back(std::move(pts_d));

      auto covs = gtsam_points::estimate_covariances(pts_d);

#ifndef GTSAM_POINTS_USE_CUDA
      py::print("using PointCloudCPU");
      auto frame = std::make_shared<gtsam_points::PointCloudCPU>();
#else
      py::print("using PointCloudGPU");
      auto frame = std::make_shared<gtsam_points::PointCloudGPU>();
#endif
      py::print("adding points for idx:", i);
      frame->add_points(pts_d);
      py::print("adding covs for idx:", i);
      frame->add_covs(covs);
      py::print("adding norms for idx:", i);
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

  }

  gtsam::NonlinearFactor::shared_ptr create_factor(
       gtsam::Key tgt, gtsam::Key src,
      const gtsam_points::PointCloud::ConstPtr& tgt_pc,
      const gtsam_points::GaussianVoxelMap::ConstPtr& tgt_vm,
      const gtsam_points::GaussianVoxelMap::ConstPtr& tgt_vm_gpu,
      const gtsam_points::PointCloud::ConstPtr& src_pc){


    if (opt_params.factor_type == "ICP") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(opt_params.correspondence_update_tolerance_rot,
                                             opt_params.correspondence_update_tolerance_trans);
      f->set_num_threads(opt_params.num_threads);
      return f;
    }
    if (opt_params.factor_type == "ICP_PLANE") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedPointToPlaneICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(opt_params.correspondence_update_tolerance_rot,
                                             opt_params.correspondence_update_tolerance_trans);
      f->set_num_threads(opt_params.num_threads);
      return f;
    }
    if (opt_params.factor_type == "GICP") {
      auto f = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(tgt, src, tgt_pc, src_pc);
      f->set_correspondence_update_tolerance(opt_params.correspondence_update_tolerance_rot,
                                             opt_params.correspondence_update_tolerance_trans);
      f->set_num_threads(opt_params.num_threads);
      return f;
    }
    if (opt_params.factor_type == "VGICP") {
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactor>(tgt, src, tgt_vm, src_pc);
    }
#ifdef GTSAM_POINTS_USE_CUDA
    if (opt_params.factor_type == "VGICP_GPU") {
      return gtsam::make_shared<gtsam_points::IntegratedVGICPFactorGPU>(tgt, src, tgt_vm_gpu, src_pc);
    }
#endif

    std::cerr << "unknown factor type " <<  opt_params.factor_type << '\n';
    return nullptr;
  }


        
  void run_optimization(){


    gtsam::NonlinearFactorGraph graph;

    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
      0, poses.at<gtsam::Pose3>(0),
      gtsam::noiseModel::Isotropic::Precision(6, 1e6)
    ));
   
    
    size_t num_added = 0;

    for (size_t i = 0; i < num_frames; ++i){
      size_t j_end = opt_params.full_connection ? num_frames : std::min(i + 2, num_frames);
      for (size_t j = i + 1; j < j_end; ++j) {
        auto f = create_factor(i, j, frames[i], voxelmaps[i], voxelmaps_gpu[i], frames[j]); 
        if (f) {graph.add(f); ++num_added;}
      }
    }
    if (num_added == 0) {
      std::cerr << "[ERROR] No pairwise factors were created. Check --factor_type.\n";
      std::exit(1);
    }


    if (opt_params.optimizer_type == std::string("LM")){
      gtsam_points::LevenbergMarquardtExtParams prm;
      prm.maxIterations = 300;
      prm.callback = [](auto& st, const gtsam::Values&){
        std::cout << st.to_string() << std::endl;
      };

      gtsam_points::LevenbergMarquardtOptimizerExt opt(graph, poses, prm);
      opt.optimize();
      result_ = opt.values();
    }

    else{
      gtsam::ISAM2Params p;
      p.relinearizeSkip = 1;
      p.setRelinearizeThreshold(0.f);
      gtsam_points::ISAM2Ext isam2(p);

      isam2.update(graph, poses);
      for (size_t i = 0; i < num_frames; ++i) {
        isam2.update();
      }
      result_ = isam2.calculateEstimate(); 
    }

    
  }
  py::dict get_optimized_poses() const {
    py::dict out;
    for (size_t i = 0; i < num_frames; ++i) {
      if (!result_.exists(i)) continue;
      const auto& P = result_.at<gtsam::Pose3>(i);
      const auto& t = P.translation();
      gtsam::Quaternion q = P.rotation().toQuaternion();

      Eigen::Matrix<double,7,1> v;
      v << t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w();
      out[py::str(ids[i])] = v;   // auto-converts to numpy
    }
    return out;
  }
  std::size_t size() const { return ids.size(); }
  std::vector<std::string> get_ids() const { return ids; }
private: 
  size_t num_frames;

  std::vector<std::string> ids;
  std::vector<std::vector<Eigen::Vector4d>> points_d;
  gtsam_points::Values poses, poses_gt;
  std::vector<gtsam_points::PointCloud::Ptr> frames;
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps, voxelmaps_gpu;

  OptimizerParams opt_params;
  gtsam::Values result_;


};



PYBIND11_MODULE(gtsam_points_py, m){
  py::class_<FrameData>(m, "FrameData")
    .def(py::init<const std::string&, const Eigen::VectorXd&, const Eigen::MatrixXd&>(),
         py::arg("id"), py::arg("pose"), py::arg("points"))
    .def_readwrite("id", &FrameData::id)
    .def_readwrite("pose", &FrameData::pose)
    .def_readwrite("points", &FrameData::points);
   // m.def("consume_dict", &consume_dict, "Parse dict[str, ndarray] into C++");

  py::class_<OptimizerParams>(m, "OptimizerParams")
    .def(py::init<const bool&, 
      const int&, 
      const float&, 
      const float&, 
      const std::string&, 
      const std::string&>(),
    py::arg("full_connection"),
    py::arg("num_threads"), py::arg("correspondence_update_tolerance_rot"), 
    py::arg("correspondence_update_tolerance_trans"), py::arg("optimizer_type"),
    py::arg("factor_type"))
    .def_readwrite("full_connection", &OptimizerParams::full_connection)
    .def_readwrite("num_threads", &OptimizerParams::num_threads)
    .def_readwrite("correspondence_update_tolerance_rot", &OptimizerParams::correspondence_update_tolerance_rot)
    .def_readwrite("correspondence_update_tolerance_trans", &OptimizerParams::correspondence_update_tolerance_trans)
    .def_readwrite("optimizer_type", &OptimizerParams::optimizer_type)
    .def_readwrite("factor_type", &OptimizerParams::factor_type);


   
  py::class_<CostFactorMerge>(m, "CostFactorMerge")
    .def(py::init<const OptimizerParams&>(), py::arg("params"))
    .def("load_frames", &CostFactorMerge::load_frames, py::arg("frames"),
         "Load a list of FrameData into the merger")
    .def("size", &CostFactorMerge::size)
    .def("ids", &CostFactorMerge::get_ids)
    .def("run_optimization", &CostFactorMerge::run_optimization)
    .def("get_optimized_poses", &CostFactorMerge::get_optimized_poses);


}