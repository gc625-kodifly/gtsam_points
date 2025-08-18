cmake_minimum_required(VERSION 3.15)

# You can also use CONFIG if you installed pybind11 system-wide:
# find_package(pybind11 CONFIG REQUIRED)
find_package(pybind11 REQUIRED)

add_library(gtsam_points_py MODULE
  gtsam_points_py.cpp
)

# Link against your core library target
target_link_libraries(gtsam_points_py
  PRIVATE
    pybind11::module
    gtsam_points
)

# Propagate CUDA define if you build with CUDA
if(BUILD_WITH_CUDA)
  target_compile_definitions(gtsam_points_py PRIVATE GTSAM_POINTS_USE_CUDA)
endif()

# Nice module name & drop the "lib" prefix
set_target_properties(gtsam_points_py PROPERTIES
  PREFIX ""
  OUTPUT_NAME "gtsam_points_py"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

target_compile_features(gtsam_points_py PRIVATE cxx_std_17)

# (Optional) install rule — uncomment and adjust if you want `make install` to place it.
install(TARGETS gtsam_points_py
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python
)
