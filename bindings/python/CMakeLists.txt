# We are inside bindings/python. No need for project() again.

# Find dependencies (safe to call here too)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

# Build the Python extension. The name here must match PYBIND11_MODULE(...)
pybind11_add_module(gtsam_points_py
    gtsam_points_py.cpp
)

# Link dependencies
target_link_libraries(gtsam_points_py
    PRIVATE
      Eigen3::Eigen
      $<TARGET_NAME_IF_EXISTS:gtsam_points>   # link core lib if it exists
)

# Nice settings for Python extensions
set_target_properties(gtsam_points_py PROPERTIES
    PREFIX ""                                         # no "lib" prefix
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}  # drop .so here
)

# Optional: more detailed pybind11 error messages
target_compile_definitions(gtsam_points_py PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

set_target_properties(gtsam_points_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)
