cmake_minimum_required(VERSION 2.8.3)
project(openface2_bridge)

find_package(catkin REQUIRED COMPONENTS
  image_transport
  cv_bridge
  std_msgs
  geometry_msgs
  sensor_msgs
  roscpp
  tf2
  tf2_ros
  openface2_ros_msgs
)

catkin_package(
    CATKIN_DEPENDS roscpp package cv_bridge tf2 image_transport openface2_ros_msgs    
)

find_package(Boost 1.5.9 REQUIRED COMPONENTS filesystem system)
find_package(OpenCV 4 REQUIRED)
find_package(dlib REQUIRED)

message("dlib_DIR:  ${dlib_DIR}")

find_library(OPENFACE_LD_LIB LandmarkDetector)
find_library(OPENFACE_FA_LIB FaceAnalyser)
find_library(OPENFACE_GA_LIB GazeAnalyser)
find_library(OPENFACE_UT_LIB Utilities)

set(OPENFACE_LIBS ${OPENFACE_LD_LIB} ${OPENFACE_FA_LIB} ${OPENFACE_GA_LIB} ${OPENFACE_UT_LIB})

find_path(OPENFACE_INCLUDE_DIR LandmarkDetectorModel.h PATH_SUFFIXES OpenFace)

add_definitions(-std=c++11)
add_definitions(-g -pg)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OPENFACE_INCLUDE_DIR}
  ${OpenBLAS_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}/boost
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(openface2_bridge
    src/openface2_ros.cpp
)

add_executable(headpose
        src/headpose.cpp
        )

add_executable(headpose_single
        src/headpose_single.cpp
        )

add_dependencies(openface2_bridge ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(headpose ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(headpose_single ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(openface2_bridge
  ${catkin_LIBRARIES}
  ${OPENFACE_LIBS}
  openblas
  dlib::dlib
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(headpose
        ${catkin_LIBRARIES}
        ${OPENFACE_LIBS}
        openblas
        dlib::dlib
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )
target_link_libraries(headpose_single
        ${catkin_LIBRARIES}
        ${OPENFACE_LIBS}
        openblas
        dlib::dlib
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        )

install(TARGETS openface2_bridge
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS headpose
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
install(TARGETS headpose_single
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )


install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
