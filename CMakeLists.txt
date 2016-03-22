cmake_minimum_required(VERSION 2.8.3)
project(quad_tree_decimation_example)

set(CMAKE_CXX_FLAGS "-O4 -g -fPIC -std=c++0x -fpermissive")

find_package(catkin REQUIRED COMPONENTS
	roscpp
	#pcl_conversions
  	pcl_ros
  	sensor_msgs
  	std_msgs
  	message_generation
    qt_build
    quad_tree_decimation
    metaroom_xml_parser
    exx_compression
)

rosbuild_prepare_qt4(QtCore QtXml)

#add_message_files(FILES
#)

#generate_messages(DEPENDENCIES
#	std_msgs
#	sensor_msgs
#)


catkin_package(
	DEPENDS
	CATKIN_DEPENDS roscpp message_runtime qt_build libqt4-dev
	INCLUDE_DIRS include
	LIBRARIES
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(qtde_insertSegments src/insertSegmentTest.cpp)
target_link_libraries(qtde_insertSegments ${catkin_LIBRARIES})

add_executable(qtde_holeCheck src/holeCheck.cpp)
target_link_libraries(qtde_holeCheck ${catkin_LIBRARIES})

add_executable(qtde_datasetMesh src/datasetMesh.cpp)
target_link_libraries(qtde_datasetMesh ${catkin_LIBRARIES} ${QT_LIBRARIES} compression)

add_executable(qtde_fusionMesh src/fusionMesh.cpp)
target_link_libraries(qtde_fusionMesh ${catkin_LIBRARIES} ${QT_LIBRARIES} compression)

add_executable(qtde_comparison src/planeExtractionComparison.cpp)
target_link_libraries(qtde_comparison ${catkin_LIBRARIES} ${QT_LIBRARIES} compression)

add_executable(qtde_test_cloud_mesh src/testCloudMesh.cpp)
target_link_libraries(qtde_test_cloud_mesh ${catkin_LIBRARIES} ${QT_LIBRARIES} compression)

add_executable(qtde_createTestCloud src/createTestCloud.cpp)
target_link_libraries(qtde_createTestCloud ${catkin_LIBRARIES})
