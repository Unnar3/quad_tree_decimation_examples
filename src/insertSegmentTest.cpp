#include <ros/ros.h>
#include <quad_tree_decimation/quad_tree/quad_tree_pcl.h>
#include <quad_tree_decimation/quad_tree/quad_tree.h>
// #include <exx_compression/planes.h>
// #include <PointTypes/surfel_type.h>
// #include <pcl/visualization/pcl_visualizer.h>
// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>


// DEFINITIONS
#define NODE_NAME               "dataset_mesh"

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using PointC = pcl::PointXYZRGB;
using PointCloudC = pcl::PointCloud<PointC>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;

// using namespace EXX;

class InsertSegmentTest
{
public:
    ros::NodeHandle nh;
private:
    // ros::Publisher point_cloud_publisher;

public:

    InsertSegmentTest()
    {
        nh = ros::NodeHandle("~");
    }

    void createMesh(void){

        PointCloudT::Ptr cloud (new PointCloudT());
        cloud->reserve(5);
        cloud->push_back(PointT(1,1,0));
        cloud->push_back(PointT(2,1,0));
        cloud->push_back(PointT(3,2,0));
        cloud->push_back(PointT(3,4,0));
        cloud->push_back(PointT(2,2,0));

        QTD::QuadTreePCL<PointT> qtpcl(1,10,0,0);
        // qtpcl.setMaxLevel(10);
        qtpcl.setMaxWidth(0.1);
        Eigen::Vector3f normal;
        normal << 0,0,1;
        qtpcl.setNormal(normal);
        qtpcl.insertBoundary(cloud);

        PointCloudC::Ptr cloud_color (new PointCloudC());
        std::vector< pcl::Vertices > vertices;
        qtpcl.createMesh<PointC>(cloud_color, vertices);

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (0, 0, 0);
        // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
        viewer->addPolygonMesh<pcl::PointXYZRGB> (cloud_color, vertices);
        // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
        // viewer->addCoordinateSystem (1.0);
        viewer->initCameraParameters ();
        viewer->spin();
    }

};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    InsertSegmentTest dMesh;

    ros::Rate loop_rate(10);
    dMesh.createMesh();
    return 0;
}
