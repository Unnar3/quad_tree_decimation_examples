#include <ros/ros.h>
#include <quad_tree_decimation/plane_extraction/plane_extraction.h>
#include <quad_tree_decimation/quad_tree/quad_tree_pcl.h>
#include <quad_tree_decimation/quad_tree/quad_tree.h>
#include <quad_tree_decimation/utils/utils.h>
#include <exx_compression/compression.h>
// #include <exx_compression/planes.h>
#include <metaroom_xml_parser/simple_xml_parser.h>
// #include <PointTypes/surfel_type.h>
// #include <pcl/visualization/pcl_visualizer.h>
// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>


// DEFINITIONS
#define NODE_NAME               "test_surfel"

using PointT = pcl::PointXYZRGB;
using PointCloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;

using namespace QTD::Parameters;
// using namespace EXX;

class Comparison
{
public:
    ros::NodeHandle nh;
private:
    // ros::Publisher point_cloud_publisher;
    EXX::compression cmprs;
    QTD::planeExtraction planeEx;
    primitive_params params;

public:

    Comparison()
    {
        nh = ros::NodeHandle("~");
        cmprs.setVoxelLeafSize(loadParam<double>("VoxelLeafSize", nh));
        cmprs.setSVVoxelResolution(loadParam<double>("SVVoxelResolution", nh));
        cmprs.setSVSeedResolution(loadParam<double>("SVSeedResolution", nh));
        cmprs.setSVColorImportance(loadParam<double>("SVColorImportance", nh));
        cmprs.setSVSpatialImportance(loadParam<double>("SVSpatialImportance", nh));
        cmprs.setRWHullMaxDist(loadParam<double>("RWHullMaxDist", nh));
        cmprs.setHULLAlpha(loadParam<double>("hullAlpha", nh));
        cmprs.setGP3SearchRad( loadParam<double>("GP3SearchRad", nh) );
        cmprs.setGP3Mu( loadParam<double>("GP3Mu", nh) );
        cmprs.setGP3MaxNearestNeighbours( loadParam<double>("GP3MaxNearestNeighbours", nh) );
        cmprs.setGP3Ksearch( loadParam<double>("GP3Ksearch", nh) );
        //
        QTD::QuadTreePCL<pcl::PointXYZ> qtpcl(1,10,0,0);

        params.number_disjoint_subsets = loadParam<int>("disjoinedSet", nh);
        params.octree_res              = loadParam<double>("octree_res", nh);
        params.normal_neigbourhood     = loadParam<double>("normal_neigbourhood", nh);
        params.inlier_threshold        = loadParam<double>("inlier_threshold", nh);
        params.angle_threshold         = loadParam<double>("angle_threshold", nh);
        params.add_threshold           = loadParam<double>("add_threshold", nh);
        params.connectedness_res       = loadParam<double>("connectedness_res", nh);
        params.distance_threshold      = loadParam<double>("distance_threshold", nh);
        params.inlier_min              = loadParam<int>("inlier_min", nh);
        params.min_shape               = loadParam<int>("min_shape", nh);

        planeEx.setPrimitiveParameters(params);
        // std::cout << "leaf size: " << cmprs.getVoxelLeafSize() << std::endl;
    }

    void testComparison(void){
        std::string path = loadParam<std::string>("path_test", nh);
        std::string path_save = loadParam<std::string>("path_save", nh);

        PointCloudT::Ptr segment (new PointCloudT());
        NormalCloudT::Ptr normals (new NormalCloudT());

        pcl::PCDReader reader;
        reader.read (path + "testCloud_complete.pcd", *segment);

        std::cout << "segment: " << segment->points.size() << std::endl;


        std::vector<PointCloudT::Ptr> plane_vec;
        std::vector<Eigen::Vector4d> normal_vec;

        ////////////////////////////////////////////////////////////////////////
        // EFFICIENT RANSAC with PPR
        ////////////////////////////////////////////////////////////////////////

        PointCloudT::Ptr nonPlanar (new PointCloudT());

        std::vector<PointCloudT::Ptr> plane_vec_efficient_ppr;
        std::vector<pcl::ModelCoefficients::Ptr> normal_vec_efficient_ppr;

        std::cout << "Efficient PPR..................." << std::endl;
        // planeDetection::planeSegmentationEfficientPPR(segment, params, plane_vec, normal_vec, nonPlanar);
        planeEx.planeSegmentationEfficientPPR(segment, normals, plane_vec_efficient_ppr, normal_vec_efficient_ppr, nonPlanar);
        // PROJECT TO PLANE
        for ( size_t i = 0; i < normal_vec_efficient_ppr.size(); ++i ){
            EXX::compression::projectToPlaneS( plane_vec_efficient_ppr[i], normal_vec_efficient_ppr[i] );
        }
        PointCloudT::Ptr outCloudEfficientPPR( new PointCloudT() );
        planeEx.combinePlanes(plane_vec_efficient_ppr, outCloudEfficientPPR, true);
        //
        std::cout << "Extracted " << plane_vec_efficient_ppr.size() << "  planes, Efficient PPR" << std::endl;
        std::cout << "Planar points: " << outCloudEfficientPPR->points.size() << std::endl;
        std::cout << "Non planar points: " << nonPlanar->points.size() << std::endl;
        std::cout << "Combined: " << outCloudEfficientPPR->points.size() + nonPlanar->points.size() << std::endl;
        std::cout << " " << std::endl;
        //
        // ////////////////////////////////////////////////////////////////////////
        // // EFFICIENT RANSAC
        // ////////////////////////////////////////////////////////////////////////
        PointCloudT::Ptr nonPlanar_efficient (new PointCloudT());
        std::vector<PointCloudT::Ptr> plane_vec_efficient;
        std::vector<pcl::ModelCoefficients::Ptr> normal_vec_efficient;
        normals->clear();
        // planeSegmentationNILS( segment, plane_vec_efficient, normal_vec_efficient, nonPlanar_efficient);
        planeEx.planeSegmentationEfficient(segment, normals, plane_vec_efficient, normal_vec_efficient, nonPlanar_efficient);
        for ( size_t i = 0; i < normal_vec_efficient.size(); ++i ){
            EXX::compression::projectToPlaneS( plane_vec_efficient[i], normal_vec_efficient[i] );
        }
        PointCloudT::Ptr outCloudEfficient( new PointCloudT() );
        planeEx.combinePlanes(plane_vec_efficient, outCloudEfficient, true);
        std::cout << "Extracted " << normal_vec_efficient.size() << "  planes, Efficient" << std::endl;
        std::cout << "Planar points: " << outCloudEfficient->points.size() << std::endl;
        std::cout << "Non planar points: " << nonPlanar_efficient->points.size() << std::endl;
        std::cout << "Combined: " << outCloudEfficient->points.size() + nonPlanar_efficient->points.size() << std::endl;
        std::cout << " " << std::endl;

        pcl::PCDWriter writer;
        writer.write(path_save + "outCloudEfficientPPR.pcd", *outCloudEfficientPPR);
        writer.write(path + "outCloudEfficient.pcd", *outCloudEfficient);

    }
};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    Comparison test;

    ros::Rate loop_rate(10);
    test.testComparison();

    return 0;
}
