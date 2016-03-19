#include <ros/ros.h>
#include <quad_tree_decimation/plane_extraction/plane_extraction.h>
#include <quad_tree_decimation/quad_tree/quad_tree_pcl.h>
#include <quad_tree_decimation/quad_tree/quad_tree.h>
#include <quad_tree_decimation/utils/utils.h>
#include <exx_compression/compression.h>
#include <pcl/kdtree/kdtree_flann.h>
// #include <exx_compression/planes.h>
#include <metaroom_xml_parser/simple_xml_parser.h>
// #include <PointTypes/surfel_type.h>
// #include <pcl/visualization/pcl_visualizer.h>
// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


// DEFINITIONS
#define NODE_NAME               "test_cloud_mesh"

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

        std::string test_cloud = loadParam<std::string>("test_cloud", nh);
        std::string test_cloud_name = loadParam<std::string>("test_cloud_name", nh);
        std::string save_path = loadParam<std::string>("save_path_testCloudMesh", nh);

        // find variance from name
        std::string var = test_cloud_name.substr(test_cloud_name.find_last_of("_")+1,3);
        std::string norm = std::to_string(params.normal_neigbourhood);
        norm = norm.substr(norm.find_last_of(".")+1,3);

        // read appropriate txt file;
        std::string txt_name = test_cloud_name.substr(0,test_cloud_name.find_last_of(".")) + ".txt";


        PointCloudT::Ptr segment (new PointCloudT());
        NormalCloudT::Ptr normals (new NormalCloudT());

        pcl::PCDReader reader;
        reader.read (test_cloud + test_cloud_name, *segment);

        std::cout << "segment: " << segment->points.size() << std::endl;


        ////////////////////////////////////////////////////////////////////////
        // EFFICIENT RANSAC with PPR
        ////////////////////////////////////////////////////////////////////////

        PointCloudT::Ptr nonPlanar (new PointCloudT());

        std::vector<PointCloudT::Ptr> plane_vec;
        std::vector<pcl::ModelCoefficients::Ptr> normal_vec;

        std::cout << "Efficient PPR..................." << std::endl;
        planeEx.planeSegmentationEfficientPPR(segment, normals, plane_vec, normal_vec, nonPlanar);
        // PROJECT TO PLANE

        planeEx.projectToPlane<PointT>(plane_vec, normal_vec);


        EXX::compression cmprs;
        cmprs.setRWHullMaxDist(0.02);
        cmprs.setHULLAlpha(0.07);

        std::vector<PointCloudT::Ptr> hulls;
        cmprs.planeToConcaveHull(&plane_vec, &hulls);
        std::vector<EXX::densityDescriptor> dDesc;
        EXX::densityDescriptor dens;
        dens.rw_max_dist = 0.15;
        dDesc.push_back(dens);
        std::vector<PointCloudT::Ptr> simplified_hulls;
        cmprs.reumannWitkamLineSimplification( &hulls, &simplified_hulls, dDesc);


        QTD::objStruct<PointT> object(1);
        pcl::PointCloud<PointT>::Ptr bound (new pcl::PointCloud<PointT>);

        for ( size_t i = 0; i < plane_vec.size(); ++i ){
        // for ( size_t i = 1; i < 2; ++i ){

            QTD::QuadTreePCL<PointT> qtpcl(1,10,0,0);
            // qtpcl.setMaxLevel(10);
            qtpcl.setMaxWidth(1);
            qtpcl.setNormal(normal_vec[i]);

            PointCloudT::Ptr out (new PointCloudT());
            std::vector< pcl::Vertices > vertices;
            qtpcl.insertBoundary(hulls[i]);
            qtpcl.createMeshNew<PointT>(out, vertices);

            cv::Mat image;
            std::vector<Eigen::Vector2f> vertex_texture;
            qtpcl.createTexture<PointT>(plane_vec[i], out, image, vertex_texture);

            object.clouds.push_back(out);
            object.polygons.push_back(vertices);
            object.images.push_back(image);
            object.texture_vertices.push_back(vertex_texture);
            object.coefficients.push_back(normal_vec[i]);
            // qtpcl.createBoundary(bound);

        }
        QTD::saveOBJFile(save_path + "mesh.obj", object, 5);



        // pcl::PCDWriter writer;
        // writer.write(save_path + "bound.pcd", *bound);

    }

};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    Comparison test;

    ros::Rate loop_rate(10);
    test.testComparison();

    return 0;
}
