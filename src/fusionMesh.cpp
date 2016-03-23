#include <ros/ros.h>
#include <quad_tree_decimation/plane_extraction/plane_extraction.h>
#include <quad_tree_decimation/quad_tree/quad_tree_pcl.h>
#include <quad_tree_decimation/quad_tree/quad_tree.h>
#include <quad_tree_decimation/utils/utils.h>
#include <exx_compression/compression.h>
#include <pcl/kdtree/kdtree_flann.h>
// #include <exx_compression/planes.h>
#include <metaroom_xml_parser/simple_xml_parser.h>
#include <pcl/filters/radius_outlier_removal.h>
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
#define NODE_NAME               "fusion_cloud_mesh"

using PointT = pcl::PointXYZRGB;
using PointCloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;

using namespace QTD::Parameters;
// using namespace EXX;

class FusionCloud
{
public:
    ros::NodeHandle nh;
private:
    // ros::Publisher point_cloud_publisher;
    EXX::compression cmprs;
    QTD::planeExtraction planeEx;
    primitive_params params;

public:

    FusionCloud()
    {
        nh = ros::NodeHandle("~");
        cmprs.setVoxelLeafSize(loadParam<double>("VoxelLeafSize", nh));
        cmprs.setRWHullMaxDist(loadParam<double>("RWHullMaxDist", nh));
        cmprs.setHULLAlpha(loadParam<double>("hullAlpha", nh));

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

    void testFusionCloud(void){

        std::string test_cloud_path = loadParam<std::string>("dataset_path", nh);
        std::cout << "test_cloud_path: " << test_cloud_path << std::endl;
        std::string test_cloud_name = loadParam<std::string>("dataset_cloud", nh);
        std::cout << "test_cloud_name: " << test_cloud_name << std::endl;
        std::string test_cloud = test_cloud_path + test_cloud_name;
        std::cout << "test_cloud: " << test_cloud << std::endl;
        std::string save_path = test_cloud.substr(0,test_cloud.find_last_of('/')) + '/';
        std::cout << "save_path: " << save_path << std::endl;
        std::string mesh_name = test_cloud.substr(test_cloud.find_last_of('/')+1);
        mesh_name = mesh_name.substr(0, mesh_name.find_last_of('.')) + ".obj";
        std::cout << "mesh_name: " << mesh_name << std::endl;

        PointCloudT::Ptr segment (new PointCloudT());
        NormalCloudT::Ptr normals (new NormalCloudT());

        pcl::PCDReader reader;
        reader.read (test_cloud, *segment);

        std::cout << "segment: " << segment->points.size() << std::endl;


        ////////////////////////////////////////////////////////////////////////
        // EFFICIENT RANSAC with PPR
        ////////////////////////////////////////////////////////////////////////

        PointCloudT::Ptr nonPlanar (new PointCloudT());

        std::vector<PointCloudT::Ptr> plane_vec;
        std::vector<pcl::ModelCoefficients::Ptr> normal_vec;
        while(true){
            std::vector<PointCloudT::Ptr> plane_vec_tmp;
            std::vector<pcl::ModelCoefficients::Ptr> normal_vec_tmp;

            planeEx.planeSegmentationEfficientPPR(segment, normals, plane_vec_tmp, normal_vec_tmp, nonPlanar);

            std::cout << "size: " << plane_vec_tmp.size() << std::endl;

            if(plane_vec_tmp.size() < 2) break;
            plane_vec.insert(plane_vec.end(), plane_vec_tmp.begin(), plane_vec_tmp.end());
            normal_vec.insert(normal_vec.end(), normal_vec_tmp.begin(), normal_vec_tmp.end());
            *segment = *nonPlanar;

            params.angle_threshold = params.angle_threshold + params.angle_threshold/2.0;
            params.inlier_min = params.inlier_min * 2.0;
            params.min_shape = params.inlier_min;
            planeEx.setPrimitiveParameters(params);
        }



        std::cout << "Efficient PPR..................." << std::endl;
        // planeEx.planeSegmentationEfficientPPR(segment, normals, plane_vec, normal_vec, nonPlanar);
        // PROJECT TO PLANE

        planeEx.projectToPlane<PointT>( plane_vec, normal_vec );
        planeEx.mergePlanes<PointT>( plane_vec, normal_vec );
        planeEx.projectToPlane<PointT>( plane_vec, normal_vec );


        for(auto &plane : plane_vec){
            if(plane->size() == 0){
                std::cout << "dafuck" << std::endl;
                continue;
            }
            pcl::RadiusOutlierRemoval<PointT> outrem;
            // build the filter
            outrem.setInputCloud(plane);
            outrem.setRadiusSearch(0.05);
            outrem.setMinNeighborsInRadius (15);
            // apply filter
            outrem.filter (*plane);
        }

        // EXX::compression cmprs;
        // cmprs.setRWHullMaxDist(0.02);
        // cmprs.setHULLAlpha(0.03);

        // for(int i = 0; i < plane_vec.size(); ++i){
        //     if(plane_vec[i]->points.size() < 100){
        //         plane_vec.erase(plane_vec.begin() + i);
        //         normal_vec.erase(normal_vec.begin() + i);
        //     }
        // }


        std::vector<PointCloudT::Ptr> hulls;
        cmprs.planeToConcaveHull(&plane_vec, &hulls);
        std::vector<EXX::densityDescriptor> dDesc;
        EXX::densityDescriptor dens;
        dens.rw_max_dist = 0.15;
        dDesc.push_back(dens);
        std::vector<PointCloudT::Ptr> simplified_hulls;
        // cmprs.reumannWitkamLineSimplification( &hulls, &simplified_hulls, dDesc);


        QTD::objStruct<PointT> object(1);
        pcl::PointCloud<PointT>::Ptr bound (new pcl::PointCloud<PointT>);

        for ( size_t i = 0; i < plane_vec.size(); ++i ){
        // for ( size_t i = 1; i < 2; ++i ){
            std::cout << "hmm1" << std::endl;
            QTD::QuadTreePCL<PointT> qtpcl(1,10,0,0);
            // qtpcl.setMaxLevel(10);
            qtpcl.setMaxWidth(1);
            qtpcl.setNormal(normal_vec[i]);

            std::cout << "hmm2" << std::endl;
            PointCloudT::Ptr out (new PointCloudT());
            std::vector< pcl::Vertices > vertices;
            qtpcl.insertBoundary(hulls[i]);
            std::cout << "hmm2.5" << std::endl;
            qtpcl.createMeshNew<PointT>(out, vertices);

            std::cout << "hmm3" << std::endl;
            cv::Mat image;
            std::vector<Eigen::Vector2f> vertex_texture;
            qtpcl.createTexture<PointT>(plane_vec[i], out, image, vertex_texture);
            std::cout << "hmm4" << std::endl;

            object.clouds.push_back(out);
            object.polygons.push_back(vertices);
            object.images.push_back(image);
            object.texture_vertices.push_back(vertex_texture);
            object.coefficients.push_back(normal_vec[i]);
            std::cout << "hmm5" << std::endl;
            // qtpcl.createBoundary(bound);

        }
        QTD::saveOBJFile(save_path + "mesh.obj", object, 5);
        std::cout << "hmm6" << std::endl;

        PointCloudT::Ptr outCloudEfficientPPR( new PointCloudT() );
        planeEx.combinePlanes(plane_vec, outCloudEfficientPPR, true);

        PointCloudT::Ptr outCloudEfficientPPRnc( new PointCloudT() );
        planeEx.combinePlanes(plane_vec, outCloudEfficientPPRnc, false);

        // pcl::PCDWriter writer;
        // int R = 255; int G = 255;
        // int j = 0;
        // for(auto &clouds : hulls){
        //     int B = 0;
        //     for(int i = 0; i < clouds->size(); i++){
        //         clouds->at(i).r = R;
        //         clouds->at(i).g = G;
        //         clouds->at(i).b = (int)(255*i/clouds->size());
        //     }
        //     writer.write(save_path + "hulls_" + std::to_string(j) + ".pcd", *clouds);
        //     j++;
        // }
        // j = 0;
        // for(auto &clouds : plane_vec){
        //     writer.write(save_path + "clouds_" + std::to_string(j) + ".pcd", *clouds);
        //     j++;
        // }


        PointCloudT::Ptr hullsOut( new PointCloudT() );
        planeEx.combinePlanes(hulls, hullsOut, false);

        writer.write(save_path + "segmented.pcd", *outCloudEfficientPPR);
        writer.write(save_path + "segmented_nc.pcd", *outCloudEfficientPPRnc);

        // pcl::PCDWriter writer;
        // writer.write(save_path + "bound.pcd", *bound);

    }

};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    FusionCloud test;

    ros::Rate loop_rate(10);
    test.testFusionCloud();


    return 0;
}
