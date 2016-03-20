#include <ros/ros.h>
#include <quad_tree_decimation/plane_extraction/plane_extraction.h>
#include <quad_tree_decimation/quad_tree/quad_tree_pcl.h>
#include <quad_tree_decimation/quad_tree/quad_tree.h>
#include <quad_tree_decimation/point_types/surfel_type.h>
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
#include <algorithm>


// DEFINITIONS
#define NODE_NAME               "dataset_mesh"

using PointT = pcl::PointXYZRGB;
using PointCloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using PointS = SurfelType;
using PointCloudS = pcl::PointCloud<PointS>;

using namespace QTD::Parameters;
// using namespace EXX;

class DatasetMesh
{
public:
    ros::NodeHandle nh;
private:
    // ros::Publisher point_cloud_publisher;
    EXX::compression cmprs;
    QTD::planeExtraction planeEx;
    primitive_params params;

public:

    DatasetMesh()
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

    void createMesh(void){

        PointCloudS::Ptr surfel_cloud (new PointCloudS());

        std::string dataset_path = loadParam<std::string>("dataset_path", nh);
        std::string dataset_cloud = loadParam<std::string>("dataset_cloud", nh);
        std::string dataset_save = loadParam<std::string>("dataset_save", nh);
        std::string save_prefix = dataset_cloud.substr(0,dataset_cloud.find_last_of('/'));
        std::cout << save_prefix << std::endl;
        std::replace(save_prefix.begin(), save_prefix.end(), '/', '_');

        std::cout << save_prefix << std::endl;

        PointCloudT::Ptr segment (new PointCloudT());
        // PointCloudT::Ptr segment_v (new PointCloudT());
        NormalCloudT::Ptr normals (new NormalCloudT());
        pcl::PCDReader reader;
        reader.read (dataset_path + dataset_cloud, *surfel_cloud);
        // cmprs.voxelGridFilter(segment, segment_v);


        std::cout << "surfel_cloud: " << surfel_cloud->points.size() << std::endl;

        // create regular pointcloud and normal cloud.
        // normals->reserve(surfel_cloud->size());
        // segment->reserve(surfel_cloud->size());
        for(auto p : surfel_cloud->points){
            if(p.confidence >= 0.3){
                PointT pt;
                NormalT pn;
                pt.x = p.x;
                pt.y = p.y;
                pt.z = p.z;
                pt.r = (p.rgba >> 16) & 0x0000ff  ;
                pt.g = (p.rgba >> 8)  & 0x0000ff;
                pt.b = (p.rgba)       & 0x0000ff;
                pn.normal_x = p.normal_x;
                pn.normal_y = p.normal_y;
                pn.normal_z = p.normal_z;
                normals->push_back(pn);
                segment->push_back(pt);
            }
        }


        ////////////////////////////////////////////////////////////////////////
        // EFFICIENT RANSAC with PPR
        ////////////////////////////////////////////////////////////////////////

        PointCloudT::Ptr nonPlanar (new PointCloudT());

        std::vector<PointCloudT::Ptr> plane_vec;
        std::vector<pcl::ModelCoefficients::Ptr> normal_vec;

        std::cout << "Efficient PPR..................." << std::endl;
        // planeDetection::planeSegmentationEfficientPPR(segment, params, plane_vec, normal_vec, nonPlanar);
        std::cout << "segment size: " << segment->points.size() << std::endl;

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
        // PROJECT TO PLANE


        planeEx.projectToPlane<PointT>( plane_vec, normal_vec );
        planeEx.mergePlanes<PointT>( plane_vec, normal_vec );
        planeEx.projectToPlane<PointT>( plane_vec, normal_vec );


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
        for ( size_t i = 0; i < plane_vec.size(); ++i ){
        // for ( size_t i = 0; i < 2; ++i ){

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

        }
        QTD::saveOBJFile(dataset_save + save_prefix + "_mesh.obj", object, 5);


        PointCloudT::Ptr outCloudEfficientPPR( new PointCloudT() );
        planeEx.combinePlanes(plane_vec, outCloudEfficientPPR, true);

        PointCloudT::Ptr outCloudEfficientPPRnc( new PointCloudT() );
        planeEx.combinePlanes(plane_vec, outCloudEfficientPPRnc, false);

        pcl::PCDWriter writer;
        writer.write(dataset_save + save_prefix + "_segmented.pcd", *outCloudEfficientPPR);
        writer.write(dataset_save + save_prefix + "_segmented_nc.pcd", *outCloudEfficientPPRnc);

    }


};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    DatasetMesh dMesh;

    ros::Rate loop_rate(10);
    dMesh.createMesh();
    return 0;
}
