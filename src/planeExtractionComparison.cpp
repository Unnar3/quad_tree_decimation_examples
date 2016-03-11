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
        std::string save_path = loadParam<std::string>("save_path_comparison", nh);

        // find variance from name
        std::string var = test_cloud_name.substr(test_cloud_name.find_last_of("_")+1,3);


        // read appropriate txt file;
        std::string txt_name = test_cloud_name.substr(0,test_cloud_name.find_last_of(".")) + ".txt";

        std::vector <std::vector <int> > data;
        std::vector <float > data_f;
        std::ifstream infile( test_cloud + txt_name );
        while (infile)
        {
            std::string s;
            if (!std::getline( infile, s )) break;

            std::istringstream ss( s );
            std::vector <int> record;
            int count = 0;
            while (ss)
            {
                std::string s;
                if (!getline( ss, s, ';' )) break;
                if(count < 6){
                    record.push_back( std::stoi(s) );
                }
                else
                    data_f.push_back( std::stof(s) );
            }

            data.push_back( record );
        }
        if (!infile.eof())
        {
            std::cerr << "Fooey!\n";
            // std::cout << data.size() << std::endl;
            // std::cout << test_cloud + txt_name << std::endl;
        }


        PointCloudT::Ptr segment (new PointCloudT());
        NormalCloudT::Ptr normals (new NormalCloudT());

        pcl::PCDReader reader;
        reader.read (test_cloud + test_cloud_name, *segment);

        std::cout << "segment: " << segment->points.size() << std::endl;

        std::vector<PointCloudT::Ptr> plane_vec;
        std::vector<Eigen::Vector4d> normal_vec;

        // create a kd-tree from the original data
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(segment);


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

        PointCloudT::Ptr error_cloud_EfficientPPR (new PointCloudT);
        getErrorCloud(plane_vec_efficient_ppr, normal_vec_efficient_ppr, data, data_f, error_cloud_EfficientPPR);

        PointCloudT::Ptr outCloudEfficientPPR( new PointCloudT() );
        planeEx.combinePlanes(plane_vec_efficient_ppr, outCloudEfficientPPR, true);

        std::cout << "Extracted " << plane_vec_efficient_ppr.size() << "  planes, Efficient PPR" << std::endl;
        std::cout << "Planar points: " << outCloudEfficientPPR->points.size() << std::endl;
        std::cout << "Non planar points: " << nonPlanar->points.size() << std::endl;
        std::cout << "Combined: " << outCloudEfficientPPR->points.size() + nonPlanar->points.size() << std::endl;
        std::cout << " " << std::endl;

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

        PointCloudT::Ptr error_cloud_Efficient (new PointCloudT);
        getErrorCloud(plane_vec_efficient, normal_vec_efficient, data, data_f, error_cloud_Efficient);

        PointCloudT::Ptr outCloudEfficient( new PointCloudT() );
        planeEx.combinePlanes(plane_vec_efficient, outCloudEfficient, true);
        std::cout << "Extracted " << normal_vec_efficient.size() << "  planes, Efficient" << std::endl;
        std::cout << "Planar points: " << outCloudEfficient->points.size() << std::endl;
        std::cout << "Non planar points: " << nonPlanar_efficient->points.size() << std::endl;
        std::cout << "Combined: " << outCloudEfficient->points.size() + nonPlanar_efficient->points.size() << std::endl;
        std::cout << " " << std::endl;

        pcl::PCDWriter writer;
        writer.write(save_path + "outCloudEfficientPPR_" + var + ".pcd", *outCloudEfficientPPR);
        writer.write(save_path + "outCloudEfficient_" + var + ".pcd", *outCloudEfficient);
        writer.write(save_path + "errored_cloud_EfficientPPR_" + var + ".pcd", *error_cloud_EfficientPPR);
        writer.write(save_path + "errored_cloud_Efficient_" + var + ".pcd", *error_cloud_Efficient);

    }

    void getErrorCloud(
            std::vector<PointCloudT::Ptr> plane_vec,
            std::vector<pcl::ModelCoefficients::Ptr> normal_vec,
            std::vector <std::vector <int> > data,
            std::vector <float > data_f,
            PointCloudT::Ptr error_cloud){


        // K nearest neighbor search
        for(int i = 0; i < plane_vec.size(); ++i){
        // for(int i = 0; i < 1; ++i){
            std::vector<int> count(data.size());
            for(int j = 0; j < plane_vec[i]->size(); ++j){

                // compare color to data vector to find correct plane
                // find index of same color
                int r = plane_vec[i]->at(j).r;
                int g = plane_vec[i]->at(j).g;
                int b = plane_vec[i]->at(j).b;

                for(int k = 0; k < data.size(); ++k){
                    if(r == data[k][4] && g == data[k][5] && b == data[k][6]){
                        count[k]++;
                        break;
                    }
                }

            }
            // belongs to plane max(count)
            auto result = std::max_element(count.begin(), count.end());
            int idx = std::distance(count.begin(), result);
            std::cout << "Belongs to plane " << idx << std::endl;
            std::cout << "real normal: " << data[idx][1] << ", " << data[idx][2] << ", " << data[idx][3] << std::endl;
            std::cout << "estimated normal: " << normal_vec[i]->values[0] << ", " << normal_vec[i]->values[1] << ", " << normal_vec[i]->values[2] << std::endl;

            // loop through it again and color based on correctly classified
            int r = data[idx][4];
            int g = data[idx][5];
            int b = data[idx][6];
            Eigen::Vector4f vec;

            // TODO:
            // vec should be from data, data is missing d component, add

            vec[0] = data[idx][1];
            vec[1] = data[idx][2];
            vec[2] = data[idx][3];
            vec[3] = data_f[idx];
            for(int j = 0; j < plane_vec[i]->size(); ++j){
                if(plane_vec[i]->at(j).r == r && plane_vec[i]->at(j).g == g && plane_vec[i]->at(j).b == b ){
                    float dist = pcl::pointToPlaneDistance( plane_vec[i]->at(j), vec);
                    // std::cout << "dist: " << dist << std::endl;
                    plane_vec[i]->at(j).r = 255;
                    plane_vec[i]->at(j).g = 255;
                    plane_vec[i]->at(j).b = 255-std::min(int(dist*100),255);
                } else {
                    plane_vec[i]->at(j).r = 0;
                    plane_vec[i]->at(j).g = 0;
                    plane_vec[i]->at(j).b = 255;
                }

            }
        }

        // PointCloudT::Ptr errored_cloud (new PointCloudT);
        for(auto c : plane_vec){
            *error_cloud += *c;
        }
    }


};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    Comparison test;

    ros::Rate loop_rate(10);
    test.testComparison();

    return 0;
}
