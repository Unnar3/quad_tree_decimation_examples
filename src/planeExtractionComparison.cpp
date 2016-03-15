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
        std::string save_path = loadParam<std::string>("save_path_comparison", nh);

        // find variance from name
        std::string var = test_cloud_name.substr(test_cloud_name.find_last_of("_")+1,3);
        std::string norm = std::to_string(params.normal_neigbourhood);
        norm = norm.substr(norm.find_last_of(".")+1,3);

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
                if(count < 7){
                    record.push_back( std::stoi(s) );
                }
                else{
                    // std::cout << "s: " << s << std::endl;
                    data_f.push_back( std::stof(s) );
                }
                count++;
            }

            data.push_back( record );
        }
        if (!infile.eof())
        {
            std::cerr << "Fooey!\n";
            // std::cout << data.size() << std::endl;
            // std::cout << test_cloud + txt_name << std::endl;
        }
        // for (auto p : data_f){
        //     std::cout << p << std::endl;
        // }
        // // exit(0);


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
        planeEx.planeSegmentationEfficient(segment, normals, plane_vec_efficient_ppr, normal_vec_efficient_ppr, nonPlanar);
        // PROJECT TO PLANE

        for ( size_t i = 0; i < normal_vec_efficient_ppr.size(); ++i ){
            EXX::compression::projectToPlaneS( plane_vec_efficient_ppr[i], normal_vec_efficient_ppr[i] );
        }

        PointCloudT::Ptr error_cloud_EfficientPPR (new PointCloudT);
        int incorrectPPR = 0;
        float average_distPPR = 0;
        getErrorCloud(plane_vec_efficient_ppr, normal_vec_efficient_ppr, nonPlanar, data, data_f, error_cloud_EfficientPPR, incorrectPPR, average_distPPR);

        PointCloudT::Ptr outCloudEfficientPPR( new PointCloudT() );
        planeEx.combinePlanes(plane_vec_efficient_ppr, outCloudEfficientPPR, true);

        std::ofstream fsPPR;
        std::string PPRtxt = save_path + "outCloudEfficientPPR_" + var + "_" + norm +".txt";
        fsPPR.open (PPRtxt.c_str ());
        fsPPR << "Efficient PPR" << std::endl;
        fsPPR << "Number of points: " << segment->size() << std::endl;
        fsPPR << "Number of planes: " << plane_vec_efficient_ppr.size() << std::endl;
        fsPPR << "Planar points: " << outCloudEfficientPPR->points.size() << std::endl;
        fsPPR << "NonPlanar points: " << nonPlanar->points.size() << std::endl;
        fsPPR << "Combined: " << outCloudEfficientPPR->points.size() + nonPlanar->points.size() << std::endl;
        fsPPR << "Incorrect: " << incorrectPPR << std::endl;
        fsPPR << "Average distance: " << average_distPPR << std::endl;
        fsPPR.close();

        std::cout << "Extracted " << plane_vec_efficient_ppr.size() << "  planes, Efficient PPR" << std::endl;
        std::cout << "Planar points: " << outCloudEfficientPPR->points.size() << std::endl;
        std::cout << "Non planar points: " << nonPlanar->points.size() << std::endl;
        std::cout << "Combined: " << outCloudEfficientPPR->points.size() + nonPlanar->points.size() << std::endl;
        std::cout << "incorrect: " << incorrectPPR << std::endl;
        std::cout << "average distance: " << average_distPPR << std::endl;
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
        int incorrectEPPR = 0;
        float average_distEPPR = 0;
        getErrorCloud(plane_vec_efficient, normal_vec_efficient, nonPlanar_efficient, data, data_f, error_cloud_Efficient,  incorrectEPPR, average_distEPPR);

        PointCloudT::Ptr outCloudEfficient( new PointCloudT() );
        planeEx.combinePlanes(plane_vec_efficient, outCloudEfficient, true);
        std::cout << "Extracted " << normal_vec_efficient.size() << "  planes, Efficient" << std::endl;
        std::cout << "Planar points: " << outCloudEfficient->points.size() << std::endl;
        std::cout << "Non planar points: " << nonPlanar_efficient->points.size() << std::endl;
        std::cout << "Combined: " << outCloudEfficient->points.size() + nonPlanar_efficient->points.size() << std::endl;
        std::cout << "incorrect: " << incorrectEPPR << std::endl;
        std::cout << "average distance: " << average_distEPPR << std::endl;
        std::cout << " " << std::endl;

        std::ofstream fsEPPR;
        std::string EPPRtxt = save_path + "outCloudEfficient_" + var + "_" + norm +".txt";
        fsEPPR.open (EPPRtxt.c_str ());
        fsEPPR << "Efficient" << std::endl;
        fsEPPR << "Number of points: " << segment->size() << std::endl;
        fsEPPR << "Number of planes: " << plane_vec_efficient.size() << std::endl;
        fsEPPR << "Planar points: " << outCloudEfficient->points.size() << std::endl;
        fsEPPR << "NonPlanar points: " << nonPlanar_efficient->points.size() << std::endl;
        fsEPPR << "Combined: " << outCloudEfficient->points.size() + nonPlanar_efficient->points.size() << std::endl;
        fsEPPR << "Incorrect: " << incorrectEPPR << std::endl;
        fsEPPR << "Average distance: " << average_distEPPR << std::endl;
        fsEPPR.close();

        pcl::PCDWriter writer;
        writer.write(save_path + "outCloudEfficientPPR_" + var + "_" + norm +".pcd", *outCloudEfficientPPR);
        writer.write(save_path + "outCloudEfficient_" + var + "_" + norm +".pcd", *outCloudEfficient);
        writer.write(save_path + "errored_cloud_EfficientPPR_" + var + "_" + norm +".pcd", *error_cloud_EfficientPPR);
        writer.write(save_path + "errored_cloud_Efficient_" + var + "_" + norm +".pcd", *error_cloud_Efficient);

    }

    void getErrorCloud(
            std::vector<PointCloudT::Ptr> plane_vec,
            std::vector<pcl::ModelCoefficients::Ptr> normal_vec,
            PointCloudT::Ptr nonPlanar,
            std::vector <std::vector <int> > data,
            std::vector <float > data_f,
            PointCloudT::Ptr error_cloud,
            int &incorrect,
            float &average_dist){


        int countAverage = 0;

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
            vec[3] = -data_f[idx];
            for(int j = 0; j < plane_vec[i]->size(); ++j){
                if(plane_vec[i]->at(j).r == r && plane_vec[i]->at(j).g == g && plane_vec[i]->at(j).b == b ){
                    float dist = pcl::pointToPlaneDistance( plane_vec[i]->at(j), vec);
                    average_dist += dist;
                    countAverage++;
                    int R,G,B;
                    gradientColor(34,167,240,247,202,24,R,G,B,dist,0.05);

                    plane_vec[i]->at(j).r = R;
                    plane_vec[i]->at(j).g = G;
                    plane_vec[i]->at(j).b = B;
                } else {
                    incorrect++;
                    plane_vec[i]->at(j).r = 239;
                    plane_vec[i]->at(j).g = 72;
                    plane_vec[i]->at(j).b = 54;
                }

            }
        }

        // PointCloudT::Ptr errored_cloud (new PointCloudT);
        for(auto c : plane_vec){
            *error_cloud += *c;
        }

        for(auto &p : nonPlanar->points){
            p.r = 255;
            p.g = 255;
            p.b = 255;
        }
        *error_cloud += *nonPlanar;
        average_dist /= (float)countAverage;

    }

    void gradientColor(int startR, int startG, int startB,
                        int endR, int endG, int endB,
                        int &outR, int &outG, int &outB,
                        float dist, float maxDist){

        float value = std::min(dist, maxDist) / maxDist;

        outR =  startR + (int)((endR-startR)*value);
        outG =  startG + (int)((endG-startG)*value);
        outB =  startB + (int)((endB-startB)*value);
    }


};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    Comparison test;

    ros::Rate loop_rate(10);
    test.testComparison();

    return 0;
}
