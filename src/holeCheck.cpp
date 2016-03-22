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
#define NODE_NAME               "hole_cloud_mesh"

using PointT = pcl::PointXYZRGB;
using PointCloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;


class FusionCloud
{
public:
    ros::NodeHandle nh;
private:
    // ros::Publisher point_cloud_publisher;

public:

    FusionCloud()
    {
        nh = ros::NodeHandle("~");
        // std::cout << "leaf size: " << cmprs.getVoxelLeafSize() << std::endl;
    }

    bool combineSegments(const pcl::PointCloud<PointT>::Ptr a, const pcl::PointCloud<PointT>::Ptr b, pcl::PointCloud<PointT>::Ptr out, float dist){

        if(out->size() != 0)  out->clear();

        float distance_threshold = dist*dist;

        auto squared_point_distance = [distance_threshold](PointT a, PointT b){
            return std::pow((b.x-a.x),2) + std::pow((b.y-a.y),2) + std::pow((b.z-a.z),2) < distance_threshold;
        };

        // Need to check if tmp_segment connects to new segment
        if(squared_point_distance( a->back(), b->back())){
            // Need to reverse the new boundary and add to segment.
            std::reverse(b->begin(), b->end());
            *out += *a;
            *out += *b;
            return true;

        } else if( squared_point_distance( a->front(), b->back()) ){
            std::reverse(a->begin(), a->end());
            *out += *b;
            *out += *a;
            return true;

        }
        else if( squared_point_distance( a->front(), b->front()) ){
            std::reverse(a->begin(), a->end());
            *out += *a;
            *out += *b;
            return true;
        }

        else return false;
    }


    bool holeCheck(const typename pcl::PointCloud<PointT>::Ptr boundary, std::vector<pcl::PointCloud<PointT>::Ptr> &newBoundary, float dist){
        // Do a very simple distance check and return indices to first point after split.

        float distance_threshold = dist*dist;

        // Returns true if the distance between a and b is less than dist_threshold.
        // return |b-a| < dist_threshold.
        auto squared_point_distance = [distance_threshold](PointT a, PointT b){
            return std::pow((b.x-a.x),2) + std::pow((b.y-a.y),2) + std::pow((b.z-a.z),2) < distance_threshold;
        };

        auto squared_point_distance_value = [distance_threshold](PointT a, PointT b){
            return std::sqrt(std::pow((b.x-a.x),2) + std::pow((b.y-a.y),2) + std::pow((b.z-a.z),2));
        };

        int start = 0;
        bool has_left_origin = false;

        // pcl::PointCloud<PointT>::Ptr tmp (new pcl::PointCloud<PointT>());
        newBoundary.push_back(pcl::PointCloud<PointT>::Ptr (new pcl::PointCloud<PointT>()));
        pcl::PointCloud<PointT>::Ptr tmp_segment (new pcl::PointCloud<PointT>());
        for(size_t i = 0; i < boundary->size(); ++i){

            if( i == boundary->size() - 1){
                // Need to close last boundary.

                newBoundary.back()->push_back(boundary->at(i));

                if(tmp_segment->size() != 0){
                    std::cout << "ahaa" << std::endl;
                    pcl::PointCloud<PointT>::Ptr tmp_combined (new pcl::PointCloud<PointT>());
                    if(combineSegments(newBoundary.back(), tmp_segment, tmp_combined, dist)){
                        std::cout << "hihi" << std::endl;
                        *newBoundary.back() = *tmp_combined;

                    }
                }
                newBoundary.back()->push_back(newBoundary.back()->points.front());
                return true;
            }

            else if( newBoundary.back()->size() == 0 ){
                newBoundary.back()->push_back(boundary->at(i));
            }

            else {
                // Check if distance between current and next is to large
                if( !squared_point_distance(boundary->at(i), boundary->at(i+1)) ){
                    // Step to big, still need to check if circle has been closed => close to starting point
                    if( squared_point_distance(boundary->at(i), newBoundary.back()->points.front()) ){
                        // Can close circle.
                        newBoundary.back()->push_back(boundary->at(i));
                        newBoundary.back()->push_back(newBoundary.back()->points.front());
                        start = i+1;
                        newBoundary.push_back(pcl::PointCloud<PointT>::Ptr (new pcl::PointCloud<PointT>()));
                        // newBoundary.back()->clear();

                        // has_left_origin = false;
                    } else if( squared_point_distance(boundary->at(i+1), newBoundary.back()->points.front()) ){
                        // have a stupid ordering of points
                        newBoundary.back()->push_back(boundary->at(i));
                        std::reverse(newBoundary.back()->points.begin(), newBoundary.back()->points.end());
                    } else {
                        newBoundary.back()->push_back(boundary->at(i));
                        if(tmp_segment->size() == 0){
                            *tmp_segment = *newBoundary.back();
                            newBoundary.back()->clear();
                        } else {
                            // Need to check if tmp_segment connects to new segment
                            pcl::PointCloud<PointT>::Ptr tmp_combined (new pcl::PointCloud<PointT>());
                            if(combineSegments(newBoundary.back(), tmp_segment, tmp_combined, dist)){
                                *tmp_segment = *tmp_combined;
                                std::cout << "muhahahah" << std::endl;
                                newBoundary.back()->clear();

                                // Check if we have closed boundary
                                if( squared_point_distance(tmp_segment->front(), tmp_segment->back()) ){
                                    // have closed boundary
                                    *newBoundary.back() = *tmp_segment;
                                    newBoundary.push_back(pcl::PointCloud<PointT>::Ptr (new pcl::PointCloud<PointT>()));
                                }
                            }
                        }
                    }
                } else {
                    newBoundary.back()->push_back(boundary->at(i));
                }

            }
        }
    }




    void testFusionCloud(void){


        PointCloudT::Ptr segment (new PointCloudT());
        PointCloudT::Ptr cloud (new PointCloudT());
        PointCloudT::Ptr cloud_filtered (new PointCloudT());

        pcl::PCDReader reader;
        reader.read ("/home/unnar/catkin_ws/src/quad_tree_decimation_examples/data/dataset_fusion/w5r16/hulls_11.pcd", *segment);

        reader.read ("/home/unnar/catkin_ws/src/quad_tree_decimation_examples/data/dataset_fusion/w5r16/clouds_11.pcd", *cloud);


        pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
        // build the filter
        outrem.setInputCloud(cloud);
        outrem.setRadiusSearch(0.05);
        outrem.setMinNeighborsInRadius (15);
        // apply filter
        outrem.filter (*cloud);


        std::cout << segment->size() << std::endl;

        std::vector<int> v;
        std::vector<pcl::PointCloud<PointT>::Ptr> newBoundary;
        holeCheck(segment, newBoundary, 0.4);

        int R = 255;
        int G = 255;
        int B = 0;


        pcl::PointCloud<PointT>::Ptr bound (new pcl::PointCloud<PointT>());
        for(int i = 0; i < newBoundary.size(); ++i){
            std::cout << "yeahhhh" << std::endl;
            if(newBoundary[i]->size() < 20) continue;
            int j = 0;
            for(auto &p : newBoundary[i]->points){
                // std::cout << "hmmmmmm" << std::endl;
                p.r = R;
                p.g = 255;
                p.b = j*255/newBoundary[i]->size();
                j++;
            }
            *bound += *newBoundary[i];
        }

        // auto split = v.begin();
        // for(int i = 0; i < segment->size(); i++){
        //     if(i == *split){
        //         split++;
        //         B += 50;
        //     }
        //     segment->at(i).r = R;
        //     segment->at(i).r = G;
        //     segment->at(i).r = B;
        // }

        pcl::PCDWriter writer;
        writer.write("/home/unnar/catkin_ws/src/quad_tree_decimation_examples/data/holeCheck/holes_11.pcd", *bound);
        writer.write("/home/unnar/catkin_ws/src/quad_tree_decimation_examples/data/holeCheck/clouds_11.pcd", *cloud);

    }

};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    FusionCloud test;

    ros::Rate loop_rate(10);
    test.testFusionCloud();


    return 0;
}
