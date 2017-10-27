#include <cpu_tsdf/tsdf_volume_octree.h>
#include <cpu_tsdf/marching_cubes_tsdf_octree.h>

#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cpu_tsdf;

int width = 640;
int height = 480;
double fx = 585;
double fy = 585;
double cx = 320;
double cy = 240;

bool reprojectPoint(const pcl::PointXYZRGB &pt, int &u, int &v)
{
  u = (pt.x * fx / pt.z) + cx;
  v = (pt.y * fy / pt.z) + cy;
  return (!pcl_isnan(pt.z) && pt.z > 0 && u >= 0 && u < width && v >= 0 && v < height);
}

int main()
{
    pcl::visualization::CloudViewer viewer("viewer");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr global(new pcl::PointCloud<pcl::PointXYZRGB>);
    TSDFVolumeOctree::Ptr tsdf(new TSDFVolumeOctree);
    tsdf->setGridSize(12.0 ,12.0, 12.0);      // 10m x 10m x 10m
    tsdf->setResolution(2000, 2000, 2000); // Smallest cell size = 10m / 2048 = about half a centimeter
    tsdf->setImageSize(width, height);
    tsdf->setCameraIntrinsics(fx, fy, cx, cy);
    tsdf->setNumRandomSplts(1);
    tsdf->setSensorDistanceBounds(0, 3.0);
    tsdf->setIntegrateColor(true); // Set to true if you want the TSDF to store color
    tsdf->setDepthTruncationLimits(0.03, 0.03);    
    tsdf->reset();
    std::vector<Eigen::Affine3d> poses;
    for (int i = 0; i < 50; i++)
    {
        std::cout << "*********DEAL IMAGE NUM:" << i << "*********" << std::endl;
        std::stringstream ssd;
        ssd << "../data/rgbd-frames/frame-" << std::setw(6) << std::setfill('0') << i+150 << ".depth.png";
        std::stringstream ssc;
        ssc << "../data/rgbd-frames/frame-" << std::setw(6) << std::setfill('0') << i+150 << ".color.png";
        std::stringstream ssp;
        ssp << "../data/rgbd-frames/frame-" << std::setw(6) << std::setfill('0') << i+150 << ".pose.txt";
        cv::Mat depth = cv::imread(ssd.str().c_str(), IMREAD_UNCHANGED);
        cv::Mat color = cv::imread(ssc.str().c_str(), IMREAD_COLOR);

        ifstream f (ssp.str().c_str());
        float v;
        Eigen::Matrix4d mat;
        mat (3,0) = 0; mat (3,1) = 0; mat (3,2) = 0; mat (3,3) = 1;
        for (int y = 0; y < 3; y++)
        {
          for (int x = 0; x < 4; x++)
          {
            f >> v;
            mat (y,x) = static_cast<double> (v);
          }
        }
        f.close ();
        poses.push_back(Eigen::Affine3d());
        poses[i]=mat;
        std::cout<<"fram "<<i<<" pose is："<<std::endl;
        std::cout<<poses[i].matrix()<<std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);


        for (int r = 0; r < height; r++)
        {
          for (int c = 0; c < width; c++)
          {
            pcl::PointXYZRGB p;
    
            ushort d = depth.at<ushort>(r, c);
            if (d == 0)
            {
              continue;
            }
            p.z = double(d) / 1000;
            p.x = (c - cx) * p.z / fx;
            p.y = (r - cy) * p.z / fy;
            p.r = color.at<cv::Vec3b>(r, c)[2];
            p.g = color.at<cv::Vec3b>(r, c)[1];
            p.b = color.at<cv::Vec3b>(r, c)[0];
            cloud->points.push_back(p);
          }
        }
    
        cloud->width = 1;
        cloud->height = cloud->points.size();

        viewer.showCloud(cloud);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_organized (new pcl::PointCloud<pcl::PointXYZRGB> (width, height));

        std::cout<<"Make organized"<<std::endl;
        size_t nonnan_original = 0;
        size_t nonnan_new = 0;
        float min_x = std::numeric_limits<float>::infinity ();
        float min_y = std::numeric_limits<float>::infinity ();
        float min_z = std::numeric_limits<float>::infinity ();
        float max_x = -std::numeric_limits<float>::infinity ();
        float max_y = -std::numeric_limits<float>::infinity ();
        float max_z = -std::numeric_limits<float>::infinity ();
        for (size_t j = 0; j < cloud_organized->size (); j++)
          cloud_organized->at (j).z = std::numeric_limits<float>::quiet_NaN ();
        for (size_t j = 0; j < cloud->size (); j++)
        {
          const pcl::PointXYZRGB &pt = cloud->at (j);
          int u, v;
          if (reprojectPoint (pt, u, v))
          {
            pcl::PointXYZRGB &pt_old = (*cloud_organized) (u, v);
            if (pcl_isnan (pt_old.z) || (pt_old.z > pt.z))
            {
              pt_old = pt;
            }
          }
        }

        std::cout << "frame " << i << " get  point :" << cloud->points.size() << std::endl;
        // Initialize it to be empty
        Eigen::Affine3d pose_rel_to_first_frame =  poses[0].inverse () * poses[i];
        std::cout<<"frame. "<<i<<" to first pose is："<<std::endl;
        std::cout<<pose_rel_to_first_frame.matrix()<<std::endl;
        tsdf->integrateCloud(*cloud_organized, pcl::PointCloud<pcl::Normal>(), pose_rel_to_first_frame); // Integrate the cloud
    }
    // Note, the normals aren't being used in the default settings. Feel free to pass in an empty cloud
    // Now what do you want to do with it?

    MarchingCubesTSDFOctree mc;
    mc.setInputTSDF(tsdf);
    mc.setMinWeight(0);     // Sets the minimum weight -- i.e. if a voxel sees a point less than 2 times, it will not render  a mesh triangle at that location
    mc.setColorByRGB(true); // If true, tries to use the RGB values of the TSDF for meshing -- required if you want a colored mesh
    pcl::PolygonMesh mesh;
    mc.reconstruct(mesh);
    pcl::io::savePLYFile("mesh.ply", mesh);
    std::cout << "Finish" << std::endl;
    return (0);
}
