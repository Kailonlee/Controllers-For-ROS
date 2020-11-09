#ifndef MPCCONTROLLERROSSIMPLE_H
#define MPCCONTROLLERROSSIMPLE_H
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include "MPCController.h"
#include "base_control/WheelSpeed.h"
#include "mpc_controller/TrajectoryPoint.h"

// transforms
#include <tf2/utils.h>
#include <tf2_ros/buffer.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include "mpc_controller/MPCPlan.h"
#include "mpc_controller/Trajectory.h"
#include <dwa_local_planner/ControlErrorStamped.h>
namespace mpc_controller {
    enum Status {INITIAL, RUNNING, BREAK, END, ERROR};

class MPCControllerROSSimple {
   public:
    MPCControllerROSSimple();
    ~MPCControllerROSSimple();
    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);
    bool setPlan(const mpc_controller::Trajectory& plan);
    bool computeVelocityCommands(const mpc_controller::pose2D& current_pose,
                                 const base_control::WheelSpeed& current_enc,
                                 const mpc_controller::slip_param& current_slip, 
                                 const geometry_msgs::Vector3 &body_vel,
                                 const double delta_t,
                                 base_control::WheelSpeed& cmd_vel);
    bool isGoalReached(const mpc_controller::pose2D& current_pose, const double tolarence);

   private:
    Status mpc_status_;
    ros::NodeHandle nh_;
    ros::NodeHandle pri_nh_;

    ros::Subscriber sub_odom_;
    ros::Subscriber sub_plan_;
    ros::Subscriber sub_enc_;
    ros::Subscriber sub_body_vel_;
    ros::Publisher pub_cmd_;
    ros::Publisher pub_err_;
    ros::Publisher pub_gui_plan_;
    ros::Publisher pub_control_err_;
    ros::ServiceServer mpc_plan_srv_;
    tf2_ros::Buffer* tf_;
    tf::StampedTransform transform_;
    tf::TransformListener listener;
    bool initialized_;
    mpc_controller::pose2D final_goal2D_;
    mpc_controller::slip_param current_slip_;

    double lat_error_, yaw_error_;

    Eigen::VectorXd control_output_vec_;
    base_control::WheelSpeed control_output_;
    base_control::WheelSpeed current_enc_;
    geometry_msgs::Vector3 body_vel_;

    std::string plan_frame_;

    ros::Time current_time_, last_time_;
    bool is_first_time_;

    std::shared_ptr<mpc_controller::SystemDynamicMatrix> sysdyn_;

    std::vector<mpc_controller::TrajectoryPoint> ref_traj_;
    std::shared_ptr<mpc_controller::MPC_Controller> mpc_;

    Eigen::VectorXd delta_u_lower_bound, delta_u_upper_bound;
    Eigen::VectorXd u_lower_bound, u_upper_bound;
    bool getRobotPose(mpc_controller::pose2D& current_pose);
    bool isInitialized();
    void computeSysDynamic(const mpc_controller::pose2D& current_pose,
                           const base_control::WheelSpeed& current_control_output,
                           const mpc_controller::slip_param& current_slip, const double T_sample,
                           std::shared_ptr<mpc_controller::SystemDynamicMatrix>& sysdyn_ptr);
    void odomCallback(const nav_msgs::Odometry::ConstPtr odom_msg);
    void encCallback(const base_control::WheelSpeed::ConstPtr enc_msg);
    void bodyVelCallback(const geometry_msgs::Vector3& body_vel_msg);
    bool planService(mpc_controller::MPCPlan::Request& req, mpc_controller::MPCPlan::Response& res);
    void stopControl();
    void planCallback(const mpc_controller::Trajectory& mpc_plan);
    void publishControlError();
};
}  // namespace mpc_controller
#endif