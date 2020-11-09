/*
 * @Date: 2020-07-19 16:04:43
 * @Description: test MPC_Controller
 * @LastEditors: Kailon
 * @LastEditTime: 2020-10-05 15:37:30
 */ 
#include <ros/ros.h>
#include "MPCController.h"

#include <string>
#include <fstream>
#include "tf/transform_datatypes.h"
#include "base_control/WheelSpeed.h"
#include <nav_msgs/Path.h>

const double wheel_radius_ = 0.135;
const double T_sample = 0.02;
const double width = 0.63;
const uint32_t state_dim = 3;
const uint32_t control_dim = 2;
const uint32_t NP = 10;
const uint32_t NC = 2;

auto current_slip_ptr_ = std::make_shared<mpc_controller::slip_param>();
bool ReadRefTrajectroy(std::vector<mpc_controller::TrajectoryPoint>* ref_path){
    std::ifstream inputfile;
    std::string txt_path("/home/kailon/桌面/ref_traj_ROS.txt");
    inputfile.open(txt_path, std::ios::in);
    if (!inputfile.is_open())
    {
        ROS_ERROR("CANNOT Open The File %s: ", txt_path.c_str());
        return false;
    }
    
    std::string unused;
    // std::getline(inputfile, unused);//清除第一行
    uint32_t line_num = 1;
    uint32_t max_line_num = 1025;
    double x, y, th, w_r, w_l;
    ROS_INFO("Reading Ref Path ...");
    std::string line_str;
    while(line_num < max_line_num){
        
        inputfile >> x >> y >> th >> w_r >> w_l;
        mpc_controller::TrajectoryPoint temp_ref_point;
        temp_ref_point.pose.position.x = x;
        temp_ref_point.pose.position.y = y;
        temp_ref_point.pose.orientation = tf::createQuaternionMsgFromYaw(th);
        temp_ref_point.velocity.right_vel.data = w_r;
        temp_ref_point.velocity.left_vel.data = w_l;
        (*ref_path).push_back(temp_ref_point);
        line_num ++;
        // ROS_INFO("x=%.2f, y=%.2f, th=%.2f, w_r=%.2f, w_l=%.2f", x,y,th,w_r, w_l);
    }
    ROS_INFO("Read Ref Path Done!");
    return true;
}

void TrackedRobotDynamic(const mpc_controller::pose2D &current_pose, 
                        const base_control::WheelSpeed& control_input, 
                        const double delta_t,
                        mpc_controller::pose2D* next_pose){    
    double w_l = control_input.left_vel.data;
    double w_r = control_input.right_vel.data;
    double cur_x = current_pose.x;
    double cur_y = current_pose.y;
    double cur_th = current_pose.theta;
    double v_lon = 0.5 * (w_l + w_r) * wheel_radius_;
    double v_lat = 0;
    double yaw_vel = (w_r - w_l) * wheel_radius_ / width;
    double v_x = v_lon * cos(cur_th) - v_lat * sin(cur_th);
    double v_y = v_lon * sin(cur_th) + v_lat * cos(cur_th);
    next_pose->x = v_x * delta_t + cur_x;
    next_pose->y = v_y * delta_t + cur_y;
    next_pose->theta = yaw_vel * delta_t + cur_th;

}
void FakeLocalization(const mpc_controller::pose2D & initial_pose, const geometry_msgs::PoseStampedPtr current_pose){
    // TODO
}

void ComputeSysDynamic(const mpc_controller::pose2D &current_pose,
                        const base_control::WheelSpeed &current_control_output,
                        mpc_controller::SystemDynamicMatrix *sysdyn_ptr){
    double x = current_pose.x;
    double y = current_pose.y;
    double th = current_pose.theta;

    double w_r = current_control_output.right_vel.data;
    double w_l = current_control_output.left_vel.data;

    double delta_l = current_slip_ptr_->delta_l;
    double delta_r = current_slip_ptr_->delta_r;
    double alpha = current_slip_ptr_->alpha;

    double C_l = wheel_radius_ * (1 - delta_l) / 2;
    double C_r = wheel_radius_ * (1 - delta_r) / 2;

    
    sysdyn_ptr->Matrix_a << 1, 0, T_sample * (- sin(th) - tan(alpha) * cos(th)) * (C_r * w_r + C_l * w_l),
                0, 1, T_sample * ( cos(th) - tan(alpha) * sin(th) ) * (C_r * w_r + C_l * w_l),
                0, 0, 1;

    sysdyn_ptr->Matrix_b << T_sample * C_r * ( cos(th) - tan(alpha) * sin(th) ),   T_sample * C_l * ( cos(th) - tan(alpha) * sin(th) ),
                T_sample * C_r * ( sin(th) + tan(alpha) * cos(th) ),   T_sample * C_l * ( sin(th) + tan(alpha) * cos(th) ),
                2 * T_sample * C_r / width, -2 * T_sample * C_l / width;

    sysdyn_ptr->Matrix_c = Eigen::MatrixXd::Identity(state_dim, state_dim);
}


int main(int argc, char** argv){
    ros::init(argc, argv, "mpc_controller_node");
    ros::NodeHandle nh;
    ros::Publisher real_path_pub = nh.advertise<nav_msgs::Path>("real_path", 10);
    ros::Publisher ref_path_pub = nh.advertise<nav_msgs::Path>("ref_path", 10);
    ros::Publisher output_control_pub = nh.advertise<base_control::WheelSpeed>("mpc_output", 10);
    mpc_controller::MPC_Controller mpc_test(state_dim, control_dim, NP, NC);
    mpc_controller::SystemDynamicMatrix sysdyn(state_dim, control_dim);
    
    Eigen::VectorXd delta_u_lower_bound, delta_u_upper_bound;
    delta_u_lower_bound.resize(control_dim);
    delta_u_upper_bound.resize(control_dim);
    delta_u_lower_bound << -2 , -2;
    delta_u_upper_bound << 2, 2;

    Eigen::VectorXd u_lower_bound, u_upper_bound;
    u_lower_bound.resize(control_dim);
    u_upper_bound.resize(control_dim);
    u_lower_bound << -20, -20;
    u_upper_bound << 35, 35;
    

    Eigen::Matrix<double, state_dim, state_dim> weight_Q;
    Eigen::Matrix<double, control_dim, control_dim> weight_R;
    
    // diag TODO
    weight_Q << 100, 0,  0,
                0,  100, 0,
                0,  0,  100;
    weight_R << 1, 0,
                0, 1;

    mpc_test.SetWeightQ(weight_Q);
    mpc_test.SetWeightR(weight_R);

    std::vector<geometry_msgs::PoseStamped> global_plan;
    std::vector<mpc_controller::TrajectoryPoint> ref_traj;

    if (!ReadRefTrajectroy(&ref_traj)) {return 1;}

    mpc_controller::pose2D initial_pose(0,1,0);
    mpc_controller::pose2D current_pose(initial_pose);
    mpc_controller::pose2D next_pose;

    Eigen::VectorXd output_control;
    base_control::WheelSpeed wheel_speed_output_control;

    double dt = 0.02; //s
    ros::Rate loop_rate( 1 / dt);

    nav_msgs::Path ref_path, real_path;

    ROS_INFO("Ready for MPC");
    while (ros::ok())
    {   
        ROS_INFO("Current Pose: x=%.2f, y=%.2f, th=%.2f", current_pose.x, current_pose.y, current_pose.theta);

        ComputeSysDynamic(current_pose, wheel_speed_output_control, &sysdyn);
        mpc_test.SetSysDynMatrix(sysdyn.Matrix_a, sysdyn.Matrix_b, sysdyn.Matrix_c);
        mpc_test.UpdateSystemMatrix(current_pose, ref_traj);
        mpc_test.SetDeltaUBound(delta_u_lower_bound, delta_u_upper_bound);
        mpc_test.SetUBound(u_lower_bound, u_upper_bound);
        mpc_test.SetConstraintMatrix();
        bool set_QP_data_success = mpc_test.SetQPData();
        if (set_QP_data_success)
        {
            ROS_INFO("set QP data success!");
            if (mpc_test.SolveQP())
            {
                ROS_INFO("solve QP success!");
                
                mpc_test.GetOutputControl(&output_control);
                wheel_speed_output_control.header.stamp = ros::Time::now();
                wheel_speed_output_control.right_vel.data = output_control[0];
                wheel_speed_output_control.left_vel.data = output_control[1]; 
                ROS_INFO("Current Control: w_r:%.2f, w_l=%.2f", wheel_speed_output_control.right_vel.data, wheel_speed_output_control.left_vel.data);

                output_control_pub.publish(wheel_speed_output_control);
            }
            else
            {
                ROS_ERROR("solve QP failed!");
                break;
            }
        }
        else
        {
            ROS_ERROR("Set QP data failed!");
            break;
        }

        real_path.header.stamp=ros::Time::now();
        real_path.header.frame_id="/base_link";
        geometry_msgs::PoseStamped real_pose_stamped;
        real_pose_stamped.header.stamp=ros::Time::now();
        real_pose_stamped.header.frame_id="/base_link";			
        real_pose_stamped.pose.position.x = current_pose.x;
        real_pose_stamped.pose.position.y = current_pose.y;
        real_pose_stamped.pose.position.z = 0;
        real_pose_stamped.pose.orientation = tf::createQuaternionMsgFromYaw(current_pose.theta);
        real_path.poses.push_back(real_pose_stamped);

        mpc_controller::pose2D target_pose;
        target_pose = mpc_test.GetTargetPose();

        ref_path.header.stamp=ros::Time::now();
        ref_path.header.frame_id="/base_link";
        geometry_msgs::PoseStamped ref_pose_stamped;
        ref_pose_stamped.header.stamp=ros::Time::now();
        ref_pose_stamped.header.frame_id="/base_link";			
        ref_pose_stamped.pose.position.x = target_pose.x;
        ref_pose_stamped.pose.position.y = target_pose.y;
        ref_pose_stamped.pose.position.z = 0;
        ref_pose_stamped.pose.orientation = tf::createQuaternionMsgFromYaw(target_pose.theta);
        ref_path.poses.push_back(ref_pose_stamped);

        real_path_pub.publish(real_path);
        ref_path_pub.publish(ref_path);

        if(mpc_test.isFinished()){
            ROS_INFO("=====================MPC Control Progress Finished!=====================");
            return 1;
        }

        TrackedRobotDynamic(current_pose, wheel_speed_output_control, dt, &next_pose);
        current_pose = next_pose;
        loop_rate.sleep();
    }
}