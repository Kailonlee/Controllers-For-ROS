#include "mpc_controller/MPCControllerROSSimple.h"

namespace mpc_controller {
const double wheel_radius_ = 0.105;
const double width = 0.63;
const uint32_t state_dim = 3;
const uint32_t control_dim = 2;
const uint32_t NP = 30;
const uint32_t NC = 3;

MPCControllerROSSimple::MPCControllerROSSimple() : initialized_(false), pri_nh_("~") {
    if (!isInitialized()) {
        sub_odom_ = nh_.subscribe("/odom", 1, &MPCControllerROSSimple::odomCallback, this);
        sub_plan_ = nh_.subscribe("/mpc_plan", 1, &MPCControllerROSSimple::planCallback, this);
        sub_enc_ = nh_.subscribe("/encoder", 1, &MPCControllerROSSimple::encCallback, this);
        sub_body_vel_ = nh_.subscribe("/odometry/body_vel", 1, &MPCControllerROSSimple::bodyVelCallback, this);
        pub_cmd_ = pri_nh_.advertise<base_control::WheelSpeed>("/base_control_vel", 1);
        pub_control_err_ = pri_nh_.advertise<dwa_local_planner::ControlErrorStamped>("/control_error", 1);
        pub_gui_plan_ = pri_nh_.advertise<nav_msgs::Path>("gui_plan", 1);
        mpc_plan_srv_ = nh_.advertiseService("mpc_plan", &MPCControllerROSSimple::planService, this);

        sysdyn_ = std::make_shared<mpc_controller::SystemDynamicMatrix>(state_dim, control_dim);

        delta_u_lower_bound.resize(control_dim);
        delta_u_upper_bound.resize(control_dim);
        delta_u_lower_bound << -2, -2;
        delta_u_upper_bound << 2, 2;
        u_lower_bound.resize(control_dim);
        u_upper_bound.resize(control_dim);
        u_lower_bound << -17, -17;
        u_upper_bound << 35, 35;

        Eigen::Matrix<double, state_dim, state_dim> weight_Q;
        Eigen::Matrix<double, control_dim, control_dim> weight_R;
        weight_Q << 1, 0, 0, 0, 1, 0, 0, 0, 1;
        weight_R << 0.05, 0, 0, 0.05;

        mpc_ = std::make_shared<mpc_controller::MPC_Controller>(state_dim, control_dim, NP, NC);
        mpc_->SetWeightQ(weight_Q);
        mpc_->SetWeightR(weight_R);
        mpc_status_ = INITIAL;
        initialized_ = true;
        ros::Rate loop_rate(50);
        while (ros::ok()) {
            ros::spinOnce();
            pose2D current_pose;
            base_control::WheelSpeed cmd_vel;
            double delta_t;
            switch (mpc_status_) {
                case INITIAL:
                    stopControl();
                    break;
                case RUNNING:
                    last_time_ = current_time_;
                    current_time_ = ros::Time::now();
                    if (is_first_time_) {
                        is_first_time_ = false;
                        continue;
                    }
                    if (!getRobotPose(current_pose)) {
                        stopControl();
                        mpc_status_ = ERROR;
                        ROS_ERROR("MPC Can not get robot pose");
                        continue;
                    }
                    if (isGoalReached(current_pose, 0.5)) {
                        mpc_status_ = END;
                        ROS_INFO("MPC has reach the Goal");
                        continue;
                    } else {
                        nav_msgs::Path gui_plan;
                        gui_plan.header.frame_id = plan_frame_;
                        gui_plan.header.stamp = current_time_;
                        for (size_t i = 0; i < ref_traj_.size(); i++) {
                            geometry_msgs::PoseStamped tmp;
                            tmp.pose = ref_traj_[i].pose;
                            gui_plan.poses.push_back(tmp);
                        }
                        pub_gui_plan_.publish(gui_plan);
                        delta_t = current_time_.toSec() - last_time_.toSec();
                        ROS_INFO("delta_t = %lf", delta_t);
                        computeVelocityCommands(current_pose, current_enc_, current_slip_, body_vel_, delta_t, cmd_vel);
                        publishControlError();
                        pub_cmd_.publish(cmd_vel);
                    }
                    break;
                case END:
                    //TODO
                    stopControl();
                    break;
                case ERROR:
                    //TODO
                    stopControl();
                    break;
                default:
                    break;
            }
            loop_rate.sleep();
        }
    } else {
        ROS_WARN("The Controller Has Already Been Initialized");
    }
}

MPCControllerROSSimple::~MPCControllerROSSimple() {}

bool MPCControllerROSSimple::setPlan(const mpc_controller::Trajectory& plan) {
    if (!isInitialized()) {
        ROS_ERROR("This Controller has not been initialized, please call initialize() before using this planner");
        return false;
    }
    ref_traj_.clear();
    final_goal2D_.x = plan.points.back().pose.position.x;
    final_goal2D_.y = plan.points.back().pose.position.y;
    plan_frame_ = plan.header.frame_id;
    ref_traj_ = plan.points;  // save plan;
    mpc_->SetPlan(ref_traj_);
    return true;
}

void MPCControllerROSSimple::publishControlError(){
    dwa_local_planner::ControlErrorStamped control_error_msg;
    control_error_msg.header.frame_id = "odom";
    control_error_msg.header.stamp = current_time_;
    control_error_msg.latitude_error.data = mpc_->ComputeLatErr();
    control_error_msg.heading_error.data = mpc_->ComputeYawErr();
    pub_control_err_.publish(control_error_msg);
}

bool MPCControllerROSSimple::computeVelocityCommands(const mpc_controller::pose2D& current_pose,
                                                     const base_control::WheelSpeed& current_enc,
                                                     const mpc_controller::slip_param& current_slip,
                                                     const geometry_msgs::Vector3 &body_vel,
                                                     const double delta_t, 
                                                     base_control::WheelSpeed& cmd_vel) {
    if (!isInitialized()) {
        ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
        return false;
    }
    computeSysDynamic(current_pose, current_enc, current_slip, delta_t, sysdyn_);
    mpc_->SetSysDynMatrix(sysdyn_->Matrix_a, sysdyn_->Matrix_b, sysdyn_->Matrix_c);
    mpc_->UpdateSystemMatrix(current_pose, body_vel, ref_traj_);
    mpc_->SetDeltaUBound(delta_u_lower_bound, delta_u_upper_bound);
    mpc_->SetUBound(u_lower_bound, u_upper_bound);
    mpc_->SetConstraintMatrix();

    if (mpc_->SetQPData()) {
        ROS_INFO("set QP data success!");
        if (mpc_->SolveQP()) {
            ROS_INFO("solve QP success!");

            mpc_->GetOutputControl(&control_output_vec_);
            cmd_vel.header.stamp = current_time_;
            cmd_vel.right_vel.data = control_output_vec_[0];
            cmd_vel.left_vel.data = control_output_vec_[1];
            if (cmd_vel.right_vel.data > 17.143)
            {
                cmd_vel.right_vel.data = 17.143;
                ROS_ERROR("right vel cmd is exceed the 17.143 limit");
            }
            if (cmd_vel.left_vel.data > 17.143)
            {
                cmd_vel.left_vel.data = 17.143;
                ROS_ERROR("left vel cmd is exceed the 17.143 limit");
            }
            ROS_INFO("Current Control: w_r:%.2f, w_l=%.2f", cmd_vel.right_vel.data, cmd_vel.left_vel.data);
            return true;
        } else {
            ROS_ERROR("solve QP failed!");
            stopControl();
            return false;
        }
    } else {
        ROS_ERROR("Set QP data failed!");
        stopControl();
        return false;
    }
}

bool MPCControllerROSSimple::isGoalReached(const mpc_controller::pose2D& current_pose, const double tolarence) {
    if (!isInitialized()) {
        ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
        return false;
    }
    if (ref_traj_.empty()) {
        ROS_ERROR("Empty Reference Trajectory");
        return false;
    }
    double rest_dist = sqrt(pow(current_pose.x - final_goal2D_.x, 2) + pow(current_pose.y - final_goal2D_.y, 2));
    if (rest_dist < tolarence) {
        return true;
    } else {
        return false;
    }
}

bool MPCControllerROSSimple::isInitialized() { return initialized_; }

void MPCControllerROSSimple::computeSysDynamic(const mpc_controller::pose2D& current_pose,
                                               const base_control::WheelSpeed& current_control_output,
                                               const mpc_controller::slip_param& current_slip, const double T_sample,
                                               std::shared_ptr<mpc_controller::SystemDynamicMatrix>& sysdyn_ptr) {
    double x = current_pose.x;
    double y = current_pose.y;
    double th = current_pose.theta;

    double w_r = current_control_output.right_vel.data;
    double w_l = current_control_output.left_vel.data;

    double delta_l = current_slip.delta_l;
    double delta_r = current_slip.delta_r;
    double alpha = current_slip.alpha;

    double C_l = wheel_radius_ * (1 - delta_l) / 2;
    double C_r = wheel_radius_ * (1 - delta_r) / 2;

    sysdyn_ptr->Matrix_a << 1, 0, T_sample * (-sin(th) - tan(alpha) * cos(th)) * (C_r * w_r + C_l * w_l), 0, 1,
        T_sample * (cos(th) - tan(alpha) * sin(th)) * (C_r * w_r + C_l * w_l), 0, 0, 1;

    sysdyn_ptr->Matrix_b << T_sample * C_r * (cos(th) - tan(alpha) * sin(th)),
        T_sample * C_l * (cos(th) - tan(alpha) * sin(th)), T_sample * C_r * (sin(th) + tan(alpha) * cos(th)),
        T_sample * C_l * (sin(th) + tan(alpha) * cos(th)), 2 * T_sample * C_r / width, -2 * T_sample * C_l / width;

    sysdyn_ptr->Matrix_c = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

// TODO
void MPCControllerROSSimple::odomCallback(const nav_msgs::Odometry::ConstPtr odom_msg) {}

void MPCControllerROSSimple::encCallback(base_control::WheelSpeed::ConstPtr enc_msg) {
    current_enc_ = *enc_msg;
}

void MPCControllerROSSimple::bodyVelCallback(const geometry_msgs::Vector3& body_vel_msg){
    body_vel_ = body_vel_msg;
}

bool MPCControllerROSSimple::planService(mpc_controller::MPCPlan::Request& req,
                                         mpc_controller::MPCPlan::Response& res) {
    if (req.trajectory.points.empty()) {
        ROS_ERROR("mpc controller received empty plan");
        res.success = false;
        return false;
    } else {
        res.success = true;
        if (setPlan(req.trajectory)) {
            plan_frame_ = req.trajectory.header.frame_id;
            is_first_time_ = true;
            current_time_ = last_time_ = ros::Time::now();
            mpc_status_ = RUNNING;
        }
        return true;
    }
}
void MPCControllerROSSimple::planCallback(const mpc_controller::Trajectory& mpc_plan) {
    ROS_INFO("Receive mpc plan from rviz");
    if (setPlan(mpc_plan)) {
        ROS_INFO("Set plan success, Ready to Run");
        is_first_time_ = true;
        current_time_ = last_time_ = ros::Time::now();
        mpc_status_ = RUNNING;
    }
}
bool MPCControllerROSSimple::getRobotPose(mpc_controller::pose2D& current_pose) {
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    // tf::Quaternion q;
    // tf::quaternionMsgToTF(global_pose.pose.orientation, q);
    // tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    // current_pose.x = global_pose.pose.position.x;
    // current_pose.y = global_pose.pose.position.y;
    // current_pose.theta = yaw;
    try {
        listener.lookupTransform("/odom", "/base_link", ros::Time(0), transform_);
        current_pose.x = transform_.getOrigin().x();
        current_pose.y = transform_.getOrigin().y();
        tf::Matrix3x3(transform_.getRotation()).getRPY(roll, pitch, yaw);
        current_pose.theta = yaw;
        return true;
    } catch (tf::TransformException& ex) {
        ROS_ERROR("%s", ex.what());
        ros::Duration(1.0).sleep();
        return false;
    }
}

void MPCControllerROSSimple::stopControl() {
    base_control::WheelSpeed stop_msg;
    stop_msg.header.stamp = ros::Time::now();
    stop_msg.left_vel.data = 0.0;
    stop_msg.right_vel.data = 0.0;
    pub_cmd_.publish(stop_msg);
}

}  // namespace mpc_controller