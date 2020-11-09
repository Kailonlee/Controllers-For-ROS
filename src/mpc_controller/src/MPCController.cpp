#include "mpc_controller/MPCController.h"
namespace mpc_controller{
    MPC_Controller::MPC_Controller(const uint32_t state_dim, const uint32_t control_dim,
                                    const uint32_t N_p, const uint32_t N_c){
        state_dim_ = state_dim;
        control_dim_ = control_dim;
        N_p_ = N_p;
        N_c_ = N_c;
        rho_ = 1e3;
        lower_rho_ << 0;
        upper_rho_ << 10;
        
        set_hessian_matrix_success_ = false;
        set_gradient_matrix_success_ = false;
        set_linear_constraint_matrix_success_ = false;
        set_lower_bound_success_ = false;
        set_upper_bound_success_ = false;
        initial_solver_success_ = false;

        SysDynPtr_ = std::make_shared<SystemDynamicMatrix>(state_dim_, control_dim_);

        NcOnes_.setOnes(N_c);
        Weight_P_.setZero(N_c_ * control_dim_, N_c_ * control_dim_);
        Weight_Q_.setZero(N_p_ * state_dim_, N_p_ * state_dim_);
        Weight_R_.setZero(N_c_ * control_dim_, N_c_ * control_dim_);

        Matrix_Y_.setZero();
        Matrix_PSI_.setZero(state_dim_ * N_p_, state_dim_ + control_dim_);
        Matrix_THETA_.setZero(state_dim_ * N_p_, control_dim_ * N_c_);
        Hessian_.setZero();
        Gradient_.setZero();

        control_err_.setZero(control_dim_);
        delta_U_.resize(control_dim_);
        output_U_.resize(control_dim_);

        QP_solution_.setZero(control_dim_ * N_c_);

        QP_solver_.settings()->setVerbosity(false);
        QP_solver_.settings()->setWarmStart(true);
        QP_solver_.data()->setNumberOfVariables(control_dim_ * N_c_ + 1);
        QP_solver_.data()->setNumberOfConstraints(2 * control_dim_ * N_c_ + 1);

    }

    MPC_Controller::~MPC_Controller(){

    }

    void MPC_Controller::SetSysDynMatrix(const Eigen::MatrixXd &matrix_a, 
                                        const Eigen::MatrixXd &matrix_b,
                                        const Eigen::MatrixXd &matrix_c){
        SysDynPtr_->Matrix_a = matrix_a;
        SysDynPtr_->Matrix_b = matrix_b;
        SysDynPtr_->Matrix_c = matrix_c;
    }

    void MPC_Controller::SetStateDimension(const uint32_t &state_dim){
        state_dim_ = state_dim;
    }
    void MPC_Controller::SetControlDimension(const uint32_t &control_dim){
        control_dim_ = control_dim;
    }

    void MPC_Controller::SetPlan(const std::vector<mpc_controller::TrajectoryPoint>& ref_traj){
        ref_traj_ = ref_traj;
    }

    void MPC_Controller::UpdateSystemMatrix(const pose2D &current_pose,
                                            const geometry_msgs::Vector3 &body_vel,
                                            const std::vector<mpc_controller::TrajectoryPoint> &ref_traj){
        ROS_INFO("Updating MPC System Matrix");
        current_pose2D_.x = current_pose.x;
        current_pose2D_.y = current_pose.y;
        current_pose2D_.theta = current_pose.theta;
        Eigen::VectorXd current_pose_vec, target_pose_vec;
        UpdateCurrentPoseVec(current_pose, current_pose_vec);
        std::cout << "current pose: \n" <<  current_pose_vec << std::endl;
        UpdateTargetVec(current_pose, ref_traj, body_vel, target_pose_vec, target_u_vec_);
        std::cout << "target pose: \n" <<  target_pose_vec << std::endl;
        Eigen::MatrixXd Matrix_A, Matrix_B, Matrix_C;
        Matrix_A.resize(state_dim_ + control_dim_, state_dim_ + control_dim_);
        Matrix_A.block(0,0,SysDynPtr_->Matrix_a.rows(),SysDynPtr_->Matrix_a.cols()) = SysDynPtr_->Matrix_a;
        Matrix_A.block(0,SysDynPtr_->Matrix_a.cols(),SysDynPtr_->Matrix_b.rows(),SysDynPtr_->Matrix_b.cols()) = SysDynPtr_->Matrix_b;
        Matrix_A.block(SysDynPtr_->Matrix_a.rows(), 0, control_dim_, state_dim_) = Eigen::MatrixXd::Zero(control_dim_, state_dim_);
        Matrix_A.block(SysDynPtr_->Matrix_a.rows(), SysDynPtr_->Matrix_a.cols(), control_dim_, control_dim_) = Eigen::MatrixXd::Identity(control_dim_, control_dim_);

        Matrix_B.resize(state_dim_ + control_dim_, control_dim_);
        Matrix_B.block(0,0,SysDynPtr_->Matrix_b.rows(), SysDynPtr_->Matrix_b.cols()) = SysDynPtr_->Matrix_b;
        Matrix_B.block(SysDynPtr_->Matrix_b.rows(),0, control_dim_, control_dim_) = Eigen::MatrixXd::Identity(control_dim_, control_dim_);

        Matrix_C.resize(state_dim_, state_dim_ + control_dim_);
        Matrix_C.leftCols(state_dim_) = SysDynPtr_->Matrix_c;
        Matrix_C.rightCols(control_dim_) = Eigen::MatrixXd::Zero(state_dim_, control_dim_);

        Matrix_K_.resize(N_c_, N_c_);
        Matrix_K_.bottomLeftCorner(N_c_, N_c_) = Eigen::MatrixXd::Ones(N_c_, N_c_).bottomLeftCorner(N_c_, N_c_);
        
        Matrix_M_.resize(N_c_ * control_dim_, N_c_ * control_dim_);
        Matrix_M_ = KroneckerProduct(Matrix_K_, Eigen::MatrixXd::Identity(control_dim_, control_dim_));

        std::cout << "Matrix A: \n" << Matrix_A << std::endl;
        std::cout << "Matrix B: \n" << Matrix_B << std::endl;
        std::cout << "Matrix C: \n" << Matrix_C << std::endl;

        CalculatePSI(Matrix_A, Matrix_C, Matrix_PSI_);
        CalculateTHETA(Matrix_A, Matrix_B, Matrix_C, Matrix_THETA_);

        Eigen::MatrixXd Matrix_E;
        UpdateErrVector(current_pose_vec, target_pose_vec, Matrix_E);

        CalculateHessian(Hessian_);
        CalculateGradient(Matrix_E, Gradient_);

    }
    
    void MPC_Controller::UpdateErrVector(const Eigen::VectorXd &current_pose_vec, 
                                        const Eigen::VectorXd &target_pose_vec,
                                        Eigen::MatrixXd &Matrix_E){
        ROS_INFO("Updating Error Vector");
        Eigen::VectorXd Vector_err;
        Eigen::VectorXd state_err = current_pose_vec - target_pose_vec;
        Vector_err.resize(state_dim_ + control_dim_, 1);
        Vector_err.block(0, 0, state_dim_, 1) = state_err;
        Vector_err.block(state_dim_, 0, control_dim_, 1) = control_err_;
        std::cout << "vector_err_: \n" << Vector_err << std::endl;
        Control_Err_ = KroneckerProduct(NcOnes_, control_err_);
        // std::cout << "[calculating]Control_Err_: \n" << Control_Err_ << std::endl;
        Matrix_E = Matrix_PSI_ * Vector_err;

    }

    void MPC_Controller::UpdateCurrentPoseVec(const pose2D &current_pose, 
                                            Eigen::VectorXd &current_pose_vec){

        ROS_INFO("Updating Current Pose Vector");
        current_pose_vec.resize(state_dim_);
        double x = current_pose.x;
        double y = current_pose.y;
        double th = current_pose.theta;
        current_pose_vec << x, y, th;
    }

    void MPC_Controller::UpdateTargetVec(const pose2D &current_pose, 
                                        const std::vector<mpc_controller::TrajectoryPoint> &ref_traj,
                                        const geometry_msgs::Vector3 &body_vel,
                                        Eigen::VectorXd &target_pose_vec,
                                        Eigen::VectorXd &target_u_vec){
        ROS_INFO("Updating Target Pose And Control Vector");
        target_pose_vec.resize(state_dim_);
        target_u_vec.resize(control_dim_);
        uint32_t min_dist_point_index, ref_point_index;
        min_dist_point_index = FindNearestPoint(current_pose, ref_traj);
        // ref_point_index = min_dist_point_index + ceil(body_vel.x * 0.1 / 0.03); // preview
        ref_point_index = min_dist_point_index;
        if (ref_point_index >= ref_traj.size())
        {
            ref_point_index = ref_traj.size()-1;
        }
        double x_ref = ref_traj[min_dist_point_index].pose.position.x;
        double y_ref = ref_traj[min_dist_point_index].pose.position.y;
        double th_ref = tf2::getYaw(ref_traj[ref_point_index].pose.orientation);
        double w_l_ref = ref_traj[min_dist_point_index].velocity.left_vel.data;
        double w_r_ref = ref_traj[min_dist_point_index].velocity.right_vel.data;
        
        // for error calculate
        nearest_pose2D_.x = ref_traj[min_dist_point_index].pose.position.x;
        nearest_pose2D_.y = ref_traj[min_dist_point_index].pose.position.y;
        nearest_pose2D_.theta = tf2::getYaw(ref_traj[min_dist_point_index].pose.orientation);

        target_pose_vec << x_ref, y_ref, th_ref;
        target_u_vec << w_r_ref, w_l_ref;
    }

    pose2D MPC_Controller::GetNearestPose(){
        return nearest_pose2D_;
    }
    pose2D MPC_Controller::GetCurrentPose(){
        return current_pose2D_;
    }
    
    void MPC_Controller::SetConstraintMatrix(){
        ROS_INFO("Setting Constraint Matrix");
        SetConstraintBound();
        QP_Matrix_A_.resize(2 * Matrix_M_.rows() + 1, Matrix_M_.cols() + 1);
        // std::cout << "[update] QP_Matrix_A_:\n" << QP_Matrix_A_ << std::endl;
        // QP_Matrix_A_.block(0,0,Matrix_M_.rows(),Matrix_M_.cols()) = Matrix_M_;
        for (size_t i = 0; i < Matrix_M_.rows(); i++)
        {
            for (size_t j = 0; j < Matrix_M_.cols(); j++)
            {
                double value1 = Matrix_M_(i,j);
                if (value1 != 0)
                {
                    QP_Matrix_A_.insert(i, j) = value1;
                }
            }
        }
        // std::cout << "[update] QP_Matrix_A_:\n" << QP_Matrix_A_ << std::endl;
        // QP_Matrix_A_.block(0,Matrix_M_.cols(), Matrix_M_.rows(),1) = Eigen::MatrixXd::Zero(Matrix_M_.rows(),1);
        // 0 PASS
        
        // QP_Matrix_A_.block(Matrix_M_.rows(),0,Matrix_M_.rows(),Matrix_M_.cols()) = Eigen::MatrixXd::Identity(Matrix_M_.rows(),Matrix_M_.cols());
        for (size_t i = 0; i < Matrix_M_.rows(); i++)
        {
            double value2 = 1;
            QP_Matrix_A_.insert(Matrix_M_.rows()+i, i) = value2;
        }
        // std::cout << "[update] QP_Matrix_A_:\n" << QP_Matrix_A_ << std::endl;
        // QP_Matrix_A_.block(Matrix_M_.rows(),Matrix_M_.cols(), Matrix_M_.rows(),1) = Eigen::MatrixXd::Zero(Matrix_M_.rows(),1);
        // 0 PASS

        // QP_Matrix_A_.block(2 * Matrix_M_.rows(),0, 1,Matrix_M_.cols()) = Eigen::MatrixXd::Zero(1, Matrix_M_.cols());
        // 0 PASS

        // QP_Matrix_A_.block(2 * Matrix_M_.rows(), Matrix_M_.cols(),1,1) = Eigen::MatrixXd::Identity(1,1);
        double value3 = 1;
        QP_Matrix_A_.insert(2 * Matrix_M_.rows(), Matrix_M_.cols()) = value3;

        // std::cout << "QP_Matrix_A_:\n" << QP_Matrix_A_ << std::endl;
    }

    void MPC_Controller::SetConstraintBound(){
        ROS_INFO("Setting Constraint Bound");
        Eigen::VectorXd control_err_min = u_lower_bound_ - target_u_vec_;
        Eigen::VectorXd control_err_max = u_upper_bound_ - target_u_vec_;
        Eigen::VectorXd Control_Err_Min, Control_Err_Max;

        std::cout << "control_err_min:\n" << control_err_min << std::endl;
        std::cout << "control_err_max:\n" << control_err_max << std::endl; 

        Control_Err_Min = KroneckerProduct(NcOnes_, control_err_min) - Control_Err_;
        Control_Err_Max = KroneckerProduct(NcOnes_, control_err_max) - Control_Err_;

/*         std::cout << "Control_Err_:\n" << Control_Err_ << std::endl;
        std::cout << "Control_Err_Min:\n" << Control_Err_Min << std::endl;
        std::cout << "Control_Err_Max:\n" << Control_Err_Max << std::endl; */

        Delta_U_Min_ = KroneckerProduct(NcOnes_, delta_u_lower_bound_);
        Delta_U_Max_ = KroneckerProduct(NcOnes_, delta_u_upper_bound_); 
        
        QP_lower_bound_.resize(Control_Err_Min.rows() + Delta_U_Min_.rows() + 1);
        QP_upper_bound_.resize(Control_Err_Max.rows() + Delta_U_Max_.rows() + 1);

        QP_lower_bound_.block(0,0,Control_Err_Min.rows(),1) = Control_Err_Min;
        QP_lower_bound_.block(Control_Err_Min.rows(),0,Delta_U_Min_.rows(),1) = Delta_U_Min_;
        QP_lower_bound_.block(Control_Err_Min.rows() + Delta_U_Min_.rows(),0,1,1) = lower_rho_;

        QP_upper_bound_.block(0,0,Control_Err_Max.rows(),1) = Control_Err_Max;
        QP_upper_bound_.block(Control_Err_Max.rows(),0,Delta_U_Max_.rows(),1) = Delta_U_Max_;
        QP_upper_bound_.block(Control_Err_Max.rows() + Delta_U_Max_.rows(),0,1,1) = upper_rho_;
        
/*         std::cout << "QP_lower_bound_:\n" << QP_lower_bound_ << std::endl;
        std::cout << "QP_upper_bound_:\n" << QP_upper_bound_ << std::endl; */
    }

    /**
     * @description: 设置eigen-osqp的求解参数
     * @param {type} 
     * @return: 
     */
    bool MPC_Controller::SetQPData(){
        ROS_INFO("Setting QP Data");
 
        
        if(!set_hessian_matrix_success_)
        {   
            if(QP_solver_.data()->setHessianMatrix(Hessian_)) 
            {
                set_hessian_matrix_success_ = true;
                ROS_INFO("Set Hessian Matrix Success");
            }
            else
            {
                ROS_ERROR("1");
                return false;
            }
        }
        else
        {
            QP_solver_.updateHessianMatrix(Hessian_);
        }

        if(!set_gradient_matrix_success_)
        {   
            if(QP_solver_.data()->setGradient(Gradient_)) 
            {
                set_gradient_matrix_success_ = true;
                ROS_INFO("Set Gradient Matrix Success!");
            }
            else
            {
                ROS_ERROR("2");
                return false;
            }
        }
        else
        {
            QP_solver_.updateGradient(Gradient_);
        }

        if(!set_linear_constraint_matrix_success_) 
        {
            if(QP_solver_.data()->setLinearConstraintsMatrix(QP_Matrix_A_))
            {
                set_linear_constraint_matrix_success_ = true;
                ROS_INFO("Set Linear Constraint Success!");
            }
            else
            {
                ROS_ERROR("3");
                return false;
            }
        }

        if(!set_lower_bound_success_)
        {   
            if(QP_solver_.data()->setLowerBound(QP_lower_bound_)) 
            {
                set_lower_bound_success_ = true;
                ROS_INFO("Set Lower Bound Success!");
            }
            else
            {
                ROS_ERROR("4");
                return false;
            }
        }
        else
        {
            QP_solver_.updateLowerBound(QP_lower_bound_);
        }

        if(!set_upper_bound_success_)
        {   
            if(QP_solver_.data()->setUpperBound(QP_upper_bound_)) 
            {
                set_upper_bound_success_ = true;
                ROS_INFO("Set Upper Bound Success!");
            }
            else
            {
                ROS_ERROR("5");
                return false;
            }
        }
        else
        {
            QP_solver_.updateUpperBound(QP_upper_bound_);
        }

        if(!initial_solver_success_)
        {   
            if(QP_solver_.initSolver()) 
            {
                initial_solver_success_ = true;
                ROS_INFO("Initial Solver Success!");
            }
            else
            {
                ROS_ERROR("6");
                return false;
            }
        }

        return true;
    }
    
    /**
     * @description: 使用eigen-osqp求解mpc问题
     * @param {type} 
     * @return: 求解结果的指针
     */
    bool MPC_Controller::SolveQP(){
        ROS_INFO("Solving QP");
        if (!QP_solver_.solve())
        {
            QP_solution_.setZero(control_dim_ * N_c_);
            return false;
        }
        else
        {
            QP_solution_ = QP_solver_.getSolution();
            return true;
        }
    }

    /**
     * @description: 获取控制输出
     * @param {type} 
     * @return: 控制输出的指针
     */
    void MPC_Controller::GetOutputControl(Eigen::VectorXd *output_control){
        delta_U_ = QP_solution_.head(control_dim_);
        std::cout << "delta_U_right: \n" << delta_U_[0] << std::endl;
        std::cout << "delta_U_left: \n" << delta_U_[1] << std::endl;
        control_err_ = control_err_ + delta_U_;
        *output_control = control_err_ + target_u_vec_;
    }

    /**
     * @description: 使用当前位置和参考路径,获取参考路径中的最近点
     * @param: current_pose 当前位置; ref_traj 参考路径
     * @return: 参考路径中最近点的索引
     */
    uint32_t MPC_Controller::FindNearestPoint(const pose2D &current_pose,
                                            const std::vector<mpc_controller::TrajectoryPoint> &ref_traj){
        double x = current_pose.x;
        double y = current_pose.y;

        double dist, min_dist;
        double x_ref, y_ref, th_ref;
        double min_dist_tolerance = 0.0001;
        uint32_t neareast_point_index;
        for (size_t i = 0; i < ref_traj.size(); i++)
        {
            x_ref = ref_traj[i].pose.position.x;
            y_ref = ref_traj[i].pose.position.y;
            dist = pow((x - x_ref),2) + pow((y - y_ref), 2);
            if (i == 0)
            {
                min_dist = dist;
                neareast_point_index = i;
            }
            else
            {
                if (dist < min_dist)
                {
                    min_dist = dist;
                    neareast_point_index = i;
                }
                
            }
            if (min_dist <= min_dist_tolerance)
            {
                break;
            }
        }

        ROS_INFO("Found Nearest Point: x=%.2f, y=%.2f, th=%.2f, w_r=%.2f, w_l=%.2f",
                    ref_traj[neareast_point_index].pose.position.x, 
                    ref_traj[neareast_point_index].pose.position.y,
                    tf2::getYaw(ref_traj[neareast_point_index].pose.orientation),
                    ref_traj[neareast_point_index].velocity.right_vel.data,
                    ref_traj[neareast_point_index].velocity.left_vel.data);

        return neareast_point_index;
        

    }
    /**
     * @description: 计算预测时域中的系统状态空间描述的状态转移矩阵
     * @param {type} 
     * @return: 
     */
    void MPC_Controller::CalculatePSI(const Eigen::MatrixXd &Matrix_A, const Eigen::MatrixXd &Matrix_C, Eigen::MatrixXd &Matrix_PSI){
        ROS_INFO("Calculating PSI Matrix");
        Eigen::MatrixXd matrix_pow;
        matrix_pow = Eigen::MatrixXd::Identity(Matrix_A.rows(), Matrix_A.cols());
        for (size_t i = 0; i < N_p_; i++)
        {
            matrix_pow *= Matrix_A;
            Matrix_PSI.block(state_dim_ * i, 0, state_dim_, state_dim_ + control_dim_) = Matrix_C * matrix_pow;
        }
        
    }

    /**
     * @description: 计算预测时域中的系统状态空间描述的控制矩阵
     * @param {type} 
     * @return: 
     */
    void MPC_Controller::CalculateTHETA(const Eigen::MatrixXd &Matrix_A, const Eigen::MatrixXd &Matrix_B, 
                            Eigen::MatrixXd &Matrix_C, Eigen::MatrixXd &Matrix_THETA){
        ROS_INFO("Calculating Theta Matrix");
        for (size_t i = 0; i < N_p_; i++)
        {
            for (size_t j = 0; j < N_c_; j++)
            {
                if (j <= i) 
                {
                    Matrix_THETA.block(state_dim_ * i, control_dim_ * j, 
                                                state_dim_, control_dim_) = Matrix_C * MatrixPow(Matrix_A, i-j) * Matrix_B;
                }
                else
                {
                    Matrix_THETA.block(state_dim_ * i, control_dim_ * j, 
                                                state_dim_, control_dim_) = Eigen::MatrixXd::Zero(state_dim_, control_dim_);
                }
            } 
        }                          
    }

    const Eigen::MatrixXd MPC_Controller::MatrixPow(const Eigen::MatrixXd &matrix, const int &exponent){
        Eigen::MatrixXd matrix_pow(matrix);
        matrix_pow.setIdentity();
        for (size_t i = 0; i < exponent; i++)
        {
            matrix_pow *= matrix;
        }
        return matrix_pow;
        
    }
    /**
     * @description: 克罗内克积
     * @param {type} 
     * @return: A与B 的克罗内克积
     */
    Eigen::MatrixXd MPC_Controller::KroneckerProduct(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B){
        // ROS_INFO("Using Kronecker Product");
        Eigen::MatrixXd Kronecker(A.rows() * B.rows(), A.cols() * B.cols());
        Kronecker.setZero();
        for (size_t i = 0; i < A.rows(); i++)
        {
            for (size_t j = 0; j < A.cols(); j++)
            {   
                Kronecker.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i,j) * B;
            }   
        }
        // std::cout << "[Kronecker A] Kronecker: \n" <<  A << std::endl;
        // std::cout << "[Kronecker B] Kronecker: \n" <<  B << std::endl;
        // std::cout << "[Kronecker Product] Kronecker: \n" <<  Kronecker << std::endl;

        return Kronecker;
    }

    /**
     * @description: 计算标准QP问题中的Hessian矩阵
     * @param {type} 
     * @return: 
     */
    void MPC_Controller::CalculateHessian(Eigen::SparseMatrix<double> &Hessian){
        ROS_INFO("Calculating Hessian Matrix");
        Hessian.resize(Weight_R_.rows() + 1, Weight_R_.cols() + 1);
        // Hessian.block(0,0,Weight_R_.rows(), Weight_R_.cols()) =  Matrix_THETA_.transpose() * Weight_Q_ * Matrix_THETA_ + Weight_R_ + Matrix_M_.transpose() * Weight_P_ * Matrix_M_;
        Eigen::MatrixXd temp_hessian = Matrix_THETA_.transpose() * Weight_Q_ * Matrix_THETA_ + Weight_R_ + Matrix_M_.transpose() * Weight_P_ * Matrix_M_;
        for (size_t i = 0; i < Weight_R_.rows(); i++)
        {
            for (size_t j = 0; j < Weight_R_.cols(); j++)
            {
                double value1 = temp_hessian(i,j);
                if (value1 != 0)
                {
                    Hessian.insert(i,j) = value1;
                }
            }
        }
        
        // Hessian.block(0, Weight_R_.cols(), control_dim_ * N_c_, 1) = Eigen::MatrixXd::Zero(control_dim_ * N_c_, 1);
        // 0 PASS
        // Hessian.block(Weight_R_.rows(), 0, 1, control_dim_ * N_c_) = Eigen::MatrixXd::Zero(1, control_dim_ * N_c_);
        // 0 PASS

        // Hessian.block(Weight_R_.rows(), Weight_R_.cols(), 1, 1) = rho_;
        Hessian.insert(Weight_R_.rows(), Weight_R_.cols()) = rho_;

        Hessian *= 2; /*> for QP problem form*/
    }

    /**
     * @description: 计算标准QP问题中的Gradient矩阵
     * @param {type} 
     * @return: 
     */
    void MPC_Controller::CalculateGradient(const Eigen::MatrixXd &Matrix_E, Eigen::VectorXd &Gradient){
        ROS_INFO("Calculating Gradient Matrix");
        Gradient.resize(control_dim_ * N_c_ + 1);
        // std::cout << "Control_Err_: \n" << Control_Err_ << std::endl;
        // std::cout << "Weight_P_: \n" << Weight_P_ << std::endl;
        // std::cout << "Matrix_M_: \n" << Matrix_M_ << std::endl;
        // Eigen::MatrixXd aa = 2 * Matrix_E.transpose() * Weight_Q_ * Matrix_THETA_;
        // Eigen::MatrixXd bb = 2 * Control_Err_.transpose() * Weight_P_ * Matrix_M_;
        // std::cout << "aa: \n" << aa << std::endl;
        // std::cout << "bb: \n" << bb << std::endl;    
        Eigen::MatrixXd temp_matrix = 2 * Matrix_E.transpose() * Weight_Q_ * Matrix_THETA_ + 2 * Control_Err_.transpose() * Weight_P_ * Matrix_M_;    
        Eigen::Map<Eigen::VectorXd> temp_vec(temp_matrix.data(), temp_matrix.size());
        Gradient.block(0, 0, control_dim_ * N_c_, 1) = temp_vec;
        Gradient.block(control_dim_ * N_c_,0,  1, 1) = Eigen::VectorXd::Zero(1);
    }

    void MPC_Controller::SetWeightQ(const Eigen::MatrixXd &Weight_Q){
        ROS_INFO("Setting Weight Q");
        Weight_Q_ = KroneckerProduct(Eigen::MatrixXd::Identity(N_p_, N_p_), Weight_Q);
    }

    void MPC_Controller::SetWeightR(const Eigen::MatrixXd &Weight_R){
        ROS_INFO("Setting Weight R");
        Weight_R_ = KroneckerProduct(Eigen::MatrixXd::Identity(N_c_, N_c_), Weight_R);
    }

    void MPC_Controller::SetWeightP(const Eigen::MatrixXd &Weight_P){
        ROS_INFO("Setting Weight P");
        Weight_P_ = KroneckerProduct(Eigen::MatrixXd::Identity(N_c_, N_c_), Weight_P);
    }

    void MPC_Controller::SetPredictHorizon(const uint32_t &predict_horizon){
        N_p_ = predict_horizon;
    }

    void MPC_Controller::SetControlHorizon(const uint32_t &control_horizon){
        N_c_ = control_horizon;
    }

    bool MPC_Controller::SetUBound(const Eigen::VectorXd &u_lower_bound, const Eigen::VectorXd &u_upper_bound){
        ROS_INFO("Setting U Bound");
        if (u_lower_bound.size() != control_dim_ || u_upper_bound.size() != control_dim_)
        {
            std::cout << "[Error] U bound size doesn't match the control_dim_\n";
            return false;
        }
        else
        {
            u_lower_bound_ = u_lower_bound;
            u_upper_bound_ = u_upper_bound;
            return true;
        }
    }
    
    bool MPC_Controller::SetDeltaUBound(const Eigen::VectorXd &delta_u_lower_bound, const Eigen::VectorXd &delta_u_upper_bound){
        ROS_INFO("Setting Delta U Bound");
        if (delta_u_lower_bound.size() != control_dim_ || delta_u_upper_bound.size() != control_dim_)
        {
            std::cout << "[Error] Delta U bound size doesn't match the control_dim_\n";
            return false;
        }
        else
        {
            delta_u_lower_bound_ = delta_u_lower_bound;
            delta_u_upper_bound_ = delta_u_upper_bound;
            return true;
        }
    }

    double MPC_Controller::ComputeLonErr(){
        double lon_err = (current_pose2D_.x - nearest_pose2D_.x) * cos(nearest_pose2D_.theta) + (current_pose2D_.y - nearest_pose2D_.y) * sin(nearest_pose2D_.theta);
        return lon_err;
    }

    double MPC_Controller::ComputeLatErr(){
        double lat_err = (current_pose2D_.y - nearest_pose2D_.y) * cos(nearest_pose2D_.theta) - (current_pose2D_.x - nearest_pose2D_.x) * sin(nearest_pose2D_.theta);
        return lat_err;
    }

    double MPC_Controller::ComputeYawErr(){
        double yaw_err = current_pose2D_.theta - nearest_pose2D_.theta;
        return yaw_err;
    }
}