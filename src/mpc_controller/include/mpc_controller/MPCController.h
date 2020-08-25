#ifndef MPCCONTROLLER_H
#define MPCCONTROLLER_H

#include<vector>
#include<Eigen/Core>
#include<Eigen/Sparse>
#include<geometry_msgs/PoseStamped.h>
#include<base_local_planner/local_planner_util.h>
#include<memory>
#include<OsqpEigen/OsqpEigen.h>
#include<tf2/utils.h>
#include<math.h>

// temp message definition
#include "mpc_controller/TrajectoryPoint.h"

namespace mpc_controller{

    typedef struct _pose2D
    {
        double x;
        double y;
        double theta;
        _pose2D() : x(0.0), y(0.0), theta(0.0){}
        _pose2D(double x, double y, double theta){
            this->x = x;
            this->y = y;
            this->theta = theta;
        }   
        _pose2D(const _pose2D &pose_1){
            this->x = pose_1.x;
            this->y = pose_1.y;
            this->theta = pose_1.theta;
        }
    } pose2D;

    typedef struct _slip_param
    {
        double delta_l;
        double delta_r;
        double alpha;
        _slip_param(): delta_l(0.0), delta_r(0.0), alpha(0.0){}
        _slip_param(double _delta_l, double _delta_r, double _alpha){
            this->delta_l = _delta_l;
            this->delta_r = _delta_r;
            this->alpha = _alpha;
        }
    } slip_param;

    typedef struct _control_input{
        double w_r;
        double w_l;
    } control_input;

    

    typedef struct _SystemDynamicMatrix{
        Eigen::MatrixXd Matrix_a;
        Eigen::MatrixXd Matrix_b;
        Eigen::MatrixXd Matrix_c;
        _SystemDynamicMatrix(const uint32_t state_dim, const uint32_t control_dim){
            this->Matrix_a.resize(state_dim, state_dim);
            this->Matrix_b.resize(state_dim, control_dim);
            this->Matrix_c.resize(state_dim, state_dim);
        }
    } SystemDynamicMatrix;
    
    class MPC_Controller
    {
    private:
        
        std::vector<mpc_controller::TrajectoryPoint> ref_traj_;
        std::shared_ptr<base_local_planner::LocalPlannerUtil> planner_util_ptr_;

        uint32_t N_p_;//预测时域
        uint32_t N_c_;//控制时域
        uint32_t ref_point_index_;

        Eigen::MatrixXd Weight_Q_, Weight_R_, Weight_P_;
        Eigen::MatrixXd Matrix_Y_, Matrix_PSI_, Matrix_THETA_;
        Eigen::MatrixXd Matrix_K_;
        
        Eigen::VectorXd control_err_;
        Eigen::VectorXd Control_Err_;
        
        Eigen::MatrixXd Matrix_M_;
        Eigen::SparseMatrix<double> Hessian_;
        Eigen::VectorXd Gradient_;

        double rho_;
        Eigen::Matrix<double, 1, 1> lower_rho_, upper_rho_;

        Eigen::VectorXd NcOnes_;
        Eigen::VectorXd u_lower_bound_, u_upper_bound_; 
        Eigen::VectorXd delta_u_lower_bound_, delta_u_upper_bound_;
        Eigen::VectorXd Delta_U_Min_, Delta_U_Max_;
        Eigen::VectorXd QP_lower_bound_, QP_upper_bound_;

        bool set_hessian_matrix_success_;
        bool set_gradient_matrix_success_;
        bool set_linear_constraint_matrix_success_;
        bool set_lower_bound_success_;
        bool set_upper_bound_success_;
        bool initial_solver_success_;

        bool control_finished_;
        Eigen::SparseMatrix<double> QP_Matrix_A_;

        OsqpEigen::Solver QP_solver_;
        Eigen::VectorXd QP_solution_;

        Eigen::VectorXd delta_U_, output_U_;
        Eigen::VectorXd target_u_vec_;

        pose2D target_pose2D_;

        const Eigen::MatrixXd MatrixPow(const Eigen::MatrixXd &matrix, const int& exponent);
        void CalculatePSI(const Eigen::MatrixXd &Matrix_A, const Eigen::MatrixXd &Matrix_C, Eigen::MatrixXd &Matrix_PSI);
        void CalculateTHETA(const Eigen::MatrixXd &Matrix_A, const Eigen::MatrixXd &Matrix_B, 
                            Eigen::MatrixXd &Matrix_C, Eigen::MatrixXd &Matrix_THETA);
        Eigen::MatrixXd KroneckerProduct(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);

        void UpdateCurrentPoseVec(const pose2D &current_pose, 
                                Eigen::VectorXd &current_pose_vec);
        void UpdateTargetVec(const pose2D &current_pose, 
                            const std::vector<mpc_controller::TrajectoryPoint> &ref_traj,
                            Eigen::VectorXd &target_pose_vec,
                            Eigen::VectorXd &target_u_vec);
        void UpdateErrVector(const Eigen::VectorXd &current_pose_vec, 
                            const Eigen::VectorXd &target_pose_vec,
                            Eigen::MatrixXd &Matrix_E);
        
        void CalculateHessian(Eigen::SparseMatrix<double> &Hessian);
        void CalculateGradient(const Eigen::MatrixXd &Matrix_E, Eigen::VectorXd &Gradient);

        void SetConstraintBound();
    public:
        MPC_Controller(const uint32_t state_dim, const uint32_t control_dim,
                        const uint32_t N_p, const uint32_t N_c);
        ~MPC_Controller();
        uint32_t FindNearestPoint(const pose2D &current_pose,
                                const std::vector<mpc_controller::TrajectoryPoint> &ref_traj);
        double ComputeLatErr();
        double ComputeLonErr();

        void SetPlan(const std::vector<mpc_controller::TrajectoryPoint>& ref_traj);
        void SetSysDynMatrix(const Eigen::MatrixXd &matrix_a, 
                            const Eigen::MatrixXd &matrix_b,
                            const Eigen::MatrixXd &matrix_c);

        void SetStateDimension(const uint32_t &state_dim);
        void SetControlDimension(const uint32_t &control_dim);

        void SetWeightQ(const Eigen::MatrixXd &Weight_Q);
        void SetWeightR(const Eigen::MatrixXd &Weight_R);
        void SetWeightP(const Eigen::MatrixXd &Weight_P);

        void SetPredictHorizon(const uint32_t &predict_horizon);
        void SetControlHorizon(const uint32_t &control_horizon);

        bool SetUBound(const Eigen::VectorXd &u_lower_bound, const Eigen::VectorXd &u_upper_bound);
        bool SetDeltaUBound(const Eigen::VectorXd &delta_u_lower_bound, const Eigen::VectorXd &delta_u_upper_bound);


        void UpdateSystemMatrix(const pose2D &current_pose,
                                const std::vector<mpc_controller::TrajectoryPoint> &ref_traj);

        void GetTargetPoint(uint32_t target_point_index, pose2D* target_pose);
        bool SetQPData();
        void SetConstraintMatrix();
        bool SolveQP();
        void GetOutputControl(Eigen::VectorXd *output_control);
        bool isFinished(){return control_finished_;};


    
    protected:
        double wheel_radius_;
        double width;
        double T_sample;
        uint32_t state_dim_;
        uint32_t control_dim_;
        std::shared_ptr<SystemDynamicMatrix> SysDynPtr_;
        // Eigen::VectorXd current_pose_vec_, target_pose_vec_;

    };
    
    
}
#endif