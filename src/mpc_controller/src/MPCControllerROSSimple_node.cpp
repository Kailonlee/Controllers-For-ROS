#include "mpc_controller/MPCControllerROSSimple.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "mpc_controller_node");
    mpc_controller::MPCControllerROSSimple mpc_ros_simple;
    return 0;
}