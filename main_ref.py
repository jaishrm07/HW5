import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import json
from robot import Panda
from teleop import KeyboardController

# define the relevant features
# features are (1) robot distance from cube and
# (2-4) cube distance from initial position in x, y, and z
def get_features(robot_position, cube_position, cube_init_position):
    robot_position = np.array(robot_position)
    cube_position = np.array(cube_position)
    cube_init_position = np.array(cube_init_position)
    feature = np.array([np.linalg.norm(robot_position - cube_position), 
                                abs(cube_position[0] - cube_init_position[0]),
                                abs(cube_position[1] - cube_init_position[1]),
                                abs(cube_position[2] - cube_init_position[2])])
    return feature

# get the score for your choice of theta
def get_score(feature, theta):
    theta = np.array(theta)
    return feature @ theta

# parameters
control_dt = 1. / 240.

# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
# disable keyboard shortcuts so they do not interfere with keyboard control
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-30.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
# randomize the cube x-y location
cube_x = np.random.uniform(0.4, 0.6)
cube_y = np.random.uniform(-0.3, +0.3)
cube = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[cube_x, cube_y, 0.025])

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

# teleoperation interface
teleop = KeyboardController()

# get initial conditions
state = panda.get_state()
target_position = state["ee-position"]
target_quaternion = state['ee-quaternion']
cube_init_position = p.getBasePositionAndOrientation(cube)[0]

# set your scoring function parameters
# values can be between -1 and +1
# the score is transpose(feature) * theta
#theta = [-1.0, 0, 0, 0]

# following is the theta learned after 1000 iterations, beta = 1, K=50 (number of preference pairs)
theta_learned = [-0.86868374, 0.02319405, 0.70171651, 0.90475949]
theta = theta_learned
# main loop
last_press = time.time()
while True:
    # update the target pose
    action = teleop.get_action()
    target_position = target_position + action[0:3]
    target_quaternion = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler(action[3:6]),
                                                [0, 0, 0], target_quaternion)[1]
    # move to the target pose
    panda.move_to_pose(ee_position=target_position, ee_quaternion=target_quaternion)

    # open or close the gripper
    if action[6] == +1:
        panda.open_gripper()
    elif action[6] == -1:
        panda.close_gripper()

    # print when "." is pressed
    if action[7] == +1:
        state = panda.get_state()
        robot_position = state["ee-position"]
        cube_position = p.getBasePositionAndOrientation(cube)[0]
        feature = get_features(robot_position, cube_position, cube_init_position)
        score = get_score(feature, theta)
        print("features are:", np.round(feature, 3))
        #print("\n score after theta learned = theta_learned = [-0.86868374, 0.02319405, 0.70171651, 0.90475949]\n")
        print("score is:", np.round(score, 3))

    # step simulation
    p.stepSimulation()
    time.sleep(control_dt)