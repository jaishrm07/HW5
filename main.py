import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import Panda
from objects import objects
from teleop import KeyboardController


# parameters
control_dt = 1. / 240.

# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
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
cube1 = objects.SimpleObject("cube.urdf", basePosition=[0.5, -0.3, 0.025], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.7]))
cube2 = objects.SimpleObject("cube.urdf", basePosition=[0.4, -0.2, 0.025], baseOrientation=p.getQuaternionFromEuler([0, 0, -0.3]))
cube3 = objects.SimpleObject("cube.urdf", basePosition=[0.5, -0.1, 0.025], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.2]))
cabinet = objects.CollabObject("cabinet.urdf", basePosition=[0.9, -0.3, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]))
microwave = objects.CollabObject("microwave.urdf", basePosition=[0.5, 0.3, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]))

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

# teleoperation interface
teleop = KeyboardController()

# initial robot targets for teleop updates
state = panda.get_state()
target_position = np.array(state["ee-position"], dtype=float)
target_quaternion = state["ee-quaternion"]


while True:

    # update end-effector targets from keyboard input
    action = teleop.get_action()
    target_position = target_position + action[0:3]
    target_quaternion = p.multiplyTransforms(
        [0, 0, 0],
        p.getQuaternionFromEuler(action[3:6]),
        [0, 0, 0],
        target_quaternion,
    )[1]

    # move robot to target pose
    panda.move_to_pose(ee_position=target_position, ee_quaternion=target_quaternion)

    # open/close gripper from keyboard
    if action[6] == +1:
        panda.open_gripper()
    elif action[6] == -1:
        panda.close_gripper()

    # step the simulation
    p.stepSimulation()
    time.sleep(control_dt)
