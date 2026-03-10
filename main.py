import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import Panda
from objects import objects
from teleop import KeyboardController
from helper_functions import close_cabinet, go_home, open_cabinet, open_microwave, pick_cube
from llm_router import LLMChatRouter


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

# increase contact friction for handle grasp stability in teleop
p.changeDynamics(panda.panda, 9, lateralFriction=6.0)
p.changeDynamics(panda.panda, 10, lateralFriction=6.0)
p.changeDynamics(microwave.object, 1, lateralFriction=8.0)
p.changeDynamics(cabinet.object, 1, lateralFriction=8.0)

# always start at home pose
go_home(panda, control_dt=control_dt)

# teleoperation interface
teleop = KeyboardController()

# initial robot targets for teleop updates
state = panda.get_state()
target_position = np.array(state["ee-position"], dtype=float)
target_quaternion = state["ee-quaternion"]

# interaction modes
mode = "chat"  # teleop | chat
toggle_latched = False
chat_prompt_pending = True

# whitelist and object maps used by chat execution
ALLOWED_ACTIONS = {"pick_cube", "go_home", "open_microwave", "open_cabinet", "close_cabinet"}
cube_map = {"cube1": cube1, "cube2": cube2, "cube3": cube3}

llm_router = None
llm_init_error = None
try:
    llm_router = LLMChatRouter()
except Exception as exc:
    llm_init_error = exc
    mode = "teleop"
    chat_prompt_pending = False

if llm_router is not None:
    print("[mode] llm chat mode active by default")
    print("[mode] type natural-language tasks; type 'teleop' anytime for manual control")
else:
    print(f"[llm] unavailable: {llm_init_error}")
    print("[mode] starting in teleop because the llm router is unavailable")


def _sync_teleop_targets_from_robot():
    """Prevent jumps when switching back from teleop/chat transitions."""
    global target_position, target_quaternion
    state_now = panda.get_state()
    target_position = np.array(state_now["ee-position"], dtype=float)
    target_quaternion = state_now["ee-quaternion"]

def _run_whitelisted_step(step):
    action = step.get("action", "")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"action '{action}' is not in whitelist")

    if action == "pick_cube":
        cube_name = step.get("cube", "cube1")
        if cube_name not in cube_map:
            raise ValueError(f"unknown cube '{cube_name}'")
        pick_cube(panda, cube_map[cube_name], control_dt=control_dt)
        _sync_teleop_targets_from_robot()
        return

    if action == "go_home":
        go_home(panda, control_dt=control_dt)
        _sync_teleop_targets_from_robot()
        return

    if action == "open_microwave":
        open_microwave(panda, microwave, control_dt=control_dt)
        go_home(panda, control_dt=control_dt)
        _sync_teleop_targets_from_robot()
        return

    if action == "open_cabinet":
        open_cabinet(panda, cabinet, control_dt=control_dt)
        go_home(panda, control_dt=control_dt)
        _sync_teleop_targets_from_robot()
        return

    if action == "close_cabinet":
        close_cabinet(panda, cabinet, control_dt=control_dt)
        go_home(panda, control_dt=control_dt)
        _sync_teleop_targets_from_robot()
        return


def _execute_llm_tool(tool_name, arguments):
    global mode, chat_prompt_pending

    if tool_name == "switch_to_teleop":
        reason = arguments.get("reason", "manual control requested")
        mode = "teleop"
        chat_prompt_pending = False
        _sync_teleop_targets_from_robot()
        return {"status": "handed_off", "reason": reason}

    step = {"action": tool_name}
    if tool_name == "pick_cube":
        step["cube"] = arguments.get("cube_name", "cube1")

    _run_whitelisted_step(step)
    _sync_teleop_targets_from_robot()
    return {"status": "ok", "action": tool_name, "arguments": arguments}


def _handle_chat_prompt():
    global mode, chat_prompt_pending

    task = input("\n[llm] enter instruction (or type 'teleop' / '.' for manual): ").strip()
    text = task.lower()
    if text in {"teleop", "manual", "exit", "back", "."}:
        mode = "teleop"
        chat_prompt_pending = False
        _sync_teleop_targets_from_robot()
        print("[mode] returning to teleop")
        return

    if llm_router is None:
        print(f"[llm] unavailable: {llm_init_error}")
        mode = "teleop"
        chat_prompt_pending = False
        _sync_teleop_targets_from_robot()
        return

    turn = llm_router.run_turn(task, _execute_llm_tool)
    for idx, step in enumerate(turn["executed_steps"], start=1):
        print(f"[llm] tool {idx}: {step['tool_name']} args={step['arguments']}")

    if turn["assistant_message"]:
        print(f"[assistant] {turn['assistant_message']}")

    if mode == "teleop":
        print("[mode] returning to teleop")
        return

    mode = "chat"
    chat_prompt_pending = True


def _run_chat_command_once():
    global mode, chat_prompt_pending
    try:
        _handle_chat_prompt()
    except Exception as exc:
        print(f"[llm] command failed: {exc}")
        mode = "chat"
        chat_prompt_pending = True


while True:
    action = teleop.get_action()

    if mode == "teleop":
        # update end-effector targets from keyboard input for the teleop mode
        target_position = target_position + action[0:3]
        target_quaternion = p.multiplyTransforms(
            [0, 0, 0],
            p.getQuaternionFromEuler(action[3:6]),
            [0, 0, 0],
            target_quaternion,
        )[1]

        # move robot to target pose for the teleop mode
        panda.move_to_pose(ee_position=target_position, ee_quaternion=target_quaternion)

        # open/close gripper from keyboard
        if action[6] == +1:
            panda.open_gripper()
        elif action[6] == -1:
            panda.close_gripper()

        # "." switches to chat once per key press
        if action[7] == +1 and not toggle_latched:
            toggle_latched = True
            mode = "chat"
            chat_prompt_pending = True
            print("[mode] switched to chat")
        elif action[7] == 0:
            toggle_latched = False

    elif mode == "chat":
        # hold current pose while waiting for chat command
        panda.move_to_pose(ee_position=target_position, ee_quaternion=target_quaternion)
        if chat_prompt_pending:
            chat_prompt_pending = False
            _run_chat_command_once()
    else:
        # defensive fallback if mode is ever set to an invalid value
        print(f"[mode] unknown mode '{mode}', returning to teleop")
        mode = "teleop"
        chat_prompt_pending = False
        _sync_teleop_targets_from_robot()

    # step the simulation
    p.stepSimulation()
    time.sleep(control_dt)
