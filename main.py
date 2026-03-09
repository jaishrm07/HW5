import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import Panda
from objects import objects
from teleop import KeyboardController
from helper_functions import go_home, pick_cube


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
p.changeDynamics(panda.panda, 9, lateralFriction=2.5)
p.changeDynamics(panda.panda, 10, lateralFriction=2.5)
p.changeDynamics(microwave.object, 1, lateralFriction=3.0)
p.changeDynamics(cabinet.object, 1, lateralFriction=3.0)

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
ALLOWED_ACTIONS = {"pick_cube", "go_home"}
cube_map = {"cube1": cube1, "cube2": cube2, "cube3": cube3}

print("[mode] chat mode active by default")
print("[mode] type tasks in chat; type 'teleop' anytime for manual control")
print("[chat] supported commands: pick cube1/cube2/cube3, go home")


def _sync_teleop_targets_from_robot():
    """Prevent jumps when switching back from teleop/chat transitions."""
    global target_position, target_quaternion
    state_now = panda.get_state()
    target_position = np.array(state_now["ee-position"], dtype=float)
    target_quaternion = state_now["ee-quaternion"]


def _parse_cube_name(task_text: str) -> str:
    text = task_text.lower()
    if "cube2" in text or "cube 2" in text:
        return "cube2"
    if "cube3" in text or "cube 3" in text:
        return "cube3"
    return "cube1"


def _plan_task(task_text: str):
    """
    Simple rule-based planner for chat mode.
    Returns dict: steps, handoff_message, message.
    """
    text = task_text.strip().lower()
    if text in {"", "cancel"}:
        return {"steps": [], "handoff_message": "", "message": "no task entered. still in chat mode."}

    steps = []
    if "pick" in text and "cube" in text:
        steps.append({"action": "pick_cube", "cube": _parse_cube_name(text)})
    if any(phrase in text for phrase in ("go home", "go to home", "return home")) or text == "home":
        steps.append({"action": "go_home"})

    if not steps:
        return {
            "steps": [],
            "handoff_message": "",
            "message": "unsupported task. supported: 'pick cube1/cube2/cube3' or 'go home'.",
        }

    return {"steps": steps, "handoff_message": "", "message": f"planned {len(steps)} step(s)"}


def _run_whitelisted_step(step):
    action = step.get("action", "")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"action '{action}' is not in whitelist")

    if action == "pick_cube":
        cube_name = step.get("cube", "cube1")
        if cube_name not in cube_map:
            raise ValueError(f"unknown cube '{cube_name}'")
        pick_cube(panda, cube_map[cube_name], control_dt=control_dt)
        return

    if action == "go_home":
        go_home(panda, control_dt=control_dt)
        _sync_teleop_targets_from_robot()
        return


def _handle_chat_prompt():
    global mode, chat_prompt_pending

    task = input("\n[chat] enter one command (or type 'teleop' / '.' for manual): ").strip()
    text = task.lower()
    if text in {"teleop", "manual", "exit", "back", "."}:
        mode = "teleop"
        chat_prompt_pending = False
        _sync_teleop_targets_from_robot()
        print("[mode] returning to teleop")
        return

    plan = _plan_task(task)
    print(f"[chat] {plan['message']}")
    if plan["steps"]:
        print(f"[chat] running {len(plan['steps'])} step(s)")
        for idx, step in enumerate(plan["steps"], start=1):
            print(f"[chat] step {idx}/{len(plan['steps'])}: {step}")
            _run_whitelisted_step(step)
            _sync_teleop_targets_from_robot()
        if plan["handoff_message"]:
            print(f"[chat] {plan['handoff_message']}")
        print("[chat] command complete")
    else:
        print("[chat] no action executed")

    mode = "chat"
    chat_prompt_pending = True


def _run_chat_command_once():
    global mode, chat_prompt_pending
    try:
        _handle_chat_prompt()
    except Exception as exc:
        print(f"[chat] command failed: {exc}")
        mode = "chat"
        chat_prompt_pending = True


while True:
    action = teleop.get_action()

    if mode == "teleop":
        # update end-effector targets from keyboard input
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
