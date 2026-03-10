import argparse
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

from helper_functions import go_home, open_cabinet, open_microwave, pick_cube, place_in_microwave
from objects import objects
from robot import Panda

def build_scene(gui: bool):
    control_dt = 1.0 / 240.0
    client_mode = p.GUI if gui else p.DIRECT
    p.connect(client_mode)
    p.setGravity(0, 0, -9.81)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=40.0,
            cameraPitch=-40.0,
            cameraTargetPosition=[0.5, 0.0, 0.2],
        )

    urdf_root = pybullet_data.getDataPath()
    p.loadURDF(os.path.join(urdf_root, "plane.urdf"), basePosition=[0, 0, -0.625])
    p.loadURDF(os.path.join(urdf_root, "table/table.urdf"), basePosition=[0.5, 0, -0.625])

    cube1 = objects.SimpleObject(
        "cube.urdf", basePosition=[0.5, -0.3, 0.025], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.7])
    )
    cube2 = objects.SimpleObject(
        "cube.urdf", basePosition=[0.4, -0.2, 0.025], baseOrientation=p.getQuaternionFromEuler([0, 0, -0.3])
    )
    cube3 = objects.SimpleObject(
        "cube.urdf", basePosition=[0.5, -0.1, 0.025], baseOrientation=p.getQuaternionFromEuler([0, 0, 0.2])
    )
    cabinet = objects.CollabObject(
        "cabinet.urdf", basePosition=[0.9, -0.3, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi])
    )
    microwave = objects.CollabObject(
        "microwave.urdf", basePosition=[0.5, 0.3, 0.2], baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi / 2])
    )
    joint_start_positions = [0.0, 0.0, 0.0, -2 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4, 0.0, 0.0, 0.04, 0.04]
    panda = Panda(
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        jointStartPositions=joint_start_positions,
    )
    p.changeDynamics(panda.panda, 9, lateralFriction=6.0)
    p.changeDynamics(panda.panda, 10, lateralFriction=6.0)
    p.changeDynamics(microwave.object, 1, lateralFriction=8.0)
    p.changeDynamics(cabinet.object, 1, lateralFriction=8.0)

    return control_dt, panda, {"cube1": cube1, "cube2": cube2, "cube3": cube3}, cabinet, microwave


def run_task(task: str, cube_name: str, panda, cubes, cabinet, microwave, control_dt: float):
    target_cube = cubes[cube_name]
    if task == "open_cabinet":
        result = open_cabinet(panda, cabinet, control_dt=control_dt)
        print(f"[open_cabinet] start={result['start_extension_m']:.3f} final={result['final_extension_m']:.3f}")
        go_home(panda, control_dt=control_dt)
        return

    if task == "open_microwave":
        result = open_microwave(panda, microwave, control_dt=control_dt)
        print(f"[open_microwave] start={result['start_angle_rad']:.3f} final={result['final_angle_rad']:.3f}")
        go_home(panda, control_dt=control_dt)
        return

    if task == "pick_cube":
        pick_cube(panda, target_cube, control_dt=control_dt)
        print(f"[pick_cube] picked={cube_name}")
        go_home(panda, control_dt=control_dt)
        return

    if task == "place_in_microwave":
        pick_cube(panda, target_cube, control_dt=control_dt)
        open_microwave(panda, microwave, control_dt=control_dt)
        place_in_microwave(panda, target_cube, microwave, control_dt=control_dt)
        print(f"[place_in_microwave] cube={cube_name} placed")
        go_home(panda, control_dt=control_dt)
        return

    if task == "go_home":
        go_home(panda, control_dt=control_dt)
        print("[go_home] done")
        return


def main():
    parser = argparse.ArgumentParser(description="Test helper_functions primitives without editing main.py")
    parser.add_argument(
        "--task",
        choices=["open_cabinet", "open_microwave", "pick_cube", "place_in_microwave", "go_home"],
        default="open_microwave",
    )
    parser.add_argument("--cube", choices=["cube1", "cube2", "cube3"], default="cube1")
    parser.add_argument("--nogui", action="store_true", help="Run in DIRECT mode and exit after one task")
    args = parser.parse_args()

    control_dt, panda, cubes, cabinet, microwave = build_scene(gui=not args.nogui)
    action_dt = 0.0 if args.nogui else control_dt
    did_run = False

    while True:
        if not did_run:
            run_task(args.task, args.cube, panda, cubes, cabinet, microwave, action_dt)
            did_run = True
            if args.nogui:
                break

        p.stepSimulation()
        time.sleep(control_dt)


if __name__ == "__main__":
    main()
