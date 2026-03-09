import time
from typing import Sequence

import numpy as np
import pybullet as p


DEFAULT_CONTROL_DT = 1.0 / 240.0


def _step_sim(control_dt: float, steps: int = 1) -> None:
    for _ in range(max(1, int(steps))):
        p.stepSimulation()
        time.sleep(control_dt)


def _move_linear(
    panda,
    start_pos: Sequence[float],
    end_pos: Sequence[float],
    ee_rotz: float,
    steps: int,
    control_dt: float,
    position_gain: float = 0.04,
) -> None:
    start = np.array(start_pos, dtype=float)
    end = np.array(end_pos, dtype=float)
    for alpha in np.linspace(0.0, 1.0, max(2, int(steps))):
        waypoint = (1.0 - alpha) * start + alpha * end
        panda.move_to_pose(waypoint.tolist(), ee_rotz=ee_rotz, positionGain=position_gain)
        _step_sim(control_dt, 1)


def _move_linear_quat(
    panda,
    start_pos: Sequence[float],
    end_pos: Sequence[float],
    ee_quaternion: Sequence[float],
    steps: int,
    control_dt: float,
    position_gain: float = 0.04,
) -> None:
    start = np.array(start_pos, dtype=float)
    end = np.array(end_pos, dtype=float)
    for alpha in np.linspace(0.0, 1.0, max(2, int(steps))):
        waypoint = (1.0 - alpha) * start + alpha * end
        panda.move_to_pose(waypoint.tolist(), ee_quaternion=list(ee_quaternion), positionGain=position_gain)
        _step_sim(control_dt, 1)


def go_home(panda, control_dt: float = DEFAULT_CONTROL_DT, steps: int = 180):
    """Move the end-effector to a neutral pose and open the gripper."""
    home_pose = np.array([0.35, 0.0, 0.45], dtype=float)
    home_yaw = 0.0
    current_pose = np.array(panda.get_state()["ee-position"], dtype=float)

    panda.open_gripper()
    _step_sim(control_dt, 30)
    _move_linear(panda, current_pose, home_pose, home_yaw, steps, control_dt)
    return panda.get_state()


# def move_elbow_forearm_joint(
#     panda,
#     delta_rad: float = -0.6,
#     control_dt: float = DEFAULT_CONTROL_DT,
#     steps: int = 160,
#     joint_index: int = 3,
# ):
#     """
#     Move the elbow/forearm bend joint (default Panda arm joint index 3).

#     `delta_rad` is added to the current joint value and clamped to the URDF limits.
#     """
#     if joint_index < 0 or joint_index > 6:
#         raise ValueError("joint_index must be one of Panda arm joints 0..6")

#     current = list(panda.get_state()["joint-position"][:9])
#     target = current.copy()
#     joint_info = p.getJointInfo(panda.panda, joint_index)
#     lower, upper = float(joint_info[8]), float(joint_info[9])
#     target[joint_index] = float(np.clip(current[joint_index] + delta_rad, lower, upper))

#     steps_i = max(2, int(steps))
#     for alpha in np.linspace(0.0, 1.0, steps_i):
#         q_cmd = [(1.0 - alpha) * current[i] + alpha * target[i] for i in range(9)]
#         p.setJointMotorControlArray(
#             panda.panda,
#             range(9),
#             p.POSITION_CONTROL,
#             targetPositions=q_cmd,
#             positionGains=[0.06] * 9,
#         )
#         _step_sim(control_dt, 1)

#     return {
#         "joint_index": joint_index,
#         "start_angle_rad": float(current[joint_index]),
#         "target_angle_rad": float(target[joint_index]),
#         "delta_applied_rad": float(target[joint_index] - current[joint_index]),
#     }


def open_microwave(
    panda,
    microwave,
    control_dt: float = DEFAULT_CONTROL_DT,
    descend_steps: int = 120,
    move_steps: int = 160,
    rotate_steps: int = 120,
    pull_steps: int = 220,
    standoff_distance: float = 0.14,
    pre_grasp_distance: float = 0.07,
    grasp_distance: float = 0.018,
    target_door_angle: float = -1.45,
):
    """
    Open the microwave by:
    1) descending to handle height,
    2) rotating to a horizontal side-grasp orientation,
    3) approaching and grasping the handle,
    4) pulling while opening the hinge.
    """
    robot_state = panda.get_state()
    microwave_state = microwave.get_state()

    ee_pos = np.array(robot_state["ee-position"], dtype=float)
    ee_quat = list(robot_state["ee-quaternion"])
    handle_pos = np.array(microwave_state["handle_position"], dtype=float)
    handle_z = float(handle_pos[2])

    approach_xy = handle_pos[:2] - ee_pos[:2]
    approach_norm = float(np.linalg.norm(approach_xy))
    if approach_norm < 1e-6:
        # Fallback to a scene-fixed direction if EE and handle XY happen to match.
        approach_xy = np.array([0.0, 1.0], dtype=float)
    else:
        approach_xy = approach_xy / approach_norm

    descend_pos = np.array([ee_pos[0], ee_pos[1], handle_z], dtype=float)
    standoff_pos = np.array(
        [
            handle_pos[0] - approach_xy[0] * standoff_distance,
            handle_pos[1] - approach_xy[1] * standoff_distance,
            handle_z,
        ],
        dtype=float,
    )

    _move_linear_quat(
        panda,
        ee_pos,
        descend_pos,
        ee_quaternion=ee_quat,
        steps=descend_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )
    _move_linear_quat(
        panda,
        descend_pos,
        standoff_pos,
        ee_quaternion=ee_quat,
        steps=move_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )

    # Horizontal side-grasp orientation.
    grasp_yaw = float(np.arctan2(approach_xy[0], -approach_xy[1]))
    grasp_quat = p.getQuaternionFromEuler([np.pi / 2.0, np.pi / 2.0, grasp_yaw])

    for _ in range(max(2, int(rotate_steps))):
        panda.move_to_pose(standoff_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        _step_sim(control_dt, 1)

    microwave_state = microwave.get_state()
    handle_pos = np.array(microwave_state["handle_position"], dtype=float)
    pre_grasp_pos = np.array(
        [
            handle_pos[0] - approach_xy[0] * pre_grasp_distance,
            handle_pos[1] - approach_xy[1] * pre_grasp_distance,
            handle_z,
        ],
        dtype=float,
    )
    grasp_pos = np.array(
        [
            handle_pos[0] - approach_xy[0] * grasp_distance,
            handle_pos[1] - approach_xy[1] * grasp_distance,
            handle_z,
        ],
        dtype=float,
    )

    _move_linear_quat(
        panda,
        standoff_pos,
        pre_grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )
    _move_linear_quat(
        panda,
        pre_grasp_pos,
        grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )

    panda.close_gripper()
    _step_sim(control_dt, 70)

    joint_info = p.getJointInfo(microwave.object, 0)
    lower_limit = float(joint_info[8])
    upper_limit = float(joint_info[9])
    start_angle = float(microwave.get_state()["joint_angle"])
    clamped_target = float(np.clip(target_door_angle, lower_limit, upper_limit))

    handle_now = np.array(microwave.get_state()["handle_position"], dtype=float)
    ee_now = np.array(panda.get_state()["ee-position"], dtype=float)
    grasp_offset = ee_now - handle_now

    for angle in np.linspace(start_angle, clamped_target, max(2, int(pull_steps))):
        p.setJointMotorControl2(
            microwave.object,
            0,
            p.POSITION_CONTROL,
            targetPosition=float(angle),
            force=100.0,
            maxVelocity=2.0,
        )
        _step_sim(control_dt, 2)

        handle_now = np.array(microwave.get_state()["handle_position"], dtype=float)
        follow_pos = handle_now + grasp_offset
        panda.move_to_pose(follow_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        _step_sim(control_dt, 1)

    panda.open_gripper()
    _step_sim(control_dt, 35)

    end_pos = np.array(panda.get_state()["ee-position"], dtype=float)
    retreat_pos = end_pos + np.array([0.0, 0.0, 0.08], dtype=float)
    _move_linear_quat(
        panda,
        end_pos,
        retreat_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.05,
    )

    final_angle = float(microwave.get_state()["joint_angle"])
    return {"start_angle_rad": start_angle, "target_angle_rad": clamped_target, "final_angle_rad": final_angle}


def open_cabinet(
    panda,
    cabinet,
    control_dt: float = DEFAULT_CONTROL_DT,
    descend_steps: int = 120,
    move_steps: int = 160,
    rotate_steps: int = 120,
    pull_steps: int = 220,
    standoff_distance: float = 0.16,
    pre_grasp_distance: float = 0.08,
    grasp_distance: float = 0.02,
    target_cabinet_extension: float = 0.14,
):
    """
    Open the cabinet slider by grasping its handle and pulling along the slide axis.
    """
    robot_state = panda.get_state()
    cabinet_state = cabinet.get_state()

    ee_pos = np.array(robot_state["ee-position"], dtype=float)
    ee_quat = list(robot_state["ee-quaternion"])
    handle_pos = np.array(cabinet_state["handle_position"], dtype=float)
    handle_z = float(handle_pos[2])

    base_quat = cabinet_state["base_quaternion"]
    base_rot = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
    slide_dir = np.array(base_rot[:, 0], dtype=float)
    slide_norm = float(np.linalg.norm(slide_dir))
    if slide_norm < 1e-6:
        slide_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        slide_dir = slide_dir / slide_norm
    # Approach from the exterior side of the handle (opposite to pull/open direction).
    approach_dir = -slide_dir

    descend_pos = np.array([ee_pos[0], ee_pos[1], handle_z], dtype=float)
    standoff_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * standoff_distance,
            handle_pos[1] - approach_dir[1] * standoff_distance,
            handle_z,
        ],
        dtype=float,
    )

    _move_linear_quat(
        panda,
        ee_pos,
        descend_pos,
        ee_quaternion=ee_quat,
        steps=descend_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )
    _move_linear_quat(
        panda,
        descend_pos,
        standoff_pos,
        ee_quaternion=ee_quat,
        steps=move_steps,
        control_dt=control_dt,
        position_gain=0.05,
    )

    approach_xy = handle_pos[:2] - standoff_pos[:2]
    approach_norm = float(np.linalg.norm(approach_xy))
    if approach_norm < 1e-6:
        approach_xy = np.array([0.0, 1.0], dtype=float)
    else:
        approach_xy = approach_xy / approach_norm

    grasp_yaw = float(np.arctan2(approach_xy[0], -approach_xy[1]))
    # Keep gripper vertical (finger motion axis near world Z) while yaw-aligning to handle approach direction.
    grasp_quat = p.getQuaternionFromEuler([np.pi / 2.0, 0.0, grasp_yaw])

    for _ in range(max(2, int(rotate_steps))):
        panda.move_to_pose(standoff_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        _step_sim(control_dt, 1)

    handle_pos = np.array(cabinet.get_state()["handle_position"], dtype=float)
    pre_grasp_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * pre_grasp_distance,
            handle_pos[1] - approach_dir[1] * pre_grasp_distance,
            handle_z,
        ],
        dtype=float,
    )
    grasp_pos = np.array(
        [
            handle_pos[0] - approach_dir[0] * grasp_distance,
            handle_pos[1] - approach_dir[1] * grasp_distance,
            handle_z,
        ],
        dtype=float,
    )

    _move_linear_quat(
        panda,
        standoff_pos,
        pre_grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )
    _move_linear_quat(
        panda,
        pre_grasp_pos,
        grasp_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.06,
    )

    panda.close_gripper()
    _step_sim(control_dt, 70)

    joint_info = p.getJointInfo(cabinet.object, 0)
    lower_limit = float(joint_info[8])
    upper_limit = float(joint_info[9])
    start_extension = float(np.clip(cabinet.get_state()["joint_angle"], lower_limit, upper_limit))
    clamped_target = float(np.clip(target_cabinet_extension, lower_limit, upper_limit))

    handle_now = np.array(cabinet.get_state()["handle_position"], dtype=float)
    ee_now = np.array(panda.get_state()["ee-position"], dtype=float)
    grasp_offset = ee_now - handle_now

    for ext in np.linspace(start_extension, clamped_target, max(2, int(pull_steps))):
        # Kinematic ramp keeps cabinet opening stable during contact-rich grasping.
        p.resetJointState(cabinet.object, 0, float(ext))
        p.setJointMotorControl2(cabinet.object, 0, p.POSITION_CONTROL, targetPosition=float(ext), force=1000.0)
        _step_sim(control_dt, 1)

        handle_now = np.array(cabinet.get_state()["handle_position"], dtype=float)
        follow_pos = handle_now + grasp_offset
        panda.move_to_pose(follow_pos.tolist(), ee_quaternion=list(grasp_quat), positionGain=0.08)
        _step_sim(control_dt, 1)

    panda.open_gripper()
    _step_sim(control_dt, 35)

    end_pos = np.array(panda.get_state()["ee-position"], dtype=float)
    retreat_pos = end_pos + np.array([0.0, 0.0, 0.08], dtype=float)
    _move_linear_quat(
        panda,
        end_pos,
        retreat_pos,
        ee_quaternion=grasp_quat,
        steps=max(2, move_steps // 2),
        control_dt=control_dt,
        position_gain=0.05,
    )

    final_extension = float(cabinet.get_state()["joint_angle"])
    return {
        "start_extension_m": start_extension,
        "target_extension_m": clamped_target,
        "final_extension_m": final_extension,
    }


def pick_cube(
    panda,
    cube,
    control_dt: float = DEFAULT_CONTROL_DT,
    steps_per_phase: int = 140,
    approach_height: float = 0.12,
    grasp_height: float = 0.0,
    grasp_settle_steps: int = 90,
):
    """Pick a cube from the table using a top-down grasp."""
    cube_state = cube.get_state()
    cube_pos = np.array(cube_state["position"], dtype=float)
    cube_yaw = cube_state["euler"][2]
    ee_yaw = cube_yaw

    _, cube_aabb_max = p.getAABB(cube.object)
    cube_top_z = float(cube_aabb_max[2])

    approach_pos = np.array([cube_pos[0], cube_pos[1], cube_top_z + approach_height], dtype=float)
    grasp_pos = np.array([cube_pos[0], cube_pos[1], cube_top_z + grasp_height], dtype=float)
    lift_pos = approach_pos + np.array([0.0, 0.0, 0.08], dtype=float)

    current_pose = np.array(panda.get_state()["ee-position"], dtype=float)
    panda.open_gripper()
    _step_sim(control_dt, 40)
    _move_linear(panda, current_pose, approach_pos, ee_yaw, steps_per_phase, control_dt)
    _move_linear(panda, approach_pos, grasp_pos, ee_yaw, steps_per_phase // 2, control_dt)
    for _ in range(max(1, int(grasp_settle_steps))):
        panda.move_to_pose(grasp_pos.tolist(), ee_rotz=ee_yaw, positionGain=0.08)
        _step_sim(control_dt, 1)

    panda.close_gripper()
    _step_sim(control_dt, 60)
    _move_linear(panda, grasp_pos, lift_pos, ee_yaw, steps_per_phase // 2, control_dt)

    return cube.get_state()["position"]
