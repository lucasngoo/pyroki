"""Bimanual IK

Same as 01_basic_ik.py, but with two end effectors!
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks

from mujoco_ar import MujocoARConnector
from trimesh.transformations import quaternion_from_matrix


def main():
    """Main function for bimanual IK."""

    urdf = load_robot_description("agibot_x1_description")
    target_link_names = ["right_wrist_pitch", "left_wrist_pitch"]

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2, position=(0.41, -0.3, 0.56), wxyz=(0, 0, 1, 0)
    )
    ik_target_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2, position=(0.41, 0.3, 0.56), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    
    # Add a slider to control the smoothness weight
    smoothness_slider = server.gui.add_slider(
        "Smoothness Weight", 
        min=0.0, 
        max=50.0, 
        step=1.0, 
        initial_value=10.0,
    )

    # Initialize previous solution
    previous_solution = None

    connector = MujocoARConnector()
    connector.start()

    while True:
        data = connector.get_latest_data()
        if data["position"] is None:
            continue

        T = np.eye(4)
        T[:3, :3] = np.asarray(data["rotation"])
        T[:3, 3] = np.asarray(data["position"])
        pos = T[:3, 3]
        quat = quaternion_from_matrix(T)

        # Solve IK.
        start_time = time.time()
        
        solution = pks.solve_ik_with_multiple_targets(
            robot=robot,
            target_link_names=target_link_names,
            target_positions=np.array([pos, ik_target_1.position]),
            target_wxyzs=np.array([quat, ik_target_1.wxyz]),
            prior_configuration=previous_solution,
            smoothness_weight=smoothness_slider.value,
        )
        print("solution", solution)
        
        # Save current solution for next iteration
        previous_solution = solution

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
