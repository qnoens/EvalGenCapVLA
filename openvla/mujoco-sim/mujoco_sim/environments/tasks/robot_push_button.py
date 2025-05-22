"""


"""

from __future__ import annotations

import numpy as np
from dm_control import composer, mjcf
from dm_env import specs

from mujoco_sim.entities.arenas import EmptyRobotArena
from mujoco_sim.entities.camera import Camera, CameraConfig
from mujoco_sim.entities.eef.gripper import Robotiq2f85
from mujoco_sim.entities.props.switch import Switch
from mujoco_sim.entities.robots.robot import UR5e
from mujoco_sim.environments.tasks.spaces import EuclideanSpace

from scipy.spatial.transform import Rotation as R

TOP_DOWN_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


# @dataclasses.dataclass
# class RobotPushButtonConfig(TaskConfig):
#     # add these macros in the class to make it easier to use them
#     # without having to import them separately


#     # actual config
#     reward_type: str = None
#     observation_type: str = None
#     action_type: str = None


#     scene_camera_config: CameraConfig = None
#     wrist_camera_config: CameraConfig = None

#     def __post_init__(self):
#         # set default values if not set
#         # https://stackoverflow.com/questions/56665298/how-to-apply-default-value-to-python-dataclass-field-when-none-was-passed
#         self.reward_type = self.reward_type or RobotPushButtonConfig.SPARSE_REWARD
#         self.observation_type = self.observation_type or RobotPushButtonConfig.STATE_OBS
#         self.action_type = self.action_type or RobotPushButtonConfig.ABS_EEF_ACTION
#         self.scene_camera_config = self.scene_camera_config or RobotPushButtonConfig.FRONT_TILTED_CAMERA_CONFIG
#         self.wrist_camera_config = self.wrist_camera_config or RobotPushButtonConfig.WRIST_CAMERA_CONFIG

#         assert self.observation_type in RobotPushButtonConfig.OBSERVATION_TYPES
#         assert self.reward_type in RobotPushButtonConfig.REWARD_TYPES
#         assert self.action_type in RobotPushButtonConfig.ACTION_TYPES


class RobotPushButtonTask(composer.Task):
    SPARSE_REWARD = "sparse_reward"

    STATE_OBS = "state_observations"
    VISUAL_OBS = "visual_observations"

    ABS_EEF_ACTION = "absolute_eef_action"
    ABS_JOINT_ACTION = "absolute_joint_action"
    REL_EEF_ACTION = "relative_eef_action" # For OpenVLA: They are using delta actions for all datasets in translation and rotation dimensions. For the gripper action dimension they always use absolute actions.

    REWARD_TYPES = SPARSE_REWARD
    OBSERVATION_TYPES = (STATE_OBS, VISUAL_OBS)
    ACTION_TYPES = (ABS_EEF_ACTION, ABS_JOINT_ACTION, REL_EEF_ACTION)

    FRONT_TILTED_CAMERA_CONFIG = CameraConfig(np.array([0.0, -1.7, 0.7]), np.array([-0.7, -0.35, 0, 0.0]), 70) # This was: CameraConfig(np.array([0.0, -1.7, 0.7]), np.array([-0.7, -0.35, 0, 0.0]), 70) 2DIM was put to -1.0 for closer perspective
    WRIST_CAMERA_CONFIG = CameraConfig(
        position=np.array([0.0, 0.05, 0]), orientation=np.array([0.0, 0.0, 0.999, 0.04]), fov=42, name="WristCamera"
    )

    MAX_STEP_SIZE: float = 0.05
    # TIMESTEP IS MAIN DRIVER OF SIMULATION SPEED..
    # HIGHER STEPS START TO RESULT IN UNSTABLE PHYSICS
    PHYSICS_TIMESTEP: float = 0.005  # MJC DEFAULT =0.002 (500HZ)
    CONTROL_TIMESTEP: float = 0.1
    MAX_CONTROL_STEPS_PER_EPISODE: int = 100

    GOAL_DISTANCE_THRESHOLD: float = 0.05  # TASK SOLVED IF DST(POINT,GOAL) < THRESHOLD
    TARGET_RADIUS = 0.03  # RADIUS OF THE TARGET SITE

    def __init__(
        self,
        reward_type: str = SPARSE_REWARD,
        observation_type: str = VISUAL_OBS,
        action_type: str = ABS_JOINT_ACTION,
        image_resolution: int = 224,
        button_disturbances: bool = False,
    ) -> None:
        super().__init__()

        self.reward_type = reward_type
        self.observation_type = observation_type
        self.action_type = action_type
        self.image_resolution = image_resolution
        self.button_disturbances = button_disturbances

        # create arena, robot and EEF
        self._arena = EmptyRobotArena(3)
        self.robot = UR5e()
        self.gripper = Robotiq2f85()
        self.robot.attach_end_effector(self.gripper)
        self._arena.attach(self.robot, self._arena.robot_attachment_site)

        self.switch = Switch()

        # attach switch to world
        self._arena.attach(self.switch)

        # create robot workspace and all the spawn spaces
        self.robot_workspace = EuclideanSpace((-0.2, 0.2), (-0.6, -0.3), (0.02, 0.3))
        self.robot_spawn_space = EuclideanSpace((-0.2, 0.2), (-0.6, -0.3), (0.02, 0.3))
        self.target_spawn_space = EuclideanSpace((-0.2, 0.2), (-0.6, -0.3), (0.0, 0.1))

        # for debugging camera views etc: add workspace to scene
        # self.workspace_geom = self.robot_workspace.create_visualization_site(self._arena.mjcf_model.worldbody,"robot-workspace")

        # add Camera to scene
        self.camera = Camera(self.FRONT_TILTED_CAMERA_CONFIG)
        self._arena.attach(self.camera)

        self.wrist_camera = Camera(self.WRIST_CAMERA_CONFIG)
        self.wrist_camera.observables.rgb_image.name = "wrist_camera_rgb_image"
        self.robot.attach(self.wrist_camera, self.robot.flange)

        # create additional observables / Sensors
        self._task_observables = {}

        self._configure_observables()

        # set timesteps
        # has to happen here as the _arena has to be available.
        self.physics_timestep = self.PHYSICS_TIMESTEP
        self.control_timestep = self.CONTROL_TIMESTEP

        self.robot_end_position = np.array([-0.3, -0.2, 0.3])  # end position of the robot once the switch is activated

    def _configure_observables(self):
        if self.action_type == RobotPushButtonTask.ABS_EEF_ACTION or self.action_type == RobotPushButtonTask.REL_EEF_ACTION:
            self.robot.observables.tcp_pose.enabled = True # was tcp_position before
        else:
            self.robot.observables.joint_configuration.enabled = True
        # TODO: add gripper state.

        if self.observation_type == RobotPushButtonTask.STATE_OBS:
            self.switch.observables.position.enabled = True
            self.switch.observables.active.enabled = True

        elif self.observation_type == RobotPushButtonTask.VISUAL_OBS:
            self.camera.observables.rgb_image.enabled = True
            self.wrist_camera.observables.rgb_image.enabled = True

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        robot_initial_pose = self.robot_spawn_space.sample()
        robot_initial_pose = np.concatenate([robot_initial_pose, TOP_DOWN_QUATERNION])
        self.robot.set_tcp_pose(physics, robot_initial_pose)

        # switch  position
        switch_position = self.target_spawn_space.sample()
        self.switch.set_pose(physics, position=switch_position)

        # print(f"target position: {target_position}")
        # print(self.goal_position_observable(physics))

    @property
    def root_entity(self):
        return self._arena

    def before_step(self, physics, action, random_state):
        if self.action_type == RobotPushButtonTask.ABS_EEF_ACTION:
            assert action.shape == (4,)
            griper_target = action[3]
            robot_position = action[:3]

            self.gripper.move(physics, griper_target)
            self.robot.servoL(physics, np.concatenate([robot_position, TOP_DOWN_QUATERNION]), self.control_timestep)

        elif self.action_type == RobotPushButtonTask.ABS_JOINT_ACTION:
            assert action.shape == (7,)
            gripper_target = action[6]
            joint_configuration = action[:6]
            self.gripper.move(physics, gripper_target)
            self.robot.servoJ(physics, joint_configuration, self.control_timestep)

        elif self.action_type == RobotPushButtonTask.REL_EEF_ACTION:
            assert action.shape == (7,)
            gripper_target = action[6]

            current_robot_position = self.robot.get_tcp_pose(physics).copy()
            target_robot_position = current_robot_position[:3] + action[:3]
    
            # We want the action to be Euler angles, so we need to convert the action to a quaternion since servoL requires a quaternion
            current_quaternion = current_robot_position[3:]

            current_rot_matrix = R.from_quat(current_quaternion).as_matrix()
            relative_rot_matrix = R.from_euler('xyz', action[3:6]).as_matrix()

            target_rot_matrix = current_rot_matrix @ relative_rot_matrix
            target_quaternion = R.from_matrix(target_rot_matrix).as_quat()

            # Reformat gripper [expected value is either 0 or 1]
            if gripper_target < 0.5:
                gripper_target = 0.0
            else:
                gripper_target = self.gripper.open_distance

            self.gripper.move(physics, gripper_target)
            self.robot.servoL(physics, np.concatenate([target_robot_position, target_quaternion]), self.control_timestep) # Replaced target_quaternion with TOP_DOWN_QUATERNION

    def after_step(self, physics, random_state):
        # if the button is active, with some probability make it inactive
        if self.button_disturbances:
            if (
                self.switch.is_active and not self.switch._is_pressed and random_state.rand() < 0.01
            ):  # (0.99)**30 = 0.74 probability to reach end pose before disturbance.
                self.switch.deactivate(physics)

    def get_reward(self, physics):

        if self.reward_type == RobotPushButtonTask.SPARSE_REWARD:
            return self.is_goal_reached(physics) * 1.0

    def action_spec(self, physics):
        del physics
        # bound = np.array([self.MAX_STEP_SIZE, self.MAX_STEP_SIZE])
        # normalized action space, rescaled in before_step
        if self.action_type == RobotPushButtonTask.ABS_EEF_ACTION:
            return specs.BoundedArray(
                shape=(4,),
                dtype=np.float64,
                minimum=[
                    self.robot_workspace.x_range[0],
                    self.robot_workspace.y_range[0],
                    self.robot_workspace.z_range[0],
                    0.0,
                ],
                maximum=[
                    self.robot_workspace.x_range[1],
                    self.robot_workspace.y_range[1],
                    self.robot_workspace.z_range[1],
                    self.gripper.open_distance,
                ],
            )
        elif self.action_type == RobotPushButtonTask.ABS_JOINT_ACTION:
            return specs.BoundedArray(
                shape=(7,),
                dtype=np.float64,
                minimum=[
                    -3.14,
                ]
                * 6
                + [0.0],
                maximum=[3.14] * 6 + [0.085],
            )
        elif self.action_type == RobotPushButtonTask.REL_EEF_ACTION:
            biggest_xdiff = self.robot_workspace.x_range[1] - self.robot_workspace.x_range[0]
            biggest_ydiff = self.robot_workspace.y_range[1] - self.robot_workspace.y_range[0]
            biggest_zdiff = self.robot_workspace.z_range[1] - self.robot_workspace.z_range[0]
            angle_limit = 2*np.pi
            pitch_limit = np.pi
            open_gripper = self.gripper.open_distance
            return specs.BoundedArray(
                shape=(7,),
                dtype=np.float64,
                minimum=[-biggest_xdiff, -biggest_ydiff, -biggest_zdiff, -angle_limit, -pitch_limit, -angle_limit, 0.0],
                maximum=[biggest_xdiff, biggest_ydiff, biggest_zdiff, angle_limit, pitch_limit, angle_limit, open_gripper],
            )

    def is_goal_reached(self, physics) -> bool:
        return (
            self.switch.is_active
            # and np.linalg.norm(self.robot.get_tcp_pose(physics)[:3] - self.robot_end_position)
            # < self.GOAL_DISTANCE_THRESHOLD
        )

    def should_terminate_episode(self, physics):
        return self.is_goal_reached(physics)

    def get_discount(self, physics):
        if self.should_terminate_episode(physics):
            return 0.0
        else:
            return 1.0

    def create_random_policy(self):
        physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        spec = self.action_spec(physics)

        def random_policy(time_step):
            # return np.array([0.01, 0])
            return np.random.uniform(spec.minimum, spec.maximum, spec.shape)

        return random_policy

    def create_demonstration_policy(self, environment):  # noqa C901
        def demonstration_policy(time_step: composer.TimeStep):
            physics = environment.physics
            # get the current physics state
            # get the current robot pose
            robot_position = self.robot.get_tcp_pose(physics).copy()
            robot_position = robot_position[:3]

            # get the current target pose
            switch_position = self.switch.get_position(physics).copy()
            target_position = switch_position
            # target_position[2] += 0.
            is_switch_active = self.switch.is_active

            # if robot is not above the switch and switch is not active, this is phase 1
            # if robot is above that pose and switch is not active, this is phase 2
            # if switch is active, this is phase 3

            if is_switch_active:
                phase = 3
            elif (
                robot_position[2] > target_position[2]
                and np.linalg.norm(robot_position[:2] - target_position[:2]) < 0.01
                and not is_switch_active
            ):
                phase = 2
            else:
                phase = 1
            # print(f"phase: {phase}")
            # print(f"target position: {target_position}")
            if phase == 1:
                # move towards the target, first move up to avoid collisions
                if robot_position[2] < target_position[2] + 0.02:
                    action = robot_position.copy() # if we don't do copy here, we are changing the original robot position
                    action[2] = target_position[2] + 0.05
                    #print(f"moving up to {action}")
                else:
                    action = target_position
                    action[2] = target_position[2] + 0.05
                    #print("moving towards")

            if phase == 2:
                # move down to the target
                action = target_position

            # if phase == 3:
            #     # move to the end pose
            #     action = self.robot_end_position.copy()
            #     if np.linalg.norm(switch_position[:2] - robot_position[:2]) < 0.05:
            #         action[2] = target_position[2] + 0.1  # avoid touching button upon moving to end position

            # calculate the action to reach the target
            difference = action - robot_position[:3]

            MAX_SPEED = 0.5
            if np.max(np.abs(difference)) > MAX_SPEED * environment.control_timestep():
                difference = difference * MAX_SPEED / np.max(np.abs(difference)) * environment.control_timestep()
            action = robot_position[:3] + difference

            # if needed, convert action to joint configuration
            if self.action_type == RobotPushButtonTask.ABS_JOINT_ACTION:
                tcp_pose = np.concatenate([action, TOP_DOWN_QUATERNION])
                current_joint_positions = self.robot.get_joint_positions(physics)
                action = self.robot.get_joint_positions_from_tcp_pose(tcp_pose, current_joint_positions)
            elif self.action_type == RobotPushButtonTask.REL_EEF_ACTION:
                robot_pose = self.robot.get_tcp_pose(physics).copy()
                current_quaternion = robot_pose[3:]
                target_quaternion = TOP_DOWN_QUATERNION

                current_rot_matrix = R.from_quat(current_quaternion).as_matrix()
                target_rot_matrix = R.from_quat(target_quaternion).as_matrix()

                relative_rot_matrix = current_rot_matrix.T @ target_rot_matrix 
                relative_euler = R.from_matrix(relative_rot_matrix).as_euler('xyz')

                action = difference
                action = np.concatenate([action, relative_euler, [0.0]])
                return action

            # add gripper, which is always closed
            action = np.concatenate([action, [0.0]])
            return action

        return demonstration_policy
    
    # ADDED this for inference simulation
    def demonstration_policy(self, time_step: composer.TimeStep, environment):
        physics = environment.physics
        # get the current physics state
        # get the current robot pose
        robot_position = self.robot.get_tcp_pose(physics).copy()
        robot_position = robot_position[:3]

        # get the current target pose
        switch_position = self.switch.get_position(physics).copy()
        target_position = switch_position
        # target_position[2] += 0.
        is_switch_active = self.switch.is_active

        # if robot is not above the switch and switch is not active, this is phase 1
        # if robot is above that pose and switch is not active, this is phase 2
        # if switch is active, this is phase 3

        if is_switch_active:
            phase = 3
        elif (
            robot_position[2] > target_position[2]
            and np.linalg.norm(robot_position[:2] - target_position[:2]) < 0.01
            and not is_switch_active
        ):
            phase = 2
        else:
            phase = 1
        # print(f"phase: {phase}")
        # print(f"target position: {target_position}")
        if phase == 1:
            # move towards the target, first move up to avoid collisions
            if robot_position[2] < target_position[2] + 0.02:
                action = robot_position.copy() # if we don't do copy here, we are changing the original robot position
                action[2] = target_position[2] + 0.05
                #print(f"moving up to {action}")
            else:
                action = target_position
                action[2] = target_position[2] + 0.05
                #print("moving towards")

        if phase == 2:
            # move down to the target
            action = target_position

        # if phase == 3:
        #     # move to the end pose
        #     action = self.robot_end_position.copy()
        #     if np.linalg.norm(switch_position[:2] - robot_position[:2]) < 0.05:
        #         action[2] = target_position[2] + 0.1  # avoid touching button upon moving to end position

        # calculate the action to reach the target
        difference = action - robot_position[:3]

        MAX_SPEED = 0.5
        if np.max(np.abs(difference)) > MAX_SPEED * environment.control_timestep():
            difference = difference * MAX_SPEED / np.max(np.abs(difference)) * environment.control_timestep()
        action = robot_position[:3] + difference

        # if needed, convert action to joint configuration
        if self.action_type == RobotPushButtonTask.ABS_JOINT_ACTION:
            tcp_pose = np.concatenate([action, TOP_DOWN_QUATERNION])
            current_joint_positions = self.robot.get_joint_positions(physics)
            action = self.robot.get_joint_positions_from_tcp_pose(tcp_pose, current_joint_positions)
        elif self.action_type == RobotPushButtonTask.REL_EEF_ACTION:
            robot_pose = self.robot.get_tcp_pose(physics).copy()
            current_quaternion = robot_pose[3:]
            target_quaternion = TOP_DOWN_QUATERNION

            current_rot_matrix = R.from_quat(current_quaternion).as_matrix()
            target_rot_matrix = R.from_quat(target_quaternion).as_matrix()

            relative_rot_matrix = current_rot_matrix.T @ target_rot_matrix 
            relative_euler = R.from_matrix(relative_rot_matrix).as_euler('xyz')

            action = difference
            action = np.concatenate([action, relative_euler, [0.0]])
            return action

        # add gripper, which is always closed
        action = np.concatenate([action, [0.0]])
        return action
    
    # Helper functions for converting between euler angles and quaternions
    def convert_euler_to_quaternion(self, roll, pitch, yaw):
        w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        return np.array([w, x, y, z])
    
    def convert_quaternion_to_euler(self, w, x, y, z):
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return roll, pitch, yaw



    

if __name__ == "__main__":
    #from dm_control import viewer
    from dm_control.composer import Environment
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2
    import os
    import dm_env

    task = RobotPushButtonTask(
        reward_type=RobotPushButtonTask.SPARSE_REWARD,
        observation_type=RobotPushButtonTask.VISUAL_OBS,
        action_type=RobotPushButtonTask.REL_EEF_ACTION,
        button_disturbances=True,
    )

    # dump task xml

    # mjcf.export_with_assets(task._arena.mjcf_model, ".")

    environment = Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        time_limit=task.MAX_CONTROL_STEPS_PER_EPISODE * task.CONTROL_TIMESTEP,
    )

    timestep = environment.reset()

    # ADDED

    # Define video constants
    frame_size = (task.image_resolution, task.image_resolution)  # Assuming task.camera has width and height
    fps = 30  # Adjust as needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files

    for i in range(2):
        # Define the video writer
        os.makedirs("videos", exist_ok=True) 
        video_path = f"videos/test_video_{i}.mp4"
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        # Initialize the episode
        step = 0
        done = False
        termination = False
        truncation = False

        while not done: # Add a max amount of steps (200 def seems to be enough)
            #print(task.robot.get_tcp_pose(environment.physics))
            # 1) Get action
            action = task.demonstration_policy(timestep, environment) # This now needs to be changed by the vla prediction...
            #print("ACTION", action)

            # EXTRA: check if the action makes sense given the input image
            # if step % 10 == 0:
            #     input_image = task.camera.get_rgb_image(environment.physics)
            #     print("X +=", action[0], "Y +=", action[1], "Z +=", action[2], "Roll +=", action[4], "Pitch +=", action[5], "Yaw +=", action[6])
            #     plt.imshow(input_image)
            #     plt.show()

            # 2) Take action
            ts = environment.step(action)

            # 3) Get next timestep
            timestep = environment.physics

            # 4) New code to get the image
            image: Image.Image = Image.fromarray(task.camera.get_rgb_image(environment.physics))
            image_np = np.array(image)  # Convert PIL image to NumPy array for OpenCV
            
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV compatibility
            video_writer.write(image_bgr)  # Write frame to video

            # 5) Prepare for next iteration    
            step += 1

            # Episode has been completed succesfully => termination
            if ts.reward == 1:
                termination = True
            # Episode has been taking too long => truncation     
            if ts.step_type == dm_env.StepType.LAST:
                truncation = True

            # Episode has been completed succesfully or taking too long => done
            done = termination or truncation

        print(step)

        video_writer.release()
        new_path = f"videos/test_video_{i}_{task.switch.is_active}.mp4"
        os.rename(video_path, new_path)
        print(f"Video saved at {new_path}")

        environment.reset()

    #viewer.launch(environment, policy=task.create_demonstration_policy(environment))
