import numpy as np
from enum import Enum, unique
from typing import Union, Optional, Tuple


@unique
class WorkspaceType(Enum):
    """Two Options for Workspace specification.

    ``TISSUE_ALIGNED`` Places the workspace in coronal plane of the tissue (2D).
    ``GENERAL`` 3D manipulation inside the workspace. This has to be set in ``scene_description.py``
    """

    TISSUE_ALIGNED = ("tissue_aligned", 2)
    GENERAL = ("general_3D_workspace", 3)


class Workspace:
    """Workspace for the gripper object in the Tissue Manipulation Scene.

       Args:
           bounds (Union[np.ndarray, Tuple[float, float, float, float, float, float]]): Workspace limits in scene coordinates. ``[xmin, xmax, ymin, ymax, zmin, zmax]``.
           translation (Tuple[float, float, float]): Additional translation of the workspace.
           tissue_thickness (float): Thickness of the deformable tissue in the scene. Used to place the workspace in case of 2D. Only relevant if ``workspace_type == WorkspaceType.TISSUE_ALIGNED``.
           gripper_offset_limits (dict): Defines the limits for random sampling of the grasping point.
           workspace_type (WorkspaceType): Defines the type of the workspace: 2D vs. 3D.
       """

    def __init__(
            self,
            bounds: Union[np.ndarray, Tuple[float, float, float, float, float, float]],
            translation: Union[np.ndarray, Tuple[float, float, float]] = (0.0, 0.0, 0.0),
            tissue_thickness: Optional[float] = 0.01,
            gripper_offset_limits: Optional[dict] = None,
            workspace_type: Optional[WorkspaceType] = WorkspaceType.TISSUE_ALIGNED,
    ) -> None:
        # setup the workspace
        self.mode = workspace_type
        self.tissue_thickness = tissue_thickness
        translation = np.asarray(translation)

        if self.mode == WorkspaceType.TISSUE_ALIGNED:
            if self.tissue_thickness <= 0.0:
                raise ValueError("When using the tissue aligned workspace the tissue thickness must be greater than 0.")

            if len(bounds) != 4:
                raise ValueError(f"Workspace requires lower and upper limits for X and Z passed as [xmin, xmax, zmin, zmax]. Received {bounds}.")

            self._low = np.array([bounds[0], -self.tissue_thickness / 2, bounds[2]]) + translation
            self._high = np.array([bounds[1], -self.tissue_thickness / 2, bounds[3]]) + translation
        elif self.mode == WorkspaceType.GENERAL:
            if len(bounds) != 6:
                raise ValueError(f"Workspace requires lower and upper limits for X, Y and Z passed as [xmin, xmax, ymin, ymax, zmin, zmax]. Received {bounds}.")

            self._low = np.array([bounds[0], bounds[2], bounds[4]]) + translation
            self._high = np.array([bounds[1], bounds[3], bounds[5]]) + translation

        # Set a default random number generator in init (for scene creation) that can be overwritten from the env with the seed method
        self.rng = np.random.default_rng()

        # Setup grasping offset random sampling limits
        if gripper_offset_limits is None:
            gripper_offset_limits = {
                "x_lim": (-5, 5),
                "y_lim": None,
                "z_lim": (-5, 0),
            }
        else:
            if not all(map(lambda key: key in gripper_offset_limits, ("x_lim", "y_lim", "z_lim"))):
                raise KeyError(f"If setting manual values for gripper_offset_limits, please pass values for x_lim, y_lim, and z_lim. Received {gripper_offset_limits}.")

        self.gripper_off_x_lim = gripper_offset_limits["x_lim"]
        self.gripper_off_y_lim = gripper_offset_limits["y_lim"]
        self.gripper_off_z_lim = gripper_offset_limits["z_lim"]

    def get_randomize_gripper_limits(self) -> Tuple:
        """Returns randomization offset limits for an ``AttachedGripper``."""
        return self.gripper_off_x_lim, self.gripper_off_y_lim, self.gripper_off_z_lim

    def get_bounds_as_tuple(self) -> Tuple:
        """Returns lower and upper bounds of the workspace as Tuple."""
        return tuple(self._low) + tuple(self._high)

    def get_low(self) -> np.ndarray:
        """Returns lower workspace limits"""
        return self._low

    def get_high(self) -> np.ndarray:
        """Returns upper workspace limits"""
        return self._high

    def in_workspace(self, pos: np.ndarray, ret_distance: bool = False) -> Union[bool, Tuple[bool, float]]:
        """Returns bool and manhattan distance."""
        is_in_workspace = np.all(np.logical_and(self._low <= pos, pos <= self._high))

        if ret_distance:
            distance = np.sum(x for x in pos - self._high if x > 0) + np.sum(x for x in self._low - pos if x > 0)
            return is_in_workspace, distance

        return is_in_workspace

    def random_pose(self, return_position: bool = False, margin: Union[float, np.ndarray] = 0.01, shift_workspace: Optional[np.ndarray] = None) -> tuple:
        """Samples a random pose in workspace. Margin can be set to sample a random pose for the VisualTarget (IDP).

        Args:
            return_position (bool): Whether to return the random sampled Cartesian coordinates or the full pose (7).
            margin (Union[float, np.ndarray]): Margin values added to the workspace boundaries (xxyyzz). Either a single value or same size as bounds.
            shift_workspace (Optional[np.ndarray]): Shifts the center of the workspace according to the values in the array.

        Returns:
            new_pose (tuple): random SOFA pose (7) inside the workspace. If ``return_position==True``, returns point coordinates (3)
        """
        random_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        delta = [0.0, 0.0, 0.0]
        if shift_workspace is not None:
            # Shift center of workspace to manipulation target position
            delta = shift_workspace - np.mean([self._low, self._high], axis=0)
            if self.mode == WorkspaceType.TISSUE_ALIGNED:
                delta[1] = 0.0

        if isinstance(margin, float):
            margin = np.asarray([[margin] * 3, [-margin] * 3]).reshape(-1, )
            if self.mode == WorkspaceType.TISSUE_ALIGNED:
                margin[1] = -self.tissue_thickness / 2
                margin[4] = -self.tissue_thickness / 2

        for i in range(len(self._low)):
            random_pose[i] = self.rng.uniform(self._low[i] + margin[i], self._high[i] + margin[3 + i]) + delta[i]

        new_pose = tuple(random_pose)
        if return_position:
            return new_pose[:3]

        return new_pose

    def seed(self, seed: Union[int, np.random.SeedSequence]) -> None:
        """Creates a random number generator from a seed."""
        self.rng = np.random.default_rng(seed)

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        """Transforms action (2D) to sofa translation (3D)"""
        if self.mode == WorkspaceType.TISSUE_ALIGNED:
            return np.asarray([action[0], 0.0, action[1]])
        else:
            return np.asarray([action[0], action[1], action[2]])

    def safe_normalization(self, position):
        """Normalize position in 2D/3D Workspace"""
        mask = [True] * 3
        if self.mode == WorkspaceType.TISSUE_ALIGNED:
            mask[1] = False

        normalized_in_workspace = 2 * (position - self._low)[mask] / (self._high - self._low)[mask] - 1

        if self.mode == WorkspaceType.TISSUE_ALIGNED:
            ret = [0.0] * 3
            ret[0] = normalized_in_workspace[0]
            ret[-1] = normalized_in_workspace[-1]
            normalized_in_workspace = ret

        return np.asarray(normalized_in_workspace).reshape(-1)
