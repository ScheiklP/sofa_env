import os
import csv
from typing import Dict, Optional, Union


class TissueManipulationSceneLogger:
    """Logger  for the Tissue Manipulation (TiM) Scene.

    Notes:
        - This class is utilized to log the outcome for each episode during the training.
            - IDP, TTP, TGP, distance(IDP, TTP)_start, success

    Args:
        log_path (Optional[str]): Path to the log file.
    """

    def __init__(self, log_path: Optional[str] = None) -> None:
        """Logger object to save environment infos of completed episodes to a csv."""
        self.log_path = log_path
        self._id = f"{hex(id(self))}"
        self._header_written = False

    def log(self, success: bool, env_info: Dict) -> Union[bool, None]:
        if self.log_path is None:
            return False

        env_info.update({"success": success})
        with open(os.path.join(self.log_path, str(self._id) + "_episode_info.csv"), "a") as csv_file:
            writer = csv.writer(csv_file)
            if not self._header_written:
                writer.writerow(env_info.keys())
                self._header_written = True

            writer.writerow(env_info.values())
