"""LGES task 3 (suction cup on the left arm, head camera) — data config, dataset class and mixtures.

The datasets are written by LGES_Sim/vla/npz2lerobot.py into LGES_Sim/data/vla (= datasets.vla_data.data_root_dir):
one episode per pick / place segment, 5 Hz, frames stored as jpg files whose paths sit in the parquet.
"""

from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class LGESDataset(LeRobotSingleDataset):
    """LeRobotSingleDataset whose frames keep their own size (480x300). The base class squashes every frame to
    224x224, which leaves Qwen3.5 (32 px per visual token) about 49 tokens for a cm-level task. Set
    `datasets.vla_data.lges_image_size: [w, h]` to resize here instead; leave `obs_image_size` unset, or the
    framework resizes again."""

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        size = self.data_cfg.get("lges_image_size", None) if self.data_cfg is not None else None
        images = [Image.fromarray(data[key][0]) for key in self.modality_keys["video"]]
        sample["image"] = [im.resize(tuple(size)) for im in images] if size else images
        return sample


class LGESCupDataConfig:
    """5-D action (cup tip delta x, y, z, yaw in base_link + pump), 6-D state (cup tip x, y, z, cos yaw, sin yaw,
    pump). `held` and the wrist wrench are in the datasets too (state.held, state.wrench) but not used yet."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.head"]
    state_keys = ["state.cup_pos", "state.cup_yaw", "state.pump"]
    action_keys = ["action.delta_pos", "action.delta_yaw", "action.pump"]
    # Per-key dims for PolicyNormProcessor (3+1+1 = 5-D action, 3+2+1 = 6-D state)
    action_key_dims = {"action.delta_pos": 3, "action.delta_yaw": 1, "action.pump": 1}
    state_key_dims = {"state.cup_pos": 3, "state.cup_yaw": 2, "state.pump": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))          # 8 steps at 5 Hz = 1.6 s

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                # cup_yaw is (cos, sin): already in [-1, 1], left as it is
                normalization_modes={"state.cup_pos": "min_max", "state.pump": "binary"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                # q99: the step sizes have long tails (a 5 Hz step is 0 to ~10 cm), min_max would squeeze the small ones
                normalization_modes={
                    "action.delta_pos": "q99",
                    "action.delta_yaw": "q99",
                    "action.pump": "binary",
                },
            ),
        ])

    def make_dataset(self, dataset_path, modality_configs, transforms, embodiment_tag, video_backend,
                     delete_pause_frame, data_cfg, dataset_name=None, **kwargs):
        return LGESDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )


ROBOT_TYPE_CONFIG_MAP = {
    "lges_cup": LGESCupDataConfig(),
}

DATASET_NAMED_MIXTURES = {
    "lges_sim": [
        ("lges_sim_film", 1.0, "lges_cup"),
        ("lges_sim_vis", 1.0, "lges_cup"),
    ],
    "lges_sim_film": [("lges_sim_film", 1.0, "lges_cup")],
    "lges_sim_vis_holdout": [("lges_sim_vis_holdout", 1.0, "lges_cup")],
}
