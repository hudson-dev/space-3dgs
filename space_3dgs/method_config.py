"""nerfstudio registration of the `space-3dgs` method.

Recipe (see README "Method"): antialiased splatfacto initialised from the SfM
point cloud, one achromatic appearance code per capture sequence, poses held
fixed, 90k steps with densification to 60k and pruning kept active to the end.
"""
from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from space_3dgs.appearance_model import SequenceAppearanceModelConfig
from space_3dgs.sequence_dataparser import SequenceNerfstudio

MAX_STEPS = 90_000


def _optimizers(max_steps: int = MAX_STEPS):
    return {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=max_steps),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None,
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=max_steps, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "appearance": {
            "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-4, max_steps=max_steps, warmup_steps=1000, lr_pre_warmup=1e-4
            ),
        },
    }


space_3dgs = MethodSpecification(
    config=TrainerConfig(
        method_name="space-3dgs",
        steps_per_eval_image=3000,
        steps_per_eval_batch=0,
        steps_per_save=15_000,
        steps_per_eval_all_images=MAX_STEPS,
        max_num_iterations=MAX_STEPS,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    _target=SequenceNerfstudio,
                    load_3D_points=True,
                    # Hold out every 8th frame of the name-sorted list.
                    eval_mode="interval",
                    eval_interval=8,
                ),
                cache_images_type="uint8",
            ),
            model=SequenceAppearanceModelConfig(
                num_sequences=0,  # from the dataparser
                rasterize_mode="antialiased",
                # Poses come from the global SfM solve and are photometrically
                # self-consistent to sub-pixel level; in-training pose
                # refinement adds nothing and drifts the frame vs eval cameras.
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                cull_alpha_thresh=0.05,
                densify_grad_thresh=0.0008,
                # ~5k training images: a 10k-step opacity-reset period so each
                # reset is followed by useful refinement (reset_alpha_every is
                # in units of refine_every=100 steps).
                reset_alpha_every=100,
                stop_split_at=60_000,
                cull_stop_iter=MAX_STEPS,
                stop_screen_size_at=12_000,
                resolution_schedule=6_000,
                sh_degree_interval=2_000,
            ),
        ),
        optimizers=_optimizers(),
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description=(
        "Antialiased Gaussian splatting on fixed global-SfM poses with one "
        "achromatic appearance code per capture sequence and culling kept "
        "active after densification stops"
    ),
)
