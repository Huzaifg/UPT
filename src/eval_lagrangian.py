from utils.pytorch_cuda_timing import cuda_start_event, cuda_end_event
from utils.logging_util import add_global_handlers, log_from_all_ranks
from utils.kappaconfig.util import get_stage_hp
from train_stage import train_stage
from distributed.run import run_single_or_multiprocess, run_managed
from distributed.config import barrier, get_rank, get_local_rank, get_world_size, is_managed
from datasets import dataset_from_kwargs
from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider
from configs.cli_args import parse_run_cli_args
from trainers import trainer_from_kwargs
from kappadata.wrappers import ModeWrapper
from providers.path_provider import PathProvider
from initializers.previous_run_initializer import PreviousRunInitializer
import kappaprofiler as kp
import os
import logging
import einops
import torch
import torch.nn.functional as F

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create
from utils.data_container import DataContainer

from utils.version_check import check_versions


if __name__ == "__main__":
    check_versions(verbose=False)
    cli_args = parse_run_cli_args()

    # Need static confid for the data sets
    static_config = StaticConfig(
        uri="static_config.yaml", datasets_were_preloaded=cli_args.datasets_were_preloaded)

    # Read the same config file as used for training
    stage_hp = get_stage_hp(
        cli_args.hp,
        template_path="zztemplates",
        testrun=cli_args.testrun,
        minmodelrun=cli_args.minmodelrun,
        mindatarun=cli_args.mindatarun,
        mindurationrun=cli_args.mindurationrun,
    )
    stage_name = stage_hp.get("stage_name", "default_stage")
    path_provider = PathProvider(
        output_path=static_config.output_path,
        model_path=static_config.model_path,
        stage_name=stage_name,
        stage_id=1,
        temp_path=static_config.temp_path,
    )

    # =============================================================================
    # First create the datasets and data container
    # =============================================================================
    datasets = {}
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static_config.get_global_dataset_paths(),
        local_dataset_path=static_config.get_local_dataset_path(),
        data_source_modes=static_config.get_data_source_modes(),
    )

    for dataset_key, dataset_kwargs in stage_hp["datasets"].items():
        logging.info(f"initializing {dataset_key}")
        datasets[dataset_key] = dataset_from_kwargs(
            dataset_config_provider=dataset_config_provider,
            path_provider=path_provider,
            **dataset_kwargs,
        )
    data_container_kwargs = {}
    if "prefetch_factor" in stage_hp:
        data_container_kwargs["prefetch_factor"] = stage_hp.pop(
            "prefetch_factor")
    if "max_num_workers" in stage_hp:
        data_container_kwargs["max_num_workers"] = stage_hp.pop(
            "max_num_workers")
    # Create the data container
    data_container = DataContainer(
        **datasets,
        num_workers=cli_args.num_workers,
        pin_memory=cli_args.pin_memory,
        seed=0,
        **data_container_kwargs,
    )


    # =============================================================================
    # Initialize trainer - Onlt so that we can get the input and output shapes
    # =============================================================================
    trainer_kwargs = {}
    if "max_batch_size" in stage_hp:
        trainer_kwargs["max_batch_size"] = stage_hp.pop("max_batch_size")
    trainer = trainer_from_kwargs(
        data_container=data_container,
        device='cuda:0',
        sync_batchnorm=cli_args.sync_batchnorm or static_config.default_sync_batchnorm,
        path_provider=path_provider,
        **stage_hp["trainer"],
        **trainer_kwargs,
    )

    # =============================================================================
    # Generate model and load in each of the weights
    # =============================================================================
    model = model_from_kwargs(
        **stage_hp["model"],
        input_shape=trainer.input_shape,
        output_shape=trainer.output_shape,
        update_counter=trainer.update_counter,
        path_provider=path_provider,
        data_container=data_container,
    )

    # Initilize weights using a previous_run_initializer for each part of model
    initializer = PreviousRunInitializer(stage_id = "1", model_name = "lagrangian_simformer_model.encoder", checkpoint = "best_model.ekin.valid_rollout.mse", stage_name = "stage1", path_provider = path_provider, wandb_folder = "l6g2r8dz")

    initializer.init_weights(model.encoder)

    initializer = PreviousRunInitializer(stage_id = "1", model_name = "lagrangian_simformer_model.decoder", checkpoint = "best_model.ekin.valid_rollout.mse", stage_name = "stage1", path_provider = path_provider, wandb_folder = "l6g2r8dz")

    initializer.init_weights(model.decoder)


    initializer = PreviousRunInitializer(stage_id = "1", model_name = "lagrangian_simformer_model.latent", checkpoint = "best_model.ekin.valid_rollout.mse", stage_name = "stage1", path_provider = path_provider, wandb_folder = "l6g2r8dz")

    initializer.init_weights(model.latent)


    initializer = PreviousRunInitializer(stage_id = "1", model_name = "lagrangian_simformer_model.conditioner_encoder", checkpoint = "best_model.ekin.valid_rollout.mse", stage_name = "stage1", path_provider = path_provider, wandb_folder = "l6g2r8dz")

    initializer.init_weights(model.conditioner)


    # =============================================================================
    # Evalaute model
    # =============================================================================
    mode = 'x curr_pos curr_pos_full edge_index edge_index_target timestep target_vel_large_t target_acc all_pos all_vel target_pos target_pos_encode perm' # From the logs

    test_dataset, main_collator = trainer.data_container.get_dataset("test_rollout", mode=mode)
    main_sampler = trainer.data_container.get_main_sampler(
        train_dataset=test_dataset
    )
    data_loader = trainer.data_container.get_data_loader(
            main_sampler=main_sampler,
            main_collator=main_collator,
            batch_size=100,
            epochs=1,
            updates=None,
            samples=None,
            configs=[]
    )
    data_iter = iter(data_loader)
    batch,ctx = next(data_iter)

    # Directly picked from a callback 
    # Basically extracts all the relevant pieces of information from the big dataset
    # x is needed to encode the first latent
    # These are probably the first 2 velocities need to kick off the model
    x = ModeWrapper.get_item(mode=trainer.dataset_mode, item="x", batch=batch)
    x = x.to(model.device, non_blocking=True)
    # # all positions of the sequence are needed for decoding
    all_pos = ModeWrapper.get_item(mode=trainer.dataset_mode, item="all_pos", batch=batch)
    all_pos = all_pos.to(model.device, non_blocking=True)
    # # all velocities are needed to compare the predictions
    all_vel = ModeWrapper.get_item(mode=trainer.dataset_mode, item="all_vel", batch=batch)
    all_vel = all_vel.to(model.device, non_blocking=True)
    # # get the timestep
    if 'const_timestep' in trainer.forward_kwargs and trainer.forward_kwargs['const_timestep']:
        timestep = None
    else:
        timestep = ModeWrapper.get_item(mode=trainer.dataset_mode, item="timestep", batch=batch)
        timestep = timestep.to(model.device, non_blocking=True)

    edge_index = ModeWrapper.get_item(mode=trainer.dataset_mode, item="edge_index", batch=batch)
    edge_index = edge_index.to(model.device, non_blocking=True)
    batch_idx = ctx["batch_idx"].to(model.device, non_blocking=True)

    # inputs are the velocities of all timesteps
    x = einops.rearrange(
        x,
        "a num_input_timesteps dim -> a (num_input_timesteps dim)",
    )

    unbatch_idx = ctx["unbatch_idx"].to(model.device, non_blocking=True)
    unbatch_select = ctx["unbatch_select"].to(model.device, non_blocking=True)

    print(f"Batch ready")

    with trainer.autocast_context:
        vel_pred = model.rollout_large_t(
            x=x,
            all_pos=all_pos,
            timestep=timestep,
            edge_index=edge_index,
            batch_idx=batch_idx,
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select
        )




