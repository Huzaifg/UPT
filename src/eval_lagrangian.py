from utils.kappaconfig.util import get_stage_hp
from datasets import dataset_from_kwargs
from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider
from configs.cli_args import parse_run_cli_args
from trainers import trainer_from_kwargs
from kappadata.wrappers import ModeWrapper
from providers.path_provider import PathProvider
from initializers.previous_run_initializer import PreviousRunInitializer
import kappaprofiler as kp
import logging
import einops
import torch

from models import model_from_kwargs
from utils.data_container import DataContainer

from utils.version_check import check_versions
import time

# Viz stuff
from pyevtk.hl import pointsToVTK
from matplotlib import animation
import matplotlib.pyplot as plt
import os
import numpy as np


def convert_batch_to_vtk(batch: torch.Tensor, filename: str) -> None:
    """
    Convert a batch of positions and velocity to a VTK file for a single timestep
    Args:
        batch: Tensor of shape (n_particles, 6)
        filename: Name of the VTK file
    """
    # Extract the positions
    particles = batch.cpu().detach().numpy()
    # Create the grid
    x = np.ascontiguousarray(particles[:, 0])
    y = np.ascontiguousarray(particles[:, 1])
    z = np.ascontiguousarray(particles[:, 2])
    # Separate the velocity components
    vel_x = np.ascontiguousarray(particles[:, 3])
    vel_y = np.ascontiguousarray(particles[:, 4])
    vel_z = np.ascontiguousarray(particles[:, 5])
    # Organize data for pointsToVTK
    data = {
        "Velocity_x": vel_x,
        "Velocity_y": vel_y,
        "Velocity_z": vel_z
    }
    # Call pointsToVTK with correctly shaped and contiguous data
    pointsToVTK(filename, x, y, z, data=data)
    print(f"VTK file written to {filename}")

    



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
    wandb_folder = "l6g2r8dz"
    checkpoint = "E50_U3800_S486400"
    initializer = PreviousRunInitializer(stage_id = "1", model_name = "lagrangian_simformer_model.encoder", checkpoint = checkpoint, stage_name = "stage1", path_provider = path_provider)

    initializer.init_weights(model.encoder)

    initializer = PreviousRunInitializer(stage_id = "1", model_name = "lagrangian_simformer_model.decoder", checkpoint = checkpoint, stage_name = "stage1", path_provider = path_provider)

    initializer.init_weights(model.decoder)


    initializer = PreviousRunInitializer(stage_id = "1", model_name = "lagrangian_simformer_model.latent", checkpoint = checkpoint, stage_name = "stage1", path_provider = path_provider)

    initializer.init_weights(model.latent)


    initializer = PreviousRunInitializer(stage_id = "1", model_name = "lagrangian_simformer_model.conditioner_encoder", checkpoint = checkpoint, stage_name = "stage1", path_provider = path_provider)

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
            batch_size=1,
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

    start_time = time.time()

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

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Since these are the ones that are computed in the model
    time_indicies = torch.tensor([10, 11, 20, 21, 30, 31, 40, 41, 50, 51])

    # Directly from offline_lagrangian_large_t_rollout_mesh_loss_callback
    all_vel = einops.rearrange(
            all_vel,
            "bs time n_particles dim -> (bs n_particles) time dim"
        )
    all_vel_target = all_vel[:,time_indicies,:]
    # all_vel_target = all_vel[:,2:,:]

    # Un-normalize the velocities
    vel_pred = trainer.data_container.get_dataset().unnormalize_vel(vel_pred)
    vel_pred = vel_pred

    all_vel_target = trainer.data_container.get_dataset().unnormalize_vel(all_vel_target)
    all_vel_target = all_vel_target
        # Unbatch
    all_vel_target = einops.rearrange(
        all_vel_target,
        "(bs n_particles) time dim -> bs n_particles time dim",
        bs=len(unbatch_select)
    )
    vel_pred = einops.rearrange(
        vel_pred,
        "(bs n_particles) time dim -> bs n_particles time dim",
        bs=len(unbatch_select)
    )

    
    # # Append the first two velocities from the dataset
    diff_norm_mean = (vel_pred - all_vel_target).norm(dim=3).mean(dim=(1,2))
    relative_norm = ((vel_pred - all_vel_target).norm(dim=3) / all_vel_target.norm(dim=3)).mean(dim=(1,2))

    print(relative_norm)


    # ============================================================
    # Write batch to vtk
    # ============================================================
    
    # Ground truth
    all_pos = einops.rearrange(
        all_pos,
        "bs time n_particles dim -> (bs n_particles) time dim"
    )

    # unnormalize all_vel as well
    all_vel = trainer.data_container.get_dataset().unnormalize_vel(all_vel)
    all_vel = all_vel
    
    # Append all vel last 3 columns to all_pos
    batch_ground_truth = torch.cat((all_pos[:,:60,:], all_vel), dim=2)

    # Loop over timestep axis and write to vtk
    base_vtk = "./vtk_files/"
    # Ensure base_vtk exists or create it
    if not os.path.exists(base_vtk):
        os.makedirs(base_vtk)

    for i in range(batch_ground_truth.shape[1]):
        convert_batch_to_vtk(batch_ground_truth[:,i,:], base_vtk + f"ground_truth_{i}")

    
    

    # Velocity prediction
    # Remove all_pos based on time_indicies
    time_indicies = torch.tensor([0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51]) # Add the input data for better viz
    all_pos = all_pos[:,time_indicies,:]



    vel_pred = einops.rearrange(
        vel_pred,
        "bs n_particles time dim -> (bs n_particles) time dim"
    )

    # Append in the start to vel_pred all_vel first 2 vels
    all_vel_0 = all_vel[:,0,:]
    all_vel_1 = all_vel[:,1,:]

    all_vel_0_expanded = all_vel_0.unsqueeze(1)
    all_vel_1_expanded = all_vel_1.unsqueeze(1)

    # Put all_vel_0 and all_vel_1 to make vel_pred from [8000,10,3] to [8000,12,3]
    vel_pred = torch.cat((all_vel_0_expanded, all_vel_1_expanded, vel_pred), dim=1)


    print(all_pos.shape)
    print(vel_pred.shape)

    # Append vel_pred last 3 columns to all_pos
    batch_pred = torch.cat((all_pos, vel_pred), dim=2)

    # Loop over timestep axis and write to vtk
    base_vtk = "./vtk_files/"
    
    for i in range(batch_pred.shape[1]):
        convert_batch_to_vtk(batch_pred[:,i,:], base_vtk + f"pred_{i}")




