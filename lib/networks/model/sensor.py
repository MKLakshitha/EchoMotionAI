import pytorch3d.ops.knn as knn
import torch

from lib.utils.geo_transform import (apply_T_on_points, rotation_6d_to_matrix,
                                     transform_mat)
from lib.utils.registry import Registry

# Sensor Registry
SENSOR = Registry('sensor')


class Sensor():
    def __init__(self, cfg, **kwargs) -> None:
        self.coord = cfg.coord
        pass

    def get_bs_device(self, batch):
        """
        Get batch size and device from the batch data.
        """
        return batch['motion_mask'].shape[0], batch['motion_mask'].device

    def get_voxel_points(self, V, R):
        """
        Generate a 3D voxel grid based on dimensions (V) and radius (R).
        V: voxel_dim
        R: radius
        """
        x = torch.linspace(-R, R, V)
        y = torch.linspace(-R, R, V)
        z = torch.linspace(-R, R, V)
        points = torch.stack(torch.meshgrid(x, y, z), -1).view(-1, 3)
        voxel_size = 2 * R / V
        return points, voxel_size

    def get_voxel(self, voxel_points, scene_points, scene_normals, voxel_size):
        """
        Compute the voxel grid by querying the nearest scene points.
        Returns the signed distance function (SDF) and normal vectors.
        """
        dists_squared, knn_idx, _ = knn.knn_points(voxel_points, scene_points, K=1, return_nn=True)
        nn_vertices = knn.knn_gather(scene_points, knn_idx).squeeze(-2)
        nn_normal = knn.knn_gather(scene_normals, knn_idx).squeeze(-2)
        sign = (torch.sum(nn_normal * (voxel_points - nn_vertices), dim=-1, keepdim=True) > 0) * 2 - 1
        sdf = dists_squared.sqrt() * sign
        sdf[sdf < 0] = 0.
        if type(voxel_size) == torch.Tensor:
            voxel_batch = []
            for b in range(len(sdf)):
                sdf_b = sdf[b]
                sdf_b[sdf_b > (voxel_size[b] / 2)] = (voxel_size[b] / 2)
                sdf[b] = sdf_b
                voxel_batch.append(1 - (sdf_b / (voxel_size[b] / 2)))
            voxel = torch.stack(voxel_batch, dim=0)
        else:
            sdf[sdf > (voxel_size / 2)] = (voxel_size / 2)
            voxel = 1 - (sdf / (voxel_size / 2))
        return voxel, nn_normal


@SENSOR.register()
class EnvSensor(Sensor):
    def __init__(self, cfg, **kwargs) -> None:
        """
        Initializes the environment sensor and creates the voxel grid.
        """
        super().__init__(cfg, **kwargs)
        self.voxel_dim = V = cfg.voxel_dim
        R = cfg.radius
        self.env_points, self.env_voxel_size = self.get_voxel_points(V, R)

    def forward(self, batch):
        """
        Forward pass for the environment sensor.
        Computes the voxel representation and nearest normals for the scene points.
        """
        bs, device = self.get_bs_device(batch)
        if 'obj_mask' not in batch:
            # If no object mask, return zero tensors.
            return torch.zeros((bs, self.voxel_dim**3, 1), device=device), \
                   torch.zeros((bs, self.voxel_dim**3, 3), device=device), \
                   torch.zeros((bs, self.voxel_dim**3, 3), device=device)
        scene_points = batch[f'xyz_{self.coord}']
        env_points = self.env_points.clone().to(device)
        env_points = torch.repeat_interleave(env_points[None], bs, dim=0)
        env_voxel, nn_normal = self.get_voxel(env_points, scene_points, batch['normal'], self.env_voxel_size)
        return env_voxel, env_points, nn_normal


@SENSOR.register()
class TargetSensor(Sensor):
    def __init__(self, cfg, **kwargs) -> None:
        """
        Initializes the target sensor to process object-related features.
        """
        super().__init__(cfg, **kwargs)
        self.voxel_dim = V = cfg.voxel_dim

    def forward(self, batch):
        """
        Forward pass for the target sensor. It computes the voxel grid
        around object points.
        """
        bs, device = self.get_bs_device(batch)
        if 'obj_mask' not in batch:
            # If no object mask, return zero tensors.
            return torch.zeros((bs, self.voxel_dim**3, 1), device=device), \
                   torch.zeros((bs, self.voxel_dim**3, 3), device=device), \
                   torch.zeros((bs, self.voxel_dim**3, 3), device=device)

        scene_points_batch = batch[f'xyz_{self.coord}']
        target_points_batch = []
        voxel_size_batch = []
        for b in range(bs):
            scene_points = scene_points_batch[b]
            obj_mask = batch['obj_mask'][b]
            obj_points = scene_points[obj_mask]

            # Check if obj_points is empty
            if obj_points.size(0) == 0:
                # Handle case where no object points are found
                print(f"No object points found for batch {b}. Returning zero tensors.")
                return torch.zeros((bs, self.voxel_dim**3, 1), device=device), \
                       torch.zeros((bs, self.voxel_dim**3, 3), device=device), \
                       torch.zeros((bs, self.voxel_dim**3, 3), device=device)

            # Compute object center and radius
            c = obj_points.mean(dim=0)
            r = (obj_points.max(dim=0)[0] - obj_points.min(dim=0)[0]).max() * 0.5

            # Generate target points
            target_points, voxel_size = self.get_voxel_points(self.voxel_dim, r)
            voxel_size_batch.append(voxel_size)
            target_points = target_points.to(device) + c
            target_points_batch.append(target_points)

        target_points_batch = torch.stack(target_points_batch, dim=0)
        voxel_size_batch = torch.stack(voxel_size_batch, dim=0)
        target_voxel, nn_normal = self.get_voxel(target_points_batch, scene_points_batch, batch['normal'], voxel_size_batch)
        return target_voxel, target_points_batch, nn_normal


@SENSOR.register()
class TrajSensor(Sensor):
    def __init__(self, cfg, **kwargs) -> None:
        """
        Initializes the trajectory sensor.
        """
        super().__init__(cfg, **kwargs)
        self.voxel_dim = V = cfg.voxel_dim
        R = cfg.radius
        self.traj_points, self.traj_voxel_size = self.get_voxel_points(V, R)

    def forward(self, batch, x_unnorm):
        """
        Forward pass for the trajectory sensor. Computes the voxel representation
        along the trajectory.
        """
        bs, device = self.get_bs_device(batch)
        nframes = x_unnorm.shape[1]
        nvoxels = self.voxel_dim ** 3
        transl = x_unnorm[..., :3]
        orient_6d = x_unnorm[..., 3:9]
        rotate_matrix = rotation_6d_to_matrix(orient_6d)
        T_t2s = transform_mat(rotate_matrix, transl)  # Traj to scene transform

        tpoints = self.traj_points.clone().to(device)
        tpoints = torch.repeat_interleave(tpoints[None], nframes, dim=0)
        traj_points_batch, traj_voxel_batch, nn_normal_batch = [], [], []
        for b in range(bs):
            scene_points = batch[f'xyz_{self.coord}'][b]
            traj_points = apply_T_on_points(tpoints, T_t2s[b]).view(1, -1, 3)
            traj_voxel, nn_normal = self.get_voxel(traj_points, scene_points[None], batch['normal'][[b], ...], self.traj_voxel_size)
            traj_points_batch.append(traj_points.view(nframes, nvoxels, -1))
            traj_voxel_batch.append(traj_voxel.view(nframes, nvoxels, -1))
            nn_normal_batch.append(nn_normal.view(nframes, nvoxels, -1))

        traj_points_batch = torch.stack(traj_points_batch, dim=0)
        traj_voxel_batch = torch.stack(traj_voxel_batch, dim=0)
        nn_normal_batch = torch.stack(nn_normal_batch, dim=0)
        return traj_voxel_batch, traj_points_batch, nn_normal_batch
