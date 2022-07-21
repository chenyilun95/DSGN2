import torch
import numpy as np

def project_pseudo_lidar_to_rectcam(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([-ys, -zs, xs], dim=-1)

def project_rectcam_to_pseudo_lidar(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([zs, -xs, -ys], dim=-1)

def project_rect_to_image(pts_3d_rect, P, pose_transform=None):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n, 1), device=pts_3d_rect.device)
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    
    if pose_transform is not None:
        pts_3d_rect = torch.matmul(pts_3d_rect, pose_transform.T)

    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def unproject_image_to_rect(pts_image, P):
    pts_3d = torch.cat([pts_image[..., :2], torch.ones_like(pts_image[..., 2:3])], -1)
    pts_3d = pts_3d * pts_image[..., 2:3]
    pts_3d = torch.cat([pts_3d, torch.ones_like(pts_3d[..., 2:3])], -1)
    P4x4 = torch.eye(4, dtype=P.dtype, device=P.device)
    P4x4[:3, :] = P
    invP = torch.inverse(P4x4)
    pts_3d = torch.matmul(pts_3d, torch.transpose(invP, 0, 1))
    return pts_3d[..., :3]

def unproject_image_to_pseudo_lidar(pts_image, P):
    pts_3d = torch.cat([pts_image[..., :2], torch.ones_like(pts_image[..., 2:3])], -1)
    pts_3d = pts_3d * pts_image[..., 2:3]
    pts_3d = torch.cat([pts_3d, torch.ones_like(pts_3d[..., 2:3])], -1)
    P4x4 = torch.eye(4, dtype=P.dtype, device=P.device)
    P4x4[:3, :] = P
    invP = torch.inverse(P4x4)
    pts_3d = torch.matmul(pts_3d, torch.transpose(invP, 0, 1))
    pts_3d = pts_3d[..., [2, 0, 1]] * torch.as_tensor([1., -1., -1.], device='cuda')
    return pts_3d
