import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import articulate as art

from config import paths, joint_set, vel_scale
from utils import normalize_and_concat
from net import PIP
from dynamics import PhysicsOptimizer as PhysicsOptimizerRBDL
from dynamics_pino import PhysicsOptimizer as PhysicsOptimizerPino


def geodesic_angle_deg(R1, R2):
    """
    计算两个旋转矩阵之间的测地线角度（度）。
    R1, R2: [3, 3] numpy array
    """
    R_diff = R2 @ R1.T
    trace = np.trace(R_diff)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(theta * 180.0 / np.pi)


@torch.no_grad()
def run_one_sequence(seq_id=0, data_dir=paths.dipimu_dir, output_dir="compare_dynamics_output"):
    """
    对同一个序列，使用 dynamics.PhysicsOptimizer (RBDL)
    和 dynamics_pino.PhysicsOptimizer (Pinocchio) 分别跑一遍，
    对比它们输出的 pose_opt / tran_opt 差异，并画误差曲线。
    """
    device = torch.device("cpu")

    # ---- 1. 读取 DIP-IMU 预处理好的 test.pt ----
    print(f'Loading test data from "{data_dir}"')
    accs, rots, poses, trans = torch.load(os.path.join(data_dir, "test.pt")).values()

    glb_acc = accs[seq_id].to(device)        # [T, 6, 3]
    glb_rot = rots[seq_id].to(device)        # [T, 6, 3, 3]
    pose_gt_axis = poses[seq_id].to(device)  # [T, 24, 3]
    tran_gt = trans[seq_id].to(device)       # [T, 3]

    # ---- 2. 初始化网络 PIP，只用到它生成 (pose, joint_velocity, contact) 这部分 ----
    print("Initializing PIP network and loading weights ...")
    net = PIP().to(device)
    net.eval()

    # init_pose: 取 GT 的第 0 帧做初始化
    init_pose = art.math.axis_angle_to_rotation_matrix(pose_gt_axis[0:1]).view(1, 24, 3, 3)
    init_pose[0, 0] = torch.eye(3, device=device)

    # 计算 leaf 关节的初始 3D 位置（与 PIP.predict 内部保持一致）
    _, joints = net.forward_kinematics(init_pose)
    lj_init = joints[0, joint_set.leaf].view(-1)  # [joint_set.n_leaf * 3]
    jvel_init = torch.zeros(24 * 3, device=device)

    # 组装 RNN 输入
    x = (
        normalize_and_concat(glb_acc, glb_rot),  # [T, 72]
        lj_init,
        jvel_init,
    )

    # 通过 RNN 得到 pose, joint_velocity, contact （仍然在 SMPL 空间）
    leaf_joint, full_joint, global_6d_pose, joint_velocity, contact = [
        _[0] for _ in net.forward([x])
    ]

    # 还原为完整 24 关节的局部旋转矩阵
    pose_pred = net._reduced_glb_6d_to_full_local_mat(
        glb_rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose
    )  # [T, 24, 3, 3]

    # 关节线速度变换到世界坐标系（与 PIP.predict 保持一致）
    joint_velocity_world = (
        joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2)) * vel_scale
    )  # [T, 24, 3]

    # ---- 3. 分别用 RBDL 版和 Pinocchio 版 PhysicsOptimizer 跑一遍 ----
    phys_rbdl = PhysicsOptimizerRBDL(debug=False)
    phys_pino = PhysicsOptimizerPino(debug=False)

    phys_rbdl.reset_states()
    phys_pino.reset_states()

    pose_r_list, tran_r_list = [], []
    pose_p_list, tran_p_list = [], []

    print("Running dynamics with RBDL (dynamics.py) and Pinocchio (dynamics_pino.py) ...")
    for t, (p, v, c, a) in enumerate(
        zip(pose_pred, joint_velocity_world, contact, glb_acc)
    ):
        # 注意：PhysicsOptimizer 期望的是 CPU tensor
        p_cpu = p.cpu()
        v_cpu = v.cpu()
        c_cpu = c.cpu()
        a_cpu = a.cpu()

        pose_r, tran_r = phys_rbdl.optimize_frame(p_cpu, v_cpu, c_cpu, a_cpu)
        pose_p, tran_p = phys_pino.optimize_frame(p_cpu, v_cpu, c_cpu, a_cpu)

        pose_r_list.append(pose_r)
        tran_r_list.append(tran_r)
        pose_p_list.append(pose_p)
        tran_p_list.append(tran_p)

    pose_r = torch.stack(pose_r_list)  # [T, 24, 3, 3]
    tran_r = torch.stack(tran_r_list)  # [T, 3]
    pose_p = torch.stack(pose_p_list)  # [T, 24, 3, 3]
    tran_p = torch.stack(tran_p_list)  # [T, 3]

    # ---- 4. 计算两种物理结果之间的误差（而不是对 GT 的误差）----
    T = tran_r.shape[0]
    trans_err = (tran_p - tran_r).norm(dim=1).numpy()  # [T]

    root_rot_err = []
    for t in range(T):
        R_r = pose_r[t, 0].numpy()
        R_p = pose_p[t, 0].numpy()
        root_rot_err.append(geodesic_angle_deg(R_r, R_p))
    root_rot_err = np.asarray(root_rot_err)  # [T]

    # ---- 5. 画曲线并保存 ----
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 4), dpi=150)
    plt.subplot(1, 2, 1)
    plt.plot(trans_err)
    plt.xlabel("Frame")
    plt.ylabel("||Δ translation|| (m)")
    plt.title("RBDL vs Pinocchio: translation difference")

    plt.subplot(1, 2, 2)
    plt.plot(root_rot_err)
    plt.xlabel("Frame")
    plt.ylabel("Root rotation diff (deg)")
    plt.title("RBDL vs Pinocchio: root rotation difference")

    png_path = os.path.join(output_dir, f"seq_{seq_id:03d}_compare.png")
    plt.tight_layout()
    plt.savefig(png_path)
    print(f'Saved comparison plot to "{png_path}"')

    # 同时把数值保存成 npy，方便后处理
    np.savez(
        os.path.join(output_dir, f"seq_{seq_id:03d}_compare.npz"),
        trans_err=trans_err,
        root_rot_err=root_rot_err,
    )
    print("Saved raw error arrays (trans_err, root_rot_err) to npz.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare dynamics.py (RBDL) and dynamics_pino.py (Pinocchio) on the same DIP-IMU sequence."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=paths.dipimu_dir,
        help="Path to preprocessed DIP-IMU directory (containing test.pt).",
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=0,
        help="Sequence index in test.pt to compare.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="compare_dynamics_output",
        help="Directory to save comparison plots and npz.",
    )
    args = parser.parse_args()

    run_one_sequence(seq_id=args.seq, data_dir=args.data_dir, output_dir=args.out)


if __name__ == "__main__":
    main()


