import os
import sys
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import articulate as art
import config
import pickle
import pinocchio as pin
import time
from pinocchio.visualize import MeshcatVisualizer
import threading
import hppfcl
from articulate.math import *
import utils


class PhysicsToGetRootAcc:
    def __init__(self):
        self.body_model = art.ParametricModel((config.paths.smpl_file))
        self.inverse_kinematics_R = self.body_model.inverse_kinematics_R
        self.forward_kinematics = self.body_model.forward_kinematics
        self.get_se3_from_so3pose = self.body_model.get_se3_from_so3pose
        self.bone_len, self.bone_vec = self.body_model.get_bone_vector()
       ##物理模型初始化
        self.model_right,self.geom_model_right = self.get_math_model_and_geo_model(config.leg_used.RIGHT_LEG)
        self.model_right_data = self.model_right.createData()
        self.model_left,self.geom_model_left = self.get_math_model_and_geo_model(config.leg_used.LEFT_LEG)
        self.model_left_data = self.model_left.createData()
        self.leg_used = None
        self.q = None
        self.qdot = None
        self.qddot = None
        self.dt = 0.01
    def init_model(self):
        self.q = None
        self.qdot = None
        self.qddot = None

    def get_root_v(self, imu_data, pose):
        """
        基于最小二乘法，融合IMU数据与运动学约束求解根节点速度。
        适用于模型: World -> Ankle -> Knee -> Hip (Root)
        mu_data: { [左小臂，右小臂，左小腿，右小腿，头，根节点]
        vrot:[batch,6,3,3]
        vacc:[batch,6,3]
        omega:[batch,6,3]
        alpha:[batch,6,3]
        ｝

        pose: [batch, 24, 3, 3] SMPL Pose
        #vrot omega alpha都是大地坐标系下的。
        return:
        root_velocity: [batch, 3] 根节点(Hip)在世界系下的线速度
        """

        # --- 1. 数据准备与预处理 ---
        # 提取关节角 q [batch, 9] (假设Pose已转换为对应模型的9个关节角)
        current_q_full_batch = self._pose_to_pinocchio_data(pose)
        self.model = self.model_right
        self.data = self.model.createData()
        # 提取IMU数据 (注意：需确认imu_data中对应部位的Key)
        # 假设 imu_data['vacc'] 形状为 [batch, 6, 3] (包含重力)
        # 假设顺序: 0:L_Arm, 1:R_Arm, 2:L_Shank, 3:R_Shank, 4:Head, 5:Root
        # 这里的索引需要根据你实际的 imu_data 结构修改，下面以 "Root" 和 "R_Shank"(假设是右腿链) 为例

        batch_size = current_q_full_batch.shape[0]
        dt = 1/60  # 假设时间步长，需根据实际情况调整

        # 结果容器
        root_velocities = []

        # 定义Pinocchio模型相关的Frame ID (只需查找一次)
        # 假设模型构建顺序如你所述，需找到对应IMU安装位置的Frame ID
        # 这里假设: 'ankle_z'后是小腿, 'knee_z'后是大腿, 'hip_z'后是骨盆
        # 注意：实际上需要找到 Joint 对应的 Body Frame ID
        fid_shank = self.model.getFrameId("shank")
        fid_root = self.model.getFrameId("root")

        for i in range(batch_size):
            # --- A. 初始化当前帧状态 ---
            q_current = current_q_full_batch[i]  # [9,]

            # 初始猜测 (Warm Start): 利用上一帧的预测
            # x0 = [q_dot_guess (9,), q_ddot_guess (9,)]
            if i == 0:
                # 第一帧没有历史，设为0或用IMU粗略估计
                q_dot_prev = np.zeros_like(q_current)
                q_ddot_prev = np.zeros_like(q_current)
            else:
                q_dot_prev = self.qdot
                q_ddot_prev = self.qddot

            # 简单的线性预测
            q_dot_guess = q_dot_prev + q_ddot_prev * dt
            q_ddot_guess = q_ddot_prev  # 假设加速度不变
            x0 = np.hstack([q_dot_guess, q_ddot_guess])

            # --- B. 准备观测目标 (Target Observations) ---
            # 注意：这里需要根据你的 batch 数据结构提取第 i 个样本
            # 假设 imu_data 里的张量已经是 numpy 或者 torch，这里转为 numpy

            # [观测 1]: 根节点 (Hip) IMU
            idx_root = 5
            obs_root_w = imu_data['omega'][i, idx_root]
            obs_root_a = imu_data['alpha'][i, idx_root]
            obs_root_lin_acc = imu_data['vacc'][i, idx_root]

            # [观测 2]: 小腿 (Shank) IMU
            idx_shank = 3  # 假设是右小腿
            obs_shank_w = imu_data['omega'][i, idx_shank]
            obs_shank_a = imu_data['alpha'][i, idx_shank]
            obs_shank_lin_acc = imu_data['vacc'][i, idx_shank]

            # [观测 3]: 大腿 (Thigh) IMU
            # 如果 imu_data 里没有大腿数据，可以注释掉相关约束，或者用膝盖关节约束代替
            # 这里假设存在大腿数据，或者仅仅依靠运动链约束

            # --- C. 构建残差函数 (Cost Function) ---
            def  cost_function(x):
                """
                x: [q_dot (9), q_ddot (9)]
                计算 IMU 观测值与 Pinocchio 正向动力学计算值的误差
                """
                # 1. 拆包变量
                nq = self.model.nq
                v_curr = x[:nq]  # q_dot
                a_curr = x[nq:]  # q_ddot

                # 2. 积分约束 (Priors)
                # 约束 q_dot 和 q_ddot 必须符合时间积分关系，防止突变
                # q_dot_current 应该接近 q_dot_prev + a * dt
                # 这里的权重 weight_int 应该设得比较大
                res_int_vel = (v_curr - (q_dot_prev + 0.5 * (q_ddot_prev + a_curr) * dt)) * 10.0

                # 3. Pinocchio 正向运动学 (Forward Kinematics)
                # 这一步极其关键：它同时计算了 v, a 以及 Coriolis/Centrifugal 项
                pin.forwardKinematics(self.model, self.data, q_current, v_curr, a_curr)
                pin.updateFramePlacements(self.model, self.data)
                # 必须更新 Frame Placement 才能获取全局坐标系下的速度/加速度
                # 注意：Pinocchio 的计算结果通常在 Local 坐标系，需转到 World
                # 或者使用 LOCAL_WORLD_ALIGNED

                residuals = []
                residuals.extend(res_int_vel)

                # --- 约束 1: Root (Hip) ---
                # 获取 Hip 在 World 系下的速度和加速度
                # reference=pin.LOCAL_WORLD_ALIGNED 意味着：方向与世界系对齐，原点在物体上
                v_spatial_root = pin.getFrameVelocity(self.model, self.data, fid_root, pin.LOCAL_WORLD_ALIGNED)
                a_spatial_root = pin.getFrameAcceleration(self.model, self.data, fid_root, pin.LOCAL_WORLD_ALIGNED)

                # 角速度误差
                res_root_w = (v_spatial_root.angular - obs_root_w) * 1.0
                # 角加速度误差
                res_root_alpha = (a_spatial_root.angular - obs_root_a) * 0.1
                # 线加速度误差
                res_root_acc = (a_spatial_root.linear - obs_root_lin_acc) * 1.0

                residuals.extend(res_root_w)
                residuals.extend(res_root_alpha)
                residuals.extend(res_root_acc)

                # --- 约束 2: Shank (小腿) ---
                v_spatial_shank = pin.getFrameVelocity(self.model, self.data, fid_shank, pin.LOCAL_WORLD_ALIGNED)
                a_spatial_shank = pin.getFrameAcceleration(self.model, self.data, fid_shank, pin.LOCAL_WORLD_ALIGNED)

                residuals.extend((v_spatial_shank.angular - obs_shank_w) * 1.0)
                residuals.extend((a_spatial_shank.angular - obs_shank_a) * 0.1)
                residuals.extend((a_spatial_shank.linear - obs_shank_lin_acc) * 1.0)

                return np.array(residuals).flatten()

            # --- D. 执行优化 ---
            # 使用 Levenberg-Marquardt 算法求解
            res = least_squares(cost_function, x0, verbose=0, method='lm')

            # --- E. 更新状态与提取结果 ---
            x_opt = res.x
            self.qdot = x_opt[:self.model.nq]
            self.qddot = x_opt[self.model.nq:]

            # 再次执行一次 FK 以确保 self.data 里是最新最优的数据
            pin.forwardKinematics(self.model, self.data, q_current, self.qdot, self.qddot)

            # 提取 Root (Hip) 的线速度
            # 同样使用 LOCAL_WORLD_ALIGNED，确保输出是在大地坐标系下的 [vx, vy, vz]
            v_root_final = pin.getFrameVelocity(self.model, self.data, fid_root, pin.LOCAL_WORLD_ALIGNED)

            root_velocities.append(v_root_final.linear)

        return np.array(root_velocities)
    def _parse_imu_vector(self, data_slice, type_name):
        # 辅助函数：处理你的 imu_data 格式
        # 假设 'gyro' 在 12:21 (3x3矩阵)，取对角线或特定行？
        # 如果数据是 flatten 的 [acc(3), rot(9), gyro(3x3), ang_acc(3x3)]
        # 通常 Gyro 是向量，如果是矩阵可能是 skew matrix
        if type_name == 'gyro':
            # 假设是向量放在前3位，或者需要从矩阵提取
            return data_slice[12:15]
        elif type_name == 'ang_acc':
            return data_slice[21:24]
        return np.zeros(3)

    def vir_theSMPLpose(self,SMPLpose_data_seq:torch.Tensor):
        """
        输入的是SMPLpose_data [batch_size,24,3,3]
        """
        model = self.model_right
        print("=" * 30)
        print(f"模型名称: {model.name}")
        print(f"配置向量维度 (model.nq): {model.nq}")  # 固定基座应为 7，浮动基座应为 14
        print(f"速度向量维度 (model.nv): {model.nv}")  # 固定基座应为 7，浮动基座应为 13
        print(f"关节数量 (model.njoints): {model.njoints}")  # 包括 Universe 关节
        print("=" * 30)
        viz = MeshcatVisualizer(model,self.geom_model_right, self.geom_model_right)
        viz.initViewer(open=False)
        viz.loadViewerModel()
        self.viz = viz

        print("✅ MeshCat 服务已启动，请访问: http://127.0.0.1:7000/static/")

        dt = 1.0 / 30

        try:
            for SMPLpose_data in SMPLpose_data_seq:
                SMPLpose_data = art.math.axis_angle_to_rotation_matrix(SMPLpose_data).view(-1,24,3,3)
                for t in range(SMPLpose_data.shape[0]):
                    # 1. 取一帧 SMPL pose
                    smpl_pose = SMPLpose_data[t:t + 1]  # [1,24,3,3]

                    # 2. 数据对齐，把24*3*3的姿态数据转换成腿部的7*1的q
                    q = self._pose_to_pinocchio_data(smpl_pose)  # [1,7]
                    q = q[0]  # [7]
                    # 3. 显示
                    viz.display(q)
                    time.sleep(dt)
        except KeyboardInterrupt:
            print("🛑 停止可视化。")

    def verify_accuracy(self, pose_data: np.ndarray):
        """
        验证 SMPL 和 Pinocchio 模型的 [全局朝向] 是否对齐。
        """

        def log(msg):
            print(msg)
            sys.stdout.flush()

        # 设置打印精度，保留2位小数，不显示科学计数法
        np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

        # ---------------------------------------------------------
        # 1. 准备 SMPL 真值 (Rotations)
        # ---------------------------------------------------------
        if isinstance(pose_data, np.ndarray):
            pose_tensor = torch.from_numpy(pose_data).float()
        else:
            pose_tensor = pose_data.float()

        # [关键改变] 这里我们需要 global_rot (旋转矩阵 [B, 24, 3, 3])
        with torch.no_grad():
            global_rot_smpl, _ = self.body_model.forward_kinematics(pose_tensor, calc_mesh=False)

        # 转为 Numpy
        global_rot_smpl = global_rot_smpl.cpu().numpy()

        # ---------------------------------------------------------
        # 2. 准备 Pinocchio 模型
        # ---------------------------------------------------------
        model = self.model_right
        data = model.createData()  # ⚡️ 创建新 data 避免污染

        # ---------------------------------------------------------
        # 3. 提取第 0 帧进行核对
        # ---------------------------------------------------------
        t = 0

        # 定义我们要对比的关节对
        # 格式: (Pinocchio关节名, SMPL关节索引, 显示名称)
        # SMPL Right Leg 索引: 8=Ankle, 5=Knee, 2=Hip
        check_list = [
            ("ankle_z", 5, "小腿朝向"),
            ("knee_z", 2, "大腿朝向"),
            ("hip_z", 0, "髋关节朝向")
        ]

        try:
            # === A. 运行 Pinocchio FK ===
            smpl_pose_frame = pose_tensor[t:t + 1]

            # 核心转换：SMPL Pose -> Pinocchio q
            q_full = self._pose_to_pinocchio_data(smpl_pose_frame)[0]

            # 必须先 Update Geometry 才能拿 oMi
            pin.forwardKinematics(model, data, q_full)
            # 旋转主要看 Joint Placement，不需要 updateFramePlacements (除非你看的是末端 Frame)
            # === B. 打印表头 ===
            log("=" * 85)
            log(f"{'关节名称':<12} | {'Pinocchio RPY (deg)':<22} | {'SMPL RPY (deg)':<22} | {'误差(deg)':<10}")
            log(f"{'':<12} | {'(Roll  Pitch  Yaw)':<22} | {'(Roll  Pitch  Yaw)':<22} | {'(Geodesic)'}")
            log("-" * 85)

            total_error_deg = 0.0
            verification_record = {}

            for pin_name, smpl_idx, label in check_list:
                # --- 1. 获取 Pinocchio 旋转 ---
                if not model.existJointName(pin_name):
                    log(f"⚠️ 警告: 模型中找不到关节 {pin_name}，跳过。")
                    continue

                id_pin = model.getJointId(pin_name)
                R_pin = data.oMi[id_pin].rotation  # [3, 3]

                # --- 2. 获取 SMPL 旋转 ---
                R_smpl = global_rot_smpl[t, smpl_idx]  # [3, 3]

                # --- 3. 严谨误差计算 (Geodesic Error) ---
                # 计算 R_diff = R_smpl.T * R_pin
                # 如果完全重合，R_diff 应该是单位矩阵 Identity
                R_diff = R_smpl.T @ R_pin

                # 使用 log3 算出轴角，其模长就是差异角度(弧度)
                diff_vec =rotation_matrix_to_euler_angle_np(R_diff).squeeze()
                error_deg = np.linalg.norm(diff_vec) * (180 / np.pi)
                total_error_deg += error_deg

                # --- 4. 直观展示 (转欧拉角显示) ---
                # Pinocchio matrixToRpy 返回的是弧度 [Roll, Pitch, Yaw]
                rpy_pin_deg = np.degrees(pin.rpy.matrixToRpy(R_pin))
                rpy_smpl_deg = np.degrees(pin.rpy.matrixToRpy(R_smpl))

                # --- 5. 打印行 ---
                # 格式化数组为字符串
                str_pin = f"{rpy_pin_deg[0]:6.1f} {rpy_pin_deg[1]:6.1f} {rpy_pin_deg[2]:6.1f}"
                str_smpl = f"{rpy_smpl_deg[0]:6.1f} {rpy_smpl_deg[1]:6.1f} {rpy_smpl_deg[2]:6.1f}"

                # 标记高误差
                status = "✅" if error_deg < 5.0 else f"❌ {error_deg:.1f}°"

                log(f"{label:<12} | {str_pin:<22} | {str_smpl:<22} | {status}")

                # 记录数据
                verification_record[f"{label}_R_pin"] = R_pin
                verification_record[f"{label}_R_smpl"] = R_smpl

            log("-" * 85)

            # === C. 总结 ===
            avg_error = total_error_deg / len(check_list)
            if avg_error < 5.0:
                log(f"✅ [通过] 平均朝向误差: {avg_error:.2f}° (容差 < 5°)")
            else:
                log(f"❌ [失败] 平均朝向误差: {avg_error:.2f}°。请检查坐标系定义(XYZ顺序)或关节轴向。")

            return verification_record

        except Exception as e:
            log(f"🛑 验证过程出错: {e}")
            import traceback
            traceback.print_exc()
    def start_viz(self):
        threading.Thread(target=self.vir, daemon=True).start()
    def get_math_model_and_geo_model(self):
        """
        在这个函数中我们要初始化模型和可视化模型
        param
            model:物理模型的计算model
            T_motion_se3: 经过求逆矩阵之后的从踝关节到膝关节到髋关节的motion
        return
            model, geom_model:经过初始化之后的计算模型和可视化模型
        """
        Leg_used = self.leg_used
        model = pin.Model()
        bone_len, bone_vec = self.body_model.get_bone_vector()
        bone_vec = bone_vec.detach().cpu().numpy()
        if(Leg_used == config.leg_used.RIGHT_LEG):
            ankle2knee,knee2hip,hip2root = -bone_vec[config.leg_joints.right_leg]
        if(Leg_used == config.leg_used.LEFT_LEG):
            ankle2knee,knee2hip,hip2root = -bone_vec[config.leg_joints.left_leg]
        # TODO:暂时不考虑力和质量对连杆机构的影响        # 父关节为根（0）
        ankle_x = model.addJoint(0, pin.JointModelRX(), pin.SE3.Identity(), "ankle_x")
        ankle_y = model.addJoint(ankle_x, pin.JointModelRY(), pin.SE3.Identity(), "ankle_y")
        ankle_z = model.addJoint(ankle_y, pin.JointModelRZ(), pin.SE3.Identity(), "ankle_z")
        root_frame_id = model.addFrame(
            pin.Frame("shank",ankle_z , 0, pin.SE3(np.eye(3), ankle2knee), pin.FrameType.BODY)
        )
        knee_x = model.addJoint(ankle_z, pin.JointModelRX(),pin.SE3(np.eye(3),ankle2knee),"knee_x")
        knee_y = model.addJoint(knee_x, pin.JointModelRY(),pin.SE3.Identity(),"knee_y")
        knee_z = model.addJoint(knee_y, pin.JointModelRZ(),pin.SE3.Identity(),"knee_z")
        hip_id_x = model.addJoint(knee_z, pin.JointModelRX(),
                                   pin.SE3(np.eye(3),knee2hip), "hip_x")
        hip_id_y = model.addJoint(hip_id_x, pin.JointModelRY(),
                                   pin.SE3.Identity(), "hip_y")
        hip_id_z = model.addJoint(hip_id_y, pin.JointModelRZ(),
                                   pin.SE3.Identity(), "hip_z")
        root_frame_id = model.addFrame(
            pin.Frame("root", hip_id_z, 0, pin.SE3(np.eye(3), hip2root), pin.FrameType.BODY)
        )
        geom_model = pin.GeometryModel()

        def rot_from_z_to_vec(v):
            """构造 R，使 R*[0,0,1] = v/||v||"""
            v = np.asarray(v, dtype=np.float64)
            n = np.linalg.norm(v)
            if n < 1e-12:
                return np.eye(3)
            a = np.array([0., 0., 1.])
            b = v / n
            c = np.dot(a, b)
            if c > 1 - 1e-12:
                return np.eye(3)
            if c < -1 + 1e-12:
                return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
            axis = np.cross(a, b)
            s = np.linalg.norm(axis)
            axis /= s
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            return np.eye(3) + K * s + (K @ K) * (1 - c)

        # ================= ankle joint =================
        geom_model.addGeometryObject(pin.GeometryObject(
            name="ankle_joint",
            parent_joint=ankle_x,
            collision_geometry=hppfcl.Sphere(0.008),
            placement=pin.SE3.Identity()
        ))

        # ================= ankle -> knee =================
        t_ak = ankle2knee
        R_ak = rot_from_z_to_vec(t_ak)
        L_ak = float(np.linalg.norm(t_ak))
        geom_model.addGeometryObject(pin.GeometryObject(
            name="shank",
            parent_joint=ankle_z,
            collision_geometry=hppfcl.Cylinder(0.005, L_ak),
            placement=pin.SE3(R_ak, 0.5 * t_ak)
        ))

        geom_model.addGeometryObject(pin.GeometryObject(
            name="knee_joint",
            parent_joint=knee_z,
            collision_geometry=hppfcl.Sphere(0.008),
            placement=pin.SE3.Identity()
        ))

        # ================= knee -> hip =================
        t_kh = knee2hip
        R_kh = rot_from_z_to_vec(t_kh)
        L_kh = float(np.linalg.norm(t_kh))

        geom_model.addGeometryObject(pin.GeometryObject(
            name="thigh",
            parent_joint=knee_z,
            collision_geometry=hppfcl.Cylinder(0.005, L_kh),
            placement=pin.SE3(R_kh, 0.5 * t_kh)
        ))

        geom_model.addGeometryObject(pin.GeometryObject(
            name="hip_joint",
            parent_joint=hip_id_x,
            collision_geometry=hppfcl.Sphere(0.008),
            placement=pin.SE3.Identity()
        ))

        # ================= hip -> pelvis =================
        t_hp = hip2root
        R_hp = rot_from_z_to_vec(t_hp)
        L_hp = float(np.linalg.norm(t_hp))

        geom_model.addGeometryObject(pin.GeometryObject(
            name="pelvis_link",
            parent_joint=hip_id_z,
            collision_geometry=hppfcl.Cylinder(0.005, L_hp),
            placement=pin.SE3(R_hp, 0.5 * t_hp)
        ))

        geom_model.addGeometryObject(pin.GeometryObject(
            name="pelvis_joint",
            parent_joint=hip_id_z,
            collision_geometry=hppfcl.Sphere(0.008),
            placement=pin.SE3(R_hp, t_hp)
        ))
        return model, geom_model

    def _pose_to_pinocchio_data(self, SMPLpose_data: torch.Tensor) -> np.float64:
        """
        SMPLpose_data: [B, 24, 3, 3]
        return: q: [B, 9] = [ankle(3), knee(3), hip(3)]
        """
        leg_used = self.leg_used
        B = SMPLpose_data.shape[0]
        if leg_used == config.leg_used.RIGHT_LEG:
            idx_ankle_smpl, idx_knee_smpl, idx_hip_smpl = config.leg_joints.right_leg
        else:  # LEFT_LEG
            idx_ankle_smpl, idx_knee_smpl, idx_hip_smpl = config.leg_joints.left_leg
        #取一下pose全局的朝向
        with torch.no_grad():
            global_pose, _ = self.body_model.forward_kinematics(
                SMPLpose_data, calc_mesh=False
            )

        R_ankle_global_np = global_pose[:, idx_knee_smpl].cpu().numpy().astype(np.float64)
        R_knee_local_np = SMPLpose_data[:, idx_knee_smpl].cpu().numpy().astype(np.float64)
        R_hip_local_np = SMPLpose_data[:, idx_hip_smpl].cpu().numpy().astype(np.float64)
        q_out = np.zeros((B, 9), dtype=np.float64)
        try:
            for i in range(B):
                q_out[i, 0:3] = rotation_matrix_to_euler_angle_np(R_ankle_global_np[i]).squeeze()
                q_out[i, 3:6] =  rotation_matrix_to_euler_angle_np(R_knee_local_np[i].T).squeeze()
                q_out[i, 6:9] = rotation_matrix_to_euler_angle_np(R_hip_local_np[i].T).squeeze()
        except Exception as e:
            print(f"计算出错: {e}")
        return q_out

    def _get_leg_rotation_inverse(self,pose_tensor:torch.Tensor,leg_used = config.leg_used.RIGHT_LEG)->torch.Tensor:
        """
        在这个函数中我们将smpl中某条腿的朝向提出取来之后取逆矩阵来获取ankle->knee->hip的相对旋转。
        param:  SMPL_data shape[batch_size,24,3,3],torch.tensor
        return: leg_rotation_inverse [batch_size,3(joints of leg),3,3]
        """
        if pose_tensor.dim() == 3:
            pose_tensor = pose_tensor.unsqueeze(0)
        R_local_pelvis = pose_tensor[:, 0]  # [B, 3, 3]
        # 左腿局部
        R_local_L_Hip = pose_tensor[:, 1]
        R_local_L_Knee = pose_tensor[:, 4]
        R_local_L_Ankle = pose_tensor[:, 7]
        # 右腿局部
        R_local_R_Hip = pose_tensor[:, 2]
        R_local_R_Knee = pose_tensor[:, 5]
        R_local_R_Ankle = pose_tensor[:, 8]

        # 2. 链式法则计算全局旋转 (Global Rotations)
        # --- 根节点 ---
        R_world_pelvis = R_local_pelvis  # Root 的全局等于局部

        # --- 左腿链  ---
        # Global Hip = Global Parent (Pelvis) @ Local Hip
        R_world_L_Hip = torch.matmul(R_world_pelvis, R_local_L_Hip)

        # Global Knee = Global Parent (Hip) @ Local Knee
        R_world_L_Knee = torch.matmul(R_world_L_Hip, R_local_L_Knee)

        # Global Ankle = Global Parent (Knee) @ Local Ankle
        R_world_L_Ankle = torch.matmul(R_world_L_Knee, R_local_L_Ankle)

        # --- 右腿链  ---
        R_world_R_Hip = torch.matmul(R_world_pelvis, R_local_R_Hip)
        R_world_R_Knee = torch.matmul(R_world_R_Hip, R_local_R_Knee)
        R_world_R_Ankle = torch.matmul(R_world_R_Knee, R_local_R_Ankle)
        if leg_used == config.leg_used.RIGHT_LEG:
            leg_rotations = torch.cat(
                [R_world_R_Knee,
                 torch.matmul(R_world_R_Knee.transpose(-1,-2),R_world_R_Hip),
                 torch.matmul(R_world_R_Hip.transpose(-1,-2),R_world_pelvis)],dim=0).unsqueeze(0)
        elif leg_used == config.leg_used.LEFT_LEG:
            leg_rotations = torch.cat(
                [R_world_L_Knee,
                 torch.matmul(R_world_L_Knee.transpose(-1, -2), R_world_L_Hip),
                 torch.matmul(R_world_L_Hip.transpose(-1, -2), R_world_pelvis)],dim=0).unsqueeze(0)
        return leg_rotations

    def _get_leg_motion_inverse(self,T_motion_se3:np.float64,leg_used = config.leg_used.RIGHT_LEG):
        """
        这个函数是用来从初始的se3的姿态数据中获得腿部的数据，并把他们反向建模。
        具体来说是以脚踝关节为根节点来建模脚踝到根节点的节点编号是[8,5,2]
        param T_motion_se3:人物的姿态数据，形状是[batch_size,num_joints,4,4]
        return T_motion_se3: 经过求逆矩阵之后的从踝关节到膝关节到髋关节的motion
        """
        T_motion_se3 = self._se3_inverse(T_motion_se3)
        if leg_used == config.leg_used.RIGHT_LEG:
            T_motion_se3 = T_motion_se3[:,config.leg_joints.right_leg]
        if leg_used == config.leg_used.LEFT_LEG:
            T_motion_se3 = T_motion_se3[:,config.leg_joints.left_leg]
        return T_motion_se3
    def _se3_inverse(self,motion_se3:np.float64):
        """
        在这个函数中 求motion的逆矩阵
        param motion_se3: [batch_size,joint_nums,4,4]
        return: motion_se3_inversed : 已经求逆之后的矩阵[batch_size,joint_num,4,4]
        """
        b,j = motion_se3.shape[:2]
        motion_se3 = motion_se3.reshape(-1,4,4)
        motion_se3 = [pin.SE3(T[:3,:3],T[:3,3]).inverse() for T in motion_se3]
        motion_se3 = np.stack(motion_se3,axis=0).reshape(b,j,4,4)
        return motion_se3
    def _get_qdot_from_omega(self, omega):
        """
        从omega=rot_T@rot_dot中获取q_dot
        param:  omega:[batchsize,6,3,3]
        return: q_dot:[  ,6,3]
        """
        w_x = omega[..., 2, 1]
        w_y = omega[..., 0, 2]
        w_z = omega[..., 1, 0]
        q_dot = torch.stack([w_x, w_y, w_z], dim=-1)
        return q_dot
    def _get_qddot_from_alpha(self, alpha):
        """
        从alpha=omega_dot中获取q_ddot
        param:  omega:[batchsize,6,3,3]
        return: q_ddot:[batchsize,6,3]
        """
        return self._get_qdot_from_omega(alpha)

    def _get_dynamic_r_vector(self, model, data, id_start_joint, id_end_joint):
        """
        获取从 start 到 end 的位移向量 r (在世界坐标系下)
        r = Pos_end - Pos_start
        """
        pos_start = data.oMi[id_start_joint].translation
        pos_end = data.oMi[id_end_joint].translation

        # 2. 计算位移向量
        # 注意方向：公式里 r 是从已知点(Root)指向未知点(Hip)
        r = pos_end - pos_start

        return r
    def _calculate_point_acceleration(self,a_root, alpha_root, omega_root, r):
        """
        根据刚体基准点(Root)的运动状态，计算刚体上另一点(Hip)的线加速度。
        公式: a_hip = a_root + (alpha x r) + (omega x (omega x r))

        参数:
            a_root     : (3,) Root 的线加速度
            alpha_root : (3,) Root 的角加速度 (alpha)
            omega_root : (3,) Root 的角速度 (omega)
            r          : (3,) 从 Root 指向 Hip 的位移向量 (r_hip - r_root)
        返回:
            a_hip      : (3,) Hip 的线加速度
        """

        acc_tangential = np.cross(alpha_root, r)
        w_cross_r = np.cross(omega_root, r)
        acc_centripetal = np.cross(omega_root, w_cross_r)

        a_hip = a_root + acc_tangential + acc_centripetal

        return a_hip

if __name__  == "__main__":
    physicsToGetRootAcc = PhysicsToGetRootAcc()
    just_show  =2
    if(just_show == 1):
        physicsToGetRootAcc.start_viz()
        while True:
            time.sleep(1)
    if(just_show == 0):
        pose = torch.load(os.path.join(config.paths.amass_dir,'pose.pt'))
        physicsToGetRootAcc.vir_theSMPLpose(pose)
    if(just_show == 2):
        all_logs = []
        pose = torch.load(os.path.join(config.paths.amass_dir,'pose.pt'))
        for i, p in enumerate(tqdm(pose)):
            # 预处理数据
            p = art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3)

            record = physicsToGetRootAcc.verify_accuracy(p)
            clean_record = {}
            clean_record['frame_id'] = i  # 记录帧号
            for key, value in record.items():
                # 如果是 Tensor，转 Numpy
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()

                # 如果是 Numpy 数组
                if isinstance(value, np.ndarray):
                    # 方案 A: 如果你想把 XYZ 分开存成不同列 (推荐，方便画图)
                    if value.size == 3:
                        clean_record[f"{key}_x"] = value[0]
                        clean_record[f"{key}_y"] = value[1]
                        clean_record[f"{key}_z"] = value[2]
                    # 方案 B: 只是简单转成 list 存入一个单元格
                    else:
                        clean_record[key] = value.tolist()
                else:
                    # 普通数值 (float, string)
                    clean_record[key] = value

            # 添加到总列表
            all_logs.append(clean_record)
            print("💾 正在保存数据...")
        df = pd.DataFrame(all_logs)

        # 保存为 CSV
        save_path = "verification_results.csv"
        df.to_csv(save_path, index=False)

        print(f"✅ 数据已保存至: {os.path.abspath(save_path)}")
# @Time     :2025/11/11 下午5:52
