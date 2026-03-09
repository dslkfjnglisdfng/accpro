import os
import sys
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import *
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
    test_contact_joints = ['LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2',
                           'SPINE3', 'LSHOULDER', 'RSHOULDER', 'HEAD',
                           'LELBOW', 'RELBOW', 'LHAND', 'RHAND', 'LFOOT', 'RFOOT'
                           ]
    def __init__(self, config):
        # 初始化模型
        self.model = pin.buildModelFromUrdf(config.paths.physics_model_file)
        self.data = self.model.createData()
        self.dt = 1 / 60.0

        # 状态记忆 [78]
        self.q_dot_prev = np.zeros(self.model.nv)
        self.q_ddot_prev = np.zeros(self.model.nv)

        # 传感器配置
        self.sensor_map = [
            {'name': 'root', 'idx': 5, 'link': 'root_link', 'w_acc': 10.0, 'w_gyro': 1.0},
            {'name': 'head', 'idx': 4, 'link': 'head_link', 'w_acc': 1.0, 'w_gyro': 1.0},
            {'name': 'l_arm', 'idx': 0, 'link': 'left_hand_link', 'w_acc': 0.5, 'w_gyro': 0.5},
            {'name': 'r_arm', 'idx': 1, 'link': 'right_hand_link', 'w_acc': 0.5, 'w_gyro': 0.5},
            {'name': 'l_shank', 'idx': 2, 'link': 'left_shin_link', 'w_acc': 1.0, 'w_gyro': 1.0},
            {'name': 'r_shank', 'idx': 3, 'link': 'right_shin_link', 'w_acc': 1.0, 'w_gyro': 1.0},
        ]

        # 脚部定义
        self.feet_links = [
            {'side': 'left', 'link': 'left_foot_link'},
            {'side': 'right', 'link': 'right_foot_link'}
        ]

    def verify_urdf_with_smpl(self):
        urdf_path = config.paths.physics_model_file
        # 1. 加载模型
        if not os.path.exists(urdf_path):
            print(f"错误: 找不到文件 {urdf_path}")
            return

        # 加载模型和数据
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        print(f"模型加载成功: {model.name}")
        print(f"关节数量: {model.njoints}")
        print("-" * 30)

        # 2. 设置为零姿态 (Zero Pose)
        # 对于 revolute 关节是 0 rad，对于 floating 关节是 [0,0,0, 0,0,0,1]
        q = pin.neutral(model)

        # 3. 前向运动学计算
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # 4. 提取关键关节的全局位置 (Global Position)
        # 我们关注带有 'limb' 或者序列最后的关节，因为它们代表了物理关节点
        target_joints = [
            "root",
            "left_hip_rz",
            "right_hip_rz",
            "left_knee_rx",
            "right_knee_rx",
            "left_ankle_rx",
            "right_ankle_rx"
        ]

        print(f"{'关节名称':<20} | {'X':>8} {'Y':>8} {'Z':>8}")
        print("-" * 50)

        for name in target_joints:
            try:
                # 获取关节 ID
                joint_id = model.getJointId(name)
                # 获取该关节相对于世界坐标系的变换矩阵
                placement = data.oMi[joint_id]
                pos = placement.translation
                print(f"{name:<20} | {pos[0]:8.4f} {pos[1]:8.4f} {pos[2]:8.4f}")
            except:
                print(f"警告: 关节 '{name}' 未在 URDF 中找到或无法提取。")

        # 5. 检查 T-Pose 是否对称
        # 获取左腿和右腿的坐标
        l_hip = data.oMi[model.getJointId("left_hip_rz")].translation
        r_hip = data.oMi[model.getJointId("right_hip_rz")].translation

        print("-" * 50)
        print(f"左右髋部中点: {(l_hip + r_hip) / 2}")
        print(f"左右距离 (宽度): {np.linalg.norm(l_hip - r_hip):.4f}m")

    if __name__ == "__main__":
        # 将此路径替换为你保存 URDF 的实际位置
        verify_urdf_with_smpl()
    def get_root_v(self, imu_data, pose):
        """
        输入:
            imu_data: 字典 {'vacc': [B,6,3], 'omega': [B,6,3], ...}
            pose: [B, 78] 关节角 q
        注意：不再需要 contact_labels
        """
        """
        基于最小二乘法，融合IMU数据与运动学约束求解根节点速度。
        适用于模型: World -> Ankle -> Knee -> Hip (Root)
        imu_data: { [左小臂，右小臂，左小腿，右小腿，头，根节点]
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
        q_batch = smpl_to_rbdl_data(pose)
        batch_size = q_batch.shape[0]
        root_velocities = []

        # 获取脚的 Frame ID
        foot_fids = [self.model.getFrameId(f['link']) for f in self.feet_links]

        for i in range(batch_size):
            # --- A. 准备当前帧数据 ---
            q_current = q_batch[i]  # [78,]

            # [关键修改]: 在求解前，先计算一次 FK 来确定脚的高度
            # 我们只需要位置，不需要速度/加速度，所以只传 q
            pin.framesForwardKinematics(self.model, self.data, q_current)

            # 检测接触状态 (阈值 0.3)
            current_contact_flags = []
            for fid in foot_fids:
                # 获取该 Frame 在世界系下的位置
                foot_pos = self.data.oMf[fid].translation
                foot_height = foot_pos[2]  # Z轴高度

                # 如果高度 < 0.3，认为接触
                is_contact = foot_height < 0.03
                current_contact_flags.append(is_contact)

            # --- B. 初始猜测 (Warm Start) ---
            q_dot_guess = self.q_dot_prev + self.q_ddot_prev * self.dt
            q_ddot_guess = self.q_ddot_prev
            x0 = np.hstack([q_dot_guess, q_ddot_guess])

            # --- C. 定义 Cost Function ---
            def cost_function(x):
                nq = self.model.nv
                v_curr = x[:nq]
                a_curr = x[nq:]

                # 更新 Pinocchio 动力学状态 (v, a)
                pin.forwardKinematics(self.model, self.data, q_current, v_curr, a_curr)
                pin.updateFramePlacements(self.model, self.data)

                residuals = []

                # 1. 积分约束 (强约束)
                res_int = (v_curr - (self.q_dot_prev + 0.5 * (self.q_ddot_prev + a_curr) * self.dt)) * 10.0
                residuals.extend(res_int)

                # 2. IMU 观测约束
                for sensor in self.sensor_map:
                    idx = sensor['idx']
                    obs_w = imu_data['omega'][i, idx].cpu().numpy()
                    obs_a_lin = imu_data['vacc'][i, idx].cpu().numpy()

                    fid = self.model.getFrameId(sensor['link'])
                    v_frame = pin.getFrameVelocity(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
                    a_frame = pin.getFrameAcceleration(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)

                    residuals.extend((v_frame.angular - obs_w) * sensor['w_gyro'])
                    residuals.extend((a_frame.linear - obs_a_lin) * sensor['w_acc'])

                # 3. [关键修改] 基于高度的接触约束
                # 遍历两只脚
                for k, fid in enumerate(foot_fids):
                    if current_contact_flags[k]:  # 如果刚才检测到高度 < 0.3
                        # 获取脚的速度和加速度
                        v_foot = pin.getFrameVelocity(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
                        a_foot = pin.getFrameAcceleration(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)

                        # 强约束：钉死在地面
                        residuals.extend(v_foot.linear * 100.0)
                        residuals.extend(v_foot.angular * 10.0)
                        residuals.extend(a_foot.linear * 100.0)

                return np.array(residuals).flatten()

            # --- D. 求解 ---
            res = least_squares(cost_function, x0, method='lm', max_nfev=20)

            # --- E. 更新 ---
            x_opt = res.x
            self.q_dot_prev = x_opt[:self.model.nv]
            self.q_ddot_prev = x_opt[self.model.nv:]

            # 计算最终 Root 速度
            pin.forwardKinematics(self.model, self.data, q_current, self.q_dot_prev, self.q_ddot_prev)
            v_root_final = pin.getFrameVelocity(self.model, self.data,
                                                self.model.getFrameId("root_link"),
                                                pin.LOCAL_WORLD_ALIGNED)

            root_velocities.append(v_root_final.linear)

        return np.array(root_velocities)

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
