__all__ = ['set_pose', 'smpl_to_rbdl', 'rbdl_to_smpl', 'normalize_and_concat', 'print_title', 'Body', 'smpl_to_rbdl_data',    'PinocchioModelAdapter', 'smpl_to_pinocchio_q',
    'pinocchio_q_to_smpl',
    'pinocchio_q_to_legacy_q']


import enum
import torch
import numpy as np
import pybullet as p
from articulate.math import rotation_matrix_to_euler_angle_np, euler_angle_to_rotation_matrix_np, euler_convert_np, \
    normalize_angle
import numpy as np
import pinocchio as pin


def skew(v):
    v = np.asarray(v).reshape(3)
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)


def quat_xyzw_to_wxyz(q_xyzw):
    q_xyzw = np.asarray(q_xyzw).reshape(4)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(q_wxyz):
    q_wxyz = np.asarray(q_wxyz).reshape(4)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def rotation_matrix_to_quat_xyzw(R):
    """
    Input:
        R: [3, 3]
    Output:
        quat in xyzw convention
    """
    R = np.asarray(R).reshape(3, 3)
    q = pin.Quaternion(R)
    q.normalize()
    # Pinocchio python Quaternion coeffs() order is [x, y, z, w]
    return np.array(q.coeffs()).reshape(4)


def quat_xyzw_to_rotation_matrix(q_xyzw):
    """
    Input:
        q_xyzw: [4]
    Output:
        R: [3, 3]
    """
    q_xyzw = np.asarray(q_xyzw).reshape(4)
    # pin.Quaternion(x, y, z, w)
    q = pin.Quaternion(float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3]))
    q.normalize()
    return np.array(q.toRotationMatrix())

def smpl_to_legacy_q(poses, trans, _smpl_to_rbdl):
    """
    Convert SMPL poses + translations to the old 75D q format.

    poses: can reshape to [N, 24, 3, 3]
    trans: can reshape to [N, 3]

    return:
        [N, 75]
    """
    poses = np.array(poses).reshape(-1, 24, 3, 3)
    trans = np.array(trans).reshape(-1, 3)

    # joints 1..23 -> 23*3 = 69
    euler_poses = rotation_matrix_to_euler_angle_np(poses[:, 1:], 'XYZ').reshape(-1, 69)

    # root rotation
    euler_glbrots = rotation_matrix_to_euler_angle_np(poses[:, :1], 'xyz').reshape(-1, 3)
    euler_glbrots = euler_convert_np(euler_glbrots[:, [2, 1, 0]], 'xyz', 'zyx')

    qs = np.concatenate((trans, euler_glbrots, euler_poses[:, _smpl_to_rbdl]), axis=1)
    qs[:, 3:] = normalize_angle(qs[:, 3:])
    return qs


def legacy_q_to_smpl(qs, _rbdl_to_smpl):
    """
    Convert old 75D q format back to SMPL poses + translations.

    qs: [N, 75]

    return:
        poses: [N, 24, 3, 3]
        trans: [N, 3]
    """
    qs = np.asarray(qs).reshape(-1, 75)

    trans = qs[:, :3]
    euler_glbrots = qs[:, 3:6]
    euler_poses = qs[:, 6:][:, _rbdl_to_smpl]

    euler_glbrots = euler_convert_np(euler_glbrots, 'zyx', 'xyz')[:, [2, 1, 0]]
    glbrots = euler_angle_to_rotation_matrix_np(euler_glbrots, 'xyz').reshape(-1, 1, 3, 3)

    poses = euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 23, 3, 3)
    poses = np.concatenate((glbrots, poses), axis=1)

    return poses, trans


def legacy_q_to_pinocchio_q(q_legacy):
    """
    old 75D q -> pinocchio 76D q

    q_legacy:
        [tx ty tz, root_euler(3), joints(69)]

    q_pin:
        [tx ty tz, qx qy qz qw, joints(69)]
    """
    q_legacy = np.asarray(q_legacy).reshape(-1, 75)
    out = []

    for ql in q_legacy:
        trans = ql[:3]
        root_euler_zyx = ql[3:6]
        joints = ql[6:]

        # Keep consistent with the original rbdl conversion logic
        root_euler_xyz = euler_convert_np(root_euler_zyx.reshape(1, 3), 'zyx', 'xyz')[0][[2, 1, 0]]
        R_root = euler_angle_to_rotation_matrix_np(root_euler_xyz.reshape(1, 3), 'xyz')[0]

        quat_xyzw = rotation_matrix_to_quat_xyzw(R_root)
        q_pin = np.concatenate([trans, quat_xyzw, joints], axis=0)
        out.append(q_pin)

    return np.stack(out, axis=0)


def pinocchio_q_to_legacy_q(q_pin):
    """
    pinocchio 76D q -> old 75D q

    q_pin:
        [tx ty tz, qx qy qz qw, joints(69)]

    q_legacy:
        [tx ty tz, root_euler(3), joints(69)]
    """
    q_pin = np.asarray(q_pin).reshape(-1, 76)
    out = []

    for qp in q_pin:
        trans = qp[:3]
        quat_xyzw = qp[3:7]
        joints = qp[7:]

        R_root = quat_xyzw_to_rotation_matrix(quat_xyzw)

        # reshape to match your utils' likely expected input
        root_euler_xyz = rotation_matrix_to_euler_angle_np(R_root.reshape(1, 1, 3, 3), 'xyz').reshape(1, 3)[0]
        root_euler_zyx = euler_convert_np(root_euler_xyz[[2, 1, 0]].reshape(1, 3), 'xyz', 'zyx')[0]

        q_legacy = np.concatenate([trans, root_euler_zyx, joints], axis=0)
        q_legacy[3:] = normalize_angle(q_legacy[3:])
        out.append(q_legacy)

    return np.stack(out, axis=0)


def smpl_to_pinocchio_q(poses, trans):
    """
    SMPL -> legacy q -> pinocchio q
    """
    q_legacy = smpl_to_legacy_q(poses, trans, _smpl_to_rbdl)
    return legacy_q_to_pinocchio_q(q_legacy)


def pinocchio_q_to_smpl(q_pin):
    """
    pinocchio q -> legacy q -> SMPL
    """
    q_legacy = pinocchio_q_to_legacy_q(q_pin)
    return legacy_q_to_smpl(q_legacy, _rbdl_to_smpl)


# ============================================================
# 3) Pinocchio adapter
# ============================================================

class PinocchioModelAdapter:
    """
    A thin wrapper to mimic the subset of the old RBDLModel API
    used by your PhysicsOptimizer.

    This assumes the URDF itself already contains a floating root joint.
    So we use:
        pin.buildModelFromUrdf(urdf_path)

    For your URDF, expected:
        nq = 76
        nv = 75
    """

    def __init__(self, urdf_path, body_map):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.qdot_size = self.model.nv

        self.body_map = body_map
        self.frame_name_to_id = {f.name: i for i, f in enumerate(self.model.frames)}

    def _frame_id(self, body_name):
        urdf_name = self.body_map.get(body_name, body_name)
        if urdf_name not in self.frame_name_to_id:
            raise KeyError(
                f"Cannot find frame '{urdf_name}' for logical body '{body_name}'. "
                f"Available frames sample: {list(self.frame_name_to_id.keys())[:50]}"
            )
        return self.frame_name_to_id[urdf_name]

    def update_kinematics(self, q, v, a):
        q = np.asarray(q).reshape(self.nq)
        v = np.asarray(v).reshape(self.nv)
        a = np.asarray(a).reshape(self.nv)

        pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)

    def calc_body_position(self, q, body_name):
        """
        World position of the frame origin.
        """
        q = np.asarray(q).reshape(self.nq)
        fid = self._frame_id(body_name)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        return np.array(self.data.oMf[fid].translation).copy()

    def calc_base_to_body_coordinates(self, q, body_name, point_world):
        """
        Convert a world point to local frame coordinates.
        """
        q = np.asarray(q).reshape(self.nq)
        point_world = np.asarray(point_world).reshape(3)
        fid = self._frame_id(body_name)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        oMf = self.data.oMf[fid]
        point_local = oMf.inverse().act(point_world)
        return np.array(point_local).copy()

    def calc_point_Jacobian(self, q, body_name, point_local=None):
        """
        Return the 3 x nv Jacobian of a point on a frame, expressed in world-aligned coordinates.
        """
        q = np.asarray(q).reshape(self.nq)
        fid = self._frame_id(body_name)

        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        J6 = pin.getFrameJacobian(
            self.model,
            self.data,
            fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )  # [6, nv]

        Jv = np.array(J6[:3, :]).copy()
        Jw = np.array(J6[3:, :]).copy()

        if point_local is None:
            return Jv

        point_local = np.asarray(point_local).reshape(3)
        oMf = self.data.oMf[fid]
        r_world = np.array(oMf.rotation @ point_local).reshape(3)

        return Jv - skew(r_world) @ Jw

    def calc_point_velocity(self, q, v, body_name, point_local=None):
        """
        World-aligned linear velocity of a frame point.
        """
        q = np.asarray(q).reshape(self.nq)
        v = np.asarray(v).reshape(self.nv)
        fid = self._frame_id(body_name)

        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)

        vel = pin.getFrameVelocity(
            self.model,
            self.data,
            fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        v_linear = np.array(vel.linear).copy()
        omega = np.array(vel.angular).copy()

        if point_local is None:
            return v_linear

        point_local = np.asarray(point_local).reshape(3)
        oMf = self.data.oMf[fid]
        r_world = np.array(oMf.rotation @ point_local).reshape(3)

        return v_linear + np.cross(omega, r_world)

    def calc_point_acceleration(self, q, v, a, body_name, point_local=None):
        """
        World-aligned classical linear acceleration of a frame point.

        This is used to emulate the bias acceleration term expected by your original code.
        """
        q = np.asarray(q).reshape(self.nq)
        v = np.asarray(v).reshape(self.nv)
        a = np.asarray(a).reshape(self.nv)
        fid = self._frame_id(body_name)

        pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.updateFramePlacements(self.model, self.data)

        acc = pin.getFrameClassicalAcceleration(
            self.model,
            self.data,
            fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        vel = pin.getFrameVelocity(
            self.model,
            self.data,
            fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        a_linear = np.array(acc.linear).copy()
        alpha = np.array(acc.angular).copy()
        omega = np.array(vel.angular).copy()

        if point_local is None:
            return a_linear

        point_local = np.asarray(point_local).reshape(3)
        oMf = self.data.oMf[fid]
        r_world = np.array(oMf.rotation @ point_local).reshape(3)

        return (
            a_linear
            + np.cross(alpha, r_world)
            + np.cross(omega, np.cross(omega, r_world))
        )

    def calc_M(self, q):
        """
        Mass matrix M(q), shape [nv, nv]
        """
        q = np.asarray(q).reshape(self.nq)
        M = np.array(pin.crba(self.model, self.data, q))

        # symmetrize
        M = np.triu(M) + np.triu(M, 1).T
        return M

    def calc_h(self, q, v):
        """
        Nonlinear effects h(q, v), shape [nv]
        """
        q = np.asarray(q).reshape(self.nq)
        v = np.asarray(v).reshape(self.nv)
        h = pin.nonLinearEffects(self.model, self.data, q, v)
        return np.array(h).reshape(self.nv).copy()

    def integrate(self, q, v_dt):
        """
        Integrate configuration on the manifold.
        q: [nq]
        v_dt: [nv]
        """
        q = np.asarray(q).reshape(self.nq)
        v_dt = np.asarray(v_dt).reshape(self.nv)
        return np.array(pin.integrate(self.model, q, v_dt)).reshape(self.nq)

_smpl_to_rbdl = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 6, 7, 8,
                 15, 16, 17, 24, 25, 26, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 39, 40, 41,
                 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68, 33, 34, 35, 42, 43, 44]
_rbdl_to_smpl = [0, 1, 2, 12, 13, 14, 24, 25, 26, 3, 4, 5, 15, 16, 17, 27, 28, 29, 6, 7, 8, 18, 19, 20, 30, 31, 32,
                 9, 10, 11, 21, 22, 23, 63, 64, 65, 33, 34, 35, 48, 49, 50, 66, 67, 68, 36, 37, 38, 51, 52, 53, 39,
                 40, 41, 54, 55, 56, 42, 43, 44, 57, 58, 59, 45, 46, 47, 60, 61, 62]
_rbdl_to_bullet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30, 31, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 63, 64, 65, 66, 67, 68]
smpl_to_rbdl_data = _smpl_to_rbdl


def set_pose(id_robot, q):
    r"""
    Set the robot configuration.
    """
    p.resetJointStatesMultiDof(id_robot, list(range(1, p.getNumJoints(id_robot))), q[6:][_rbdl_to_bullet].reshape(-1, 1))
    glb_rot = p.getQuaternionFromEuler(euler_convert_np(q[3:6], 'zyx', 'xyz')[[2, 1, 0]])
    p.resetBasePositionAndOrientation(id_robot, q[:3], glb_rot)


def smpl_to_rbdl(poses, trans):
    r"""
    Convert smpl poses and translations to robot configuration q. (numpy, batch)

    :param poses: Array that can reshape to [n, 24, 3, 3].
    :param trans: Array that can reshape to [n, 3].
    :return: Ndarray in shape [n, 75] (3 root position + 72 joint rotation).
    """
    poses = np.array(poses).reshape(-1, 24, 3, 3)
    trans = np.array(trans).reshape(-1, 3)
    euler_poses = rotation_matrix_to_euler_angle_np(poses[:, 1:], 'XYZ').reshape(-1, 69)
    euler_glbrots = rotation_matrix_to_euler_angle_np(poses[:, :1], 'xyz').reshape(-1, 3)
    euler_glbrots = euler_convert_np(euler_glbrots[:, [2, 1, 0]], 'xyz', 'zyx')
    qs = np.concatenate((trans, euler_glbrots, euler_poses[:, _smpl_to_rbdl]), axis=1)
    qs[:, 3:] = normalize_angle(qs[:, 3:])
    return qs


def rbdl_to_smpl(qs):
    r"""
    Convert robot configuration q to smpl poses and translations. (numpy, batch)

    :param qs: Ndarray that can reshape to [n, 75] (3 root position + 72 joint rotation).
    :return: Poses ndarray in shape [n, 24, 3, 3] and translation ndarray in shape [n, 3].
    """
    qs = qs.reshape(-1, 75)
    trans, euler_glbrots, euler_poses = qs[:, :3], qs[:, 3:6], qs[:, 6:][:, _rbdl_to_smpl]
    euler_glbrots = euler_convert_np(euler_glbrots, 'zyx', 'xyz')[:, [2, 1, 0]]
    glbrots = euler_angle_to_rotation_matrix_np(euler_glbrots, 'xyz').reshape(-1, 1, 3, 3)
    poses = euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 23, 3, 3)
    poses = np.concatenate((glbrots, poses), axis=1)
    return poses, trans


def normalize_and_concat(glb_acc, glb_rot):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_rot = glb_rot.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_rot[:, -1])
    ori = torch.cat((glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data


def print_title(s):
    print('============ %s ============' % s)


class Body(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """
    ROOT = 2
    PELVIS = 2
    SPINE = 2
    LHIP = 5
    RHIP = 17
    SPINE1 = 29
    LKNEE = 8
    RKNEE = 20
    SPINE2 = 32
    LANKLE = 11
    RANKLE = 23
    SPINE3 = 35
    LFOOT = 14
    RFOOT = 26
    NECK = 68
    LCLAVICLE = 38
    RCLAVICLE = 53
    HEAD = 71
    LSHOULDER = 41
    RSHOULDER = 56
    LELBOW = 44
    RELBOW = 59
    LWRIST = 47
    RWRIST = 62
    LHAND = 50
    RHAND = 65
