import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

# 你的 URDF 路径
urdf_path = "data/rbdl_model/physics.urdf"
mesh_dir  = "data/rbdl_model"

# 1. 加载模型（不依赖 collision/visual）
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# 2. 启动 MeshCat
viz = MeshcatVisualizer(model, None, None)
viz.initViewer(open=True)
viz.viewer.delete()

print("⚡ Skeleton Viewer running! ")

# 获取所有关节的父子关系
parents = model.parents
names = model.names

# 给每个 joint 创建一个小球（点）
for i in range(len(names)):
    if i == 0:
        continue  # universe

    viz.viewer[f"skeleton/joint_{i}"].set_object(
        g.Sphere(0.015),
        meshcat.geometry.MeshLambertMaterial(color=0x00ffff)
    )

# 创建骨架线段
for i in range(1, len(names)):
    parent = parents[i]
    if parent >= 0:
        viz.viewer[f"skeleton/line_{parent}_{i}"].set_object(
            g.Line(g.PointsGeometry(), g.MeshLambertMaterial(color=0xff0000))
        )


def update_skeleton(q):
    """更新骨架线段的位置"""
    pin.forwardKinematics(model, data, q)

    for i in range(1, len(names)):
        # 当前 joint 的位姿
        Mi = data.oMi[i]
        pi = Mi.translation

        # 设置 joint 点的位置
        T = tf.translation_matrix(pi)
        viz.viewer[f"skeleton/joint_{i}"].set_transform(T)

        # 父节点位置
        parent = parents[i]
        if parent >= 0:
            p_parent = data.oMi[parent].translation

            # line geometry: two points A, B
            pts = np.array([p_parent, pi]).T  # shape = (3, 2)
            viz.viewer[f"skeleton/line_{parent}_{i}"].set_object(
                g.Line(
                    g.PointsGeometry(pts),
                    g.MeshLambertMaterial(color=0xff0000)
                )
            )

# 初始位姿
q0 = pin.neutral(model)
q0[:3] = np.array([0, 0, 1.0]) # 让 skeleton 在空中

update_skeleton(q0)

print("📌 打开浏览器访问: http://127.0.0.1:7000/static/")

