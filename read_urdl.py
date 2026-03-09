from threading import Thread
import time
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

def vix():
    model, collision_model, visual_model = pin.buildModelsFromUrdf("data/rbdl_model/physics.urdf")
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    q0 = pin.neutral(model)
    viz.display(q0)
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("🛑 停止可视化。")
Thread(target= vix()).run()