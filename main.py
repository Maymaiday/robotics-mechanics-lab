# 导入必要的库

from copy import copy
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from pydrake.all import (
    ControllabilityMatrix,
    DiagramBuilder,
    Linearize,
    LinearQuadraticRegulator,
    MeshcatVisualizer,
    Saturation,
    SceneGraph,
    Simulator,
    StartMeshcat,
    LeafSystem,
    wrap_to,
    LogVectorOutput,
)

from pydrake.examples import (
    AcrobotGeometry,
    AcrobotInput,
    AcrobotPlant,
    AcrobotState,
    AcrobotSpongController,
)

from underactuated import running_as_notebook



# Start the visualizer (run this cell only once, each instance consumes a port)
meshcat = StartMeshcat()



def UprightState():
    """
    定义平衡点状态 : [pi,0,0,0]
    """
    state = AcrobotState()
    state.set_theta1(np.pi)
    state.set_theta2(0.0)
    state.set_theta1dot(0.0)
    state.set_theta2dot(0.0)
    return state


def acrobot_controllability():
    """
    在顶部平衡点附近检查系统是否可控
    """
    acrobot = AcrobotPlant()
    context = acrobot.CreateDefaultContext()

    input = AcrobotInput()
    input.set_tau(0.0)
    acrobot.get_input_port(0).FixValue(context, input)

    context.get_mutable_continuous_state_vector().SetFromVector(
        UprightState().CopyToVector()
    )

    linearized_acrobot = Linearize(acrobot, context) # 把非线性 Acrobot 系统在顶部平衡点附近线性化
    print( f"The singular values of the controllability matrix are: {np.linalg.svd(
            ControllabilityMatrix(linearized_acrobot),
            compute_uv=False)}"
    )



def BalancingLQR(R_value=1.0):
        # Design an LQR controller for stabilizing the Acrobot around the upright.
        # Returns a (static) AffineSystem that implements the controller (in
        # the original AcrobotState coordinates).

        acrobot = AcrobotPlant()
        context = acrobot.CreateDefaultContext()

        input = AcrobotInput()
        input.set_tau(0.0)
        acrobot.get_input_port(0).FixValue(context, input)

        context.get_mutable_continuous_state_vector().SetFromVector(
            UprightState().CopyToVector()
        )

        Q = np.diag((50.0, 50.0, 5.0, 5.0))  # 惩罚状态误差
        R = [R_value]                        # 惩罚控制输入（u）

        return LinearQuadraticRegulator(acrobot, context, Q, R)



class SpongSwingUpThenLQRController(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        self.DeclareVectorInputPort("state", 4)
        self.DeclareVectorOutputPort("control", 1, self.DoCalcOutput)

        self.swingup = AcrobotSpongController()
        self.swingup_context = self.swingup.CreateDefaultContext()

        self.lqr = BalancingLQR()
        self.lqr_context = self.lqr.CreateDefaultContext()

        self.has_switched = False

    def DoCalcOutput(self, context, output):
        x = self.get_input_port(0).Eval(context)

        x_wrapped = copy(x)
        x_wrapped[0] = wrap_to(x[0], 0.0, 2.0 * np.pi)
        x_wrapped[1] = wrap_to(x[1], -np.pi, np.pi)

        theta1_error = abs(x_wrapped[0] - np.pi)
        theta2_error = abs(x_wrapped[1])
        velocity_norm = np.linalg.norm(x_wrapped[2:4])

        near_upright = (
            theta1_error < 0.6
            and theta2_error < 0.6
            and velocity_norm < 8.0
        )

        if near_upright:  
            if not self.has_switched:
                print("Switch to LQR")
                self.has_switched = True
                
            self.lqr.get_input_port(0).FixValue(self.lqr_context, x_wrapped)
            u = self.lqr.get_output_port(0).Eval(self.lqr_context)
        else:
            self.swingup.get_input_port(0).FixValue(self.swingup_context, x)
            u = self.swingup.get_output_port(0).Eval(self.swingup_context)

        output.SetFromVector(u)




class SpongSwingUpThenLQRController(LeafSystem):
    def __init__(self, R_value=1.0):
        LeafSystem.__init__(self)

        self.DeclareVectorInputPort("state", 4)
        self.DeclareVectorOutputPort("control", 1, self.DoCalcOutput)

        self.swingup = AcrobotSpongController()
        self.swingup_context = self.swingup.CreateDefaultContext()

        self.lqr = BalancingLQR(R_value)
        self.lqr_context = self.lqr.CreateDefaultContext()

        self.has_switched = False

    def DoCalcOutput(self, context, output):
        x = self.get_input_port(0).Eval(context)

        x_wrapped = copy(x)
        x_wrapped[0] = wrap_to(x[0], 0.0, 2.0 * np.pi)
        x_wrapped[1] = wrap_to(x[1], -np.pi, np.pi)

        theta1_error = abs(x_wrapped[0] - np.pi)
        theta2_error = abs(x_wrapped[1])
        velocity_norm = np.linalg.norm(x_wrapped[2:4])

        near_upright = (
            theta1_error < 0.6
            and theta2_error < 0.6
            and velocity_norm < 8.0
        )

        if near_upright:
            if not self.has_switched:
                print("Switch to LQR")
                self.has_switched = True

            self.lqr.get_input_port(0).FixValue(self.lqr_context, x_wrapped)
            u = self.lqr.get_output_port(0).Eval(self.lqr_context)
        else:
            self.swingup.get_input_port(0).FixValue(self.swingup_context, x)
            u = self.swingup.get_output_port(0).Eval(self.swingup_context)

        output.SetFromVector(u)


def simulate_and_log(R_value=1.0, sim_time=30.0, make_animation=True):
    builder = DiagramBuilder()

    acrobot = builder.AddSystem(AcrobotPlant())

    controller = builder.AddSystem(
        SpongSwingUpThenLQRController(R_value=R_value)
    )

    saturation = builder.AddSystem(
        Saturation(min_value=[-5.0], max_value=[5.0])
    )

    builder.Connect(acrobot.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), saturation.get_input_port(0))
    builder.Connect(saturation.get_output_port(0), acrobot.get_input_port(0))

    state_logger = LogVectorOutput(acrobot.get_output_port(0), builder)
    input_logger = LogVectorOutput(saturation.get_output_port(0), builder)

    if make_animation:
        scene_graph = builder.AddSystem(SceneGraph())
        AcrobotGeometry.AddToBuilder(builder, acrobot.get_output_port(0), scene_graph)

        meshcat.Delete()
        meshcat.Set2dRenderMode(xmin=-4, xmax=4, ymin=-4, ymax=4)
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    simulator.set_target_realtime_rate(1.0 if make_animation else 0.0)

    context.SetContinuousState([0.1, 0.0, 0.0, 0.5])

    simulator.Initialize()
    simulator.AdvanceTo(sim_time)

    state_log = state_logger.FindLog(context)
    input_log = input_logger.FindLog(context)

    t = state_log.sample_times()
    x = state_log.data()
    u = input_log.data()

    return t, x, u



def wrap_to_0_2pi(angle):
    return np.mod(angle, 2.0 * np.pi)


def plot_single_result(t, x, u, R_value=1.0, title_suffix=""):
    theta1 = wrap_to_0_2pi(x[0, :])   # θ1 映射到 [0, 2π]，让倒立位置显示为 π
    theta2 = x[1, :]                  # θ2 不 wrap，避免在 0 和 2π 之间出现竖线跳变

    plt.figure(figsize=(10, 5))

    plt.plot(
        t,
        theta1,
        linewidth=2,
        label=r"$\theta_1$ (Link 1 Absolute Angle)"
    )

    plt.plot(
        t,
        theta2,
        linewidth=2,
        label=r"$\theta_2$ (Link 2 Relative Angle)"
    )

    plt.axhline(
        np.pi,
        linestyle="--",
        linewidth=2,
        color="red",
        label=r"Upright Equilibrium ($\theta_1=\pi$)"
    )

    plt.axhline(
        0.0,
        linestyle=":",
        linewidth=1.5,
        color="gray",
        label=r"$\theta_2=0$"
    )

    if title_suffix == "":
        plt.title(f"Acrobot Joint Angle Response (R = {R_value})", fontsize=15)
    else:
        plt.title(
            f"Acrobot Joint Angle Response (R = {R_value}, {title_suffix})",
            fontsize=15
        )

    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Joint Angle (rad)", fontsize=13)

    plt.yticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        [
            r"$-\pi$",
            r"$-\pi/2$",
            "0",
            r"$\pi/2$",
            r"$\pi$",
            r"$3\pi/2$",
            r"$2\pi$",
        ],
    )

    plt.ylim([-np.pi, 2 * np.pi])
    plt.grid(True)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def run_R_ablation(R_values=[0.1, 1.0, 10.0], sim_time=30.0):
    results = []
    
    for R_value in R_values:
        print(f"Running simulation with R = {R_value}")
        t, x, u = simulate_and_log(
            R_value=R_value,
            sim_time=sim_time,
            make_animation=False
        )
        
        # 计算角度误差
        theta1 = x[0, :]
        angle_error = np.abs(np.arctan2(np.sin(theta1 - np.pi), np.cos(theta1 - np.pi)))
        final_error = angle_error[-1]
        
        max_u = np.max(np.abs(u[0, :]))
        rms_u = np.sqrt(np.mean(u[0, :] ** 2))
        
        results.append({
            "R_value": R_value,
            "final_theta1_error (rad)": final_error,
            "max_|u| (N·m)": max_u,
            "rms_u (N·m)": rms_u
        })
        
        print(f"   Final theta1 = {x[0, -1]:.3f} rad (norm error = {final_error:.4f})")
    
    df = pd.DataFrame(results)
    print("\n=== R Ablation Summary ===")
    display(df)
    return results



acrobot_controllability()

t, x, u = simulate_and_log(
    R_value=1.0,
    sim_time=30.0,
    make_animation=True
)

plot_single_result(t, x, u, R_value=1.0)

ablation_results = run_R_ablation(
    R_values=[0.1, 1.0, 10.0],
    sim_time=30.0
)
