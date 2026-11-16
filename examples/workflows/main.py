import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from faas.system import ScalingConfiguration
from faas.system.core import Function, FunctionContainer, FunctionImage, FunctionRequest

import examples.basic.main as basic
from sim.benchmark import Benchmark
from sim.context.platform.deployment.model import SimFunctionDeployment, SimScalingConfiguration
from sim.context.platform.request.service import RequestService
from sim.core import Environment
from sim.faas import FunctionSimulator, SimFunctionReplica, SimulatorFactory
from sim.faas.core import SimResourceConfiguration
from sim.faassim import DockerDeploySimMixin, DummySimulator, Simulation
from sim.workflow import (
    DataAwareWorkflowScheduler,
    DataPassingMode,
    MakespanWorkflowScheduler,
    RoundRobinWorkflowScheduler,
    WorkflowDAG,
    WorkflowEngine,
    WorkflowExecutionResult,
    WorkflowScheduler,
    WorkflowSkippyScheduler,
    WorkflowTask,
)

logger = logging.getLogger(__name__)


class HeterogeneousFunctionSimulator(DockerDeploySimMixin, DummySimulator):
    """
    Realistic heterogeneous function simulator for faas-sim:
      - Pulls container images from DockerRegistry on cold start (`DockerDeploySimMixin`).
      - Scales execution duration (`FET`) according to:
        1. Function base workload (`preprocess`: 0.6s, `resnet50-inference`: 2.4s, `postprocess`: 0.6s on 4000m CPU).
        2. Node hardware capacity (`cpu_millis` up to 4x single-task core speedup, plus 1.5x CUDA GPU acceleration).
        3. In-flight request contention on the host node.
    """

    BASE_DURATIONS = {
        "python-pi": 0.6,
        "resnet50-inference": 2.4,
        "postprocess-fn": 0.6,
    }

    def __init__(self, fn: FunctionContainer):
        self.fn = fn

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest):
        request_service: RequestService = env.context.request_service
        request_service.add_request(request)
        inflight = sum(
            1 for r in request_service.get_inflight_request(replica.node.name)
            if getattr(r, "replica", None) is not None
            and getattr(r.replica, "node", None) is not None
            and r.replica.node.name == replica.node.name
            and request_service.get_end_ts(r.request_id) is None
        )

        base_duration = self.BASE_DURATIONS.get(request.name, 1.0)
        ether_node = replica.node.ether_node
        speedup_ratio = WorkflowScheduler.compute_node_speedup_ratio(ether_node, reference_cpu_millis=4000.0)
        contention_multiplier = 1.0 + 0.25 * max(0, inflight - 1)
        exec_duration = base_duration * speedup_ratio * contention_multiplier

        yield env.timeout(exec_duration)
        request_service.remove_request(request.request_id)


class HeterogeneousSimulatorFactory(SimulatorFactory):
    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        return HeterogeneousFunctionSimulator(fn)


class EdgeAIWorkflowBenchmark(Benchmark):
    """
    Full faas-sim workflow benchmark evaluating a 3-stage Edge AI pipeline:
      preprocess (python-pi) -> inference (resnet50-inference) -> postprocess (postprocess-fn)
    Executes two consecutive runs:
      - Run 1 (Cold Start): scale_min=0, pods are placed on-time via WorkflowSkippyScheduler.
      - Run 2 (Warm Start): reuses warm replicas deployed during Run 1.
    """

    def __init__(
        self,
        scheduler: WorkflowScheduler,
        passing_mode: DataPassingMode = DataPassingMode.DIRECT,
        payload_bytes: int = 20 * 1024 * 1024,
        pin_root_to_rpi: bool = False
    ):
        self.scheduler = scheduler
        self.passing_mode = passing_mode
        self.payload_bytes = payload_bytes
        self.pin_root_to_rpi = pin_root_to_rpi
        self.results: List[WorkflowExecutionResult] = []

    def setup(self, env: Environment):
        basic.ExampleBenchmark().setup(env)

    @staticmethod
    def _prepare_postprocess_deployment() -> SimFunctionDeployment:
        post_img = FunctionImage(image="python-pi-cpu")
        post_fn = Function("postprocess-fn", fn_images=[post_img])
        post_container = FunctionContainer(post_img, SimResourceConfiguration())
        return SimFunctionDeployment(
            post_fn,
            [post_container],
            SimScalingConfiguration(ScalingConfiguration(scale_min=0))
        )

    def run(self, env: Environment):
        eb = basic.ExampleBenchmark()
        pi_fd = eb.prepare_python_pi_deployment()
        pi_fd.scaling_config = SimScalingConfiguration(ScalingConfiguration(scale_min=0))
        resnet_fd = eb.prepare_resnet_inference_deployment()
        resnet_fd.scaling_config = SimScalingConfiguration(ScalingConfiguration(scale_min=0))
        post_fd = self._prepare_postprocess_deployment()

        yield from env.faas.deploy(pi_fd)
        yield from env.faas.deploy(resnet_fd)
        yield from env.faas.deploy(post_fd)

        dag = WorkflowDAG("edge-ai-pipeline", default_mode=self.passing_mode)
        dag.add_task(WorkflowTask(name="preprocess", function_name="python-pi", estimated_duration=0.6))
        dag.add_task(WorkflowTask(name="inference", function_name="resnet50-inference", estimated_duration=2.4))
        dag.add_task(WorkflowTask(name="postprocess", function_name="postprocess-fn", estimated_duration=0.6))

        dag.add_dependency("preprocess", "inference", size_bytes=self.payload_bytes)
        dag.add_dependency("inference", "postprocess", size_bytes=self.payload_bytes // 2)

        storage = [n.name for n in env.topology.get_nodes() if "server" in n.name][0]
        if hasattr(self.scheduler, "storage_node_ref") and self.scheduler.storage_node_ref is None:
            self.scheduler.storage_node_ref = storage

        engine = WorkflowEngine(env, scheduler=self.scheduler, storage_node=storage)

        placements_override: Optional[Dict[str, str]] = None
        if self.pin_root_to_rpi:
            rpi_sensor = [n.name for n in env.topology.get_nodes() if "rpi" in n.name][0]
            placements_override = {"preprocess": rpi_sensor}

        # Run 1: Cold-start execution
        res_cold = yield from engine.execute_workflow(
            dag, task_placements=dict(placements_override) if placements_override else None
        )
        self.results.append(res_cold)

        # Run 2: Warm-start execution
        res_warm = yield from engine.execute_workflow(
            dag, task_placements=dict(placements_override) if placements_override else None
        )
        self.results.append(res_warm)


def run_single_experiment(
    scheduler: WorkflowScheduler,
    mode: DataPassingMode,
    payload_bytes: int = 20 * 1024 * 1024,
    pin_root_to_rpi: bool = False
) -> Tuple[WorkflowExecutionResult, WorkflowExecutionResult]:
    np.random.seed(42)
    topology = basic.example_topology()
    benchmark = EdgeAIWorkflowBenchmark(
        scheduler=scheduler,
        passing_mode=mode,
        payload_bytes=payload_bytes,
        pin_root_to_rpi=pin_root_to_rpi
    )
    sim = Simulation(topology, benchmark)
    sim.create_scheduler = WorkflowSkippyScheduler.create()
    sim.create_simulator_factory = HeterogeneousSimulatorFactory
    sim.run()
    return benchmark.results[0], benchmark.results[1]


class ParallelVideoAnalyticsBenchmark(Benchmark):
    """
    Parallel Fan-Out / Fan-In Video Analytics DAG (inspired by SONIC, USENIX ATC '21):
      split_video (python-pi on edge gateway nuc)
        -> K parallel classify_branch_i (resnet50-inference, 2 MB video chunk each)
        -> aggregate_results (postprocess-fn, 128 KB summary each)
    Demonstrates the trade-off between Data Locality (co-locating all K branches on one host,
    which suffers multi-tenant CPU/GPU contention) vs. Parallelism & Makespan Optimization.
    """

    def __init__(
        self,
        scheduler: WorkflowScheduler,
        passing_mode: DataPassingMode = DataPassingMode.DIRECT,
        fanout_degree: int = 4,
        chunk_bytes: int = 2 * 1024 * 1024
    ):
        self.scheduler = scheduler
        self.passing_mode = passing_mode
        self.fanout_degree = fanout_degree
        self.chunk_bytes = chunk_bytes
        self.results: List[WorkflowExecutionResult] = []

    def setup(self, env: Environment):
        basic.ExampleBenchmark().setup(env)

    def run(self, env: Environment):
        eb = basic.ExampleBenchmark()
        pi_fd = eb.prepare_python_pi_deployment()
        pi_fd.scaling_config = SimScalingConfiguration(ScalingConfiguration(scale_min=0))
        resnet_fd = eb.prepare_resnet_inference_deployment()
        resnet_fd.scaling_config = SimScalingConfiguration(ScalingConfiguration(scale_min=0))
        post_fd = EdgeAIWorkflowBenchmark._prepare_postprocess_deployment()

        yield from env.faas.deploy(pi_fd)
        yield from env.faas.deploy(resnet_fd)
        yield from env.faas.deploy(post_fd)

        dag = WorkflowDAG(f"video-analytics-k{self.fanout_degree}", default_mode=self.passing_mode)
        dag.add_task(WorkflowTask(name="split_video", function_name="python-pi", estimated_duration=0.6))
        dag.add_task(WorkflowTask(name="aggregate", function_name="postprocess-fn", estimated_duration=0.6))

        for i in range(self.fanout_degree):
            branch_name = f"classify_{i}"
            dag.add_task(WorkflowTask(name=branch_name, function_name="resnet50-inference", estimated_duration=2.4))
            dag.add_dependency("split_video", branch_name, size_bytes=self.chunk_bytes)
            dag.add_dependency(branch_name, "aggregate", size_bytes=128 * 1024)

        storage = [n.name for n in env.topology.get_nodes() if "server" in n.name][0]
        if hasattr(self.scheduler, "storage_node_ref") and self.scheduler.storage_node_ref is None:
            self.scheduler.storage_node_ref = storage

        nuc_gateway = [n.name for n in env.topology.get_nodes() if "nuc" in n.name][0]
        engine = WorkflowEngine(env, scheduler=self.scheduler, storage_node=storage)

        res_cold = yield from engine.execute_workflow(dag, task_placements={"split_video": nuc_gateway})
        self.results.append(res_cold)

        res_warm = yield from engine.execute_workflow(dag, task_placements={"split_video": nuc_gateway})
        self.results.append(res_warm)


def run_fanout_experiment(
    scheduler: WorkflowScheduler,
    mode: DataPassingMode,
    fanout_degree: int
) -> Tuple[WorkflowExecutionResult, WorkflowExecutionResult]:
    np.random.seed(42)
    topology = basic.example_topology()
    benchmark = ParallelVideoAnalyticsBenchmark(
        scheduler=scheduler,
        passing_mode=mode,
        fanout_degree=fanout_degree
    )
    sim = Simulation(topology, benchmark)
    sim.create_scheduler = WorkflowSkippyScheduler.create()
    sim.create_simulator_factory = HeterogeneousSimulatorFactory
    sim.run()
    return benchmark.results[0], benchmark.results[1]


def generate_plots(
    policy_summary: List[Tuple[str, WorkflowExecutionResult, WorkflowExecutionResult]],
    scaling_summary: Dict[str, List[Tuple[int, WorkflowExecutionResult, WorkflowExecutionResult]]],
    fanout_summary: Dict[str, List[Tuple[int, WorkflowExecutionResult, WorkflowExecutionResult]]],
    output_dir: str
) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    # -------------------------------------------------------------------------
    # Figure 1: Policy & Data-Passing Comparison (Cold vs Warm + Breakdown)
    # -------------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    labels = [item[0].replace(" + ", "\n+ ") for item in policy_summary]
    cold_makespans = [item[1].makespan for item in policy_summary]
    warm_makespans = [item[2].makespan for item in policy_summary]
    warm_computes = [item[2].total_compute_time for item in policy_summary]
    warm_transfers = [item[2].total_data_transfer_time for item in policy_summary]

    x = np.arange(len(labels))
    width = 0.36

    bars_cold = ax1.bar(
        x - width / 2, cold_makespans, width,
        label="Run 1: Cold Start (scale_min=0)", color="#3b82f6", edgecolor="#1e3a8a"
    )
    bars_warm = ax1.bar(
        x + width / 2, warm_makespans, width,
        label="Run 2: Warm Start (Replica Reuse)", color="#10b981", edgecolor="#065f46"
    )

    ax1.set_ylabel("End-to-End Makespan (seconds)")
    ax1.set_title("A. Cold-Start vs. Warm-Start Workflow Makespan (20 MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    for bar in bars_cold:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars_warm:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, max(cold_makespans) * 1.16)

    # Panel B: Stacked Breakdown of Warm Execution (Compute vs Data Transfer)
    ax2.bar(
        x, warm_computes, width=0.48,
        label="Function Compute Time (s)", color="#6366f1", edgecolor="#312e81"
    )
    ax2.bar(
        x, warm_transfers, width=0.48, bottom=warm_computes,
        label="Intermediate Data Transfer Time (s)", color="#f59e0b", edgecolor="#78350f"
    )

    ax2.set_ylabel("Time Breakdown (seconds)")
    ax2.set_title("B. Warm Execution Breakdown: Compute vs. Data Transfer")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    for i in range(len(x)):
        total = warm_computes[i] + warm_transfers[i]
        ax2.annotate(
            f"Total: {total:.2f}s\n(Net: {warm_transfers[i]:.2f}s)",
            xy=(x[i], total),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5
        )
    ax2.set_ylim(0, max(w + t for w, t in zip(warm_computes, warm_transfers)) * 1.22)

    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, "workflow_policy_comparison.png")
    fig1.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: Payload Size Scaling (Sensor-Anchored Edge AI Pipeline)
    # -------------------------------------------------------------------------
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    styles = {
        "RoundRobin + RemoteStorage": ("#ef4444", "o", "--"),
        "RoundRobin + Direct": ("#f59e0b", "s", "-."),
        "DataAware + Direct": ("#10b981", "^", "-"),
        "Makespan + Direct": ("#2563eb", "D", "-"),
    }

    for name, records in scaling_summary.items():
        color, marker, linestyle = styles[name]
        payloads_mb = [r[0] for r in records]
        makespans = [r[2].makespan for r in records]
        overhead_ratios = [r[2].data_transfer_overhead_ratio * 100.0 for r in records]

        ax3.plot(
            payloads_mb, makespans, label=name,
            color=color, marker=marker, linestyle=linestyle, linewidth=2.2, markersize=6
        )
        ax4.plot(
            payloads_mb, overhead_ratios, label=name,
            color=color, marker=marker, linestyle=linestyle, linewidth=2.2, markersize=6
        )

    ax3.set_xlabel("Intermediate Payload Size (MB)")
    ax3.set_ylabel("Warm End-to-End Makespan (seconds)")
    ax3.set_title("A. End-to-End Makespan vs. Payload Size (Edge Sensor Root)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left")

    ax4.set_xlabel("Intermediate Payload Size (MB)")
    ax4.set_ylabel("Data Transfer Share of Makespan (%)")
    ax4.set_title("B. Data Transfer Overhead Ratio vs. Payload Size")
    ax4.set_ylim(0, 100)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="center right")

    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, "workflow_payload_scaling.png")
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Figure 3: Parallel Fan-Out Scaling (Video Analytics DAG, K = 1..8)
    # -------------------------------------------------------------------------
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for name, records in fanout_summary.items():
        color, marker, linestyle = styles[name]
        fanouts_k = [r[0] for r in records]
        makespans = [r[2].makespan for r in records]
        distinct_nodes = [
            len({
                rec.node_name
                for t_name, rec in r[2].task_records.items()
                if t_name.startswith("classify_")
            })
            for r in records
        ]

        ax5.plot(
            fanouts_k, makespans, label=name,
            color=color, marker=marker, linestyle=linestyle, linewidth=2.2, markersize=6
        )
        ax6.plot(
            fanouts_k, distinct_nodes, label=name,
            color=color, marker=marker, linestyle=linestyle, linewidth=2.2, markersize=6
        )

    ax5.set_xlabel("Parallel Fan-Out Degree (K Classification Branches)")
    ax5.set_ylabel("Warm End-to-End Makespan (seconds)")
    ax5.set_title("A. Video Analytics Makespan vs. Fan-Out Degree (K)")
    ax5.set_xticks([1, 2, 4, 6, 8])
    ax5.grid(True, linestyle="--", alpha=0.5)
    ax5.legend(loc="upper left")

    ax6.set_xlabel("Parallel Fan-Out Degree (K Classification Branches)")
    ax6.set_ylabel("Distinct Worker Nodes Used by Parallel Stage")
    ax6.set_title("B. Parallel Stage Distribution Across Worker Nodes")
    ax6.set_xticks([1, 2, 4, 6, 8])
    ax6.set_ylim(0.5, 8.8)
    ax6.grid(True, linestyle="--", alpha=0.5)
    ax6.legend(loc="upper left")

    fig3.tight_layout()
    fig3_path = os.path.join(output_dir, "workflow_fanout_scaling.png")
    fig3.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig3)

    return fig1_path, fig2_path, fig3_path


def main():
    logging.basicConfig(level=logging.WARNING)

    # 1. Policy & Passing Mode Comparison (Unpinned Root, 20 MB Intermediate Payload)
    experiments = [
        ("RoundRobin + RemoteStorage", RoundRobinWorkflowScheduler(), DataPassingMode.REMOTE_STORAGE),
        ("RoundRobin + Direct", RoundRobinWorkflowScheduler(), DataPassingMode.DIRECT),
        ("DataAware + Direct", DataAwareWorkflowScheduler(), DataPassingMode.DIRECT),
        ("Makespan + Direct", MakespanWorkflowScheduler(base_compute_time=1.2), DataPassingMode.DIRECT),
    ]

    policy_summary = []
    print("\n[Experiment 1] 3-Stage Edge AI Pipeline (preprocess -> inference -> postprocess, 20 MB Payload)")
    print("=" * 118)
    print(
        f"{'Policy & Passing Mode':<28} | {'Cold Makespan':>13} | {'Warm Makespan':>13} | "
        f"{'Compute (s)':>11} | {'Transfer (s)':>12} | {'Nodes (prep -> inf -> post)'}"
    )
    print("-" * 118)

    for label, scheduler, mode in experiments:
        cold, warm = run_single_experiment(scheduler, mode, payload_bytes=20 * 1024 * 1024, pin_root_to_rpi=False)
        policy_summary.append((label, cold, warm))
        nodes_str = (
            f"{warm.task_records['preprocess'].node_name} -> "
            f"{warm.task_records['inference'].node_name} -> "
            f"{warm.task_records['postprocess'].node_name}"
        )
        print(
            f"{label:<28} | {cold.makespan:>12.4f}s | {warm.makespan:>12.4f}s | "
            f"{warm.total_compute_time:>10.4f}s | {warm.total_data_transfer_time:>11.4f}s | {nodes_str}"
        )
    print("=" * 118)

    # 2. Payload Scaling Benchmark (Edge Sensor Root Pinned to rpi3, 1 MB to 50 MB)
    payload_sizes_mb = [1, 5, 15, 30, 50]
    scaling_summary: Dict[str, List[Tuple[int, WorkflowExecutionResult, WorkflowExecutionResult]]] = {
        "RoundRobin + RemoteStorage": [],
        "RoundRobin + Direct": [],
        "DataAware + Direct": [],
        "Makespan + Direct": [],
    }

    print("\n[Experiment 2] Payload Scaling with Sensor-Anchored Root (preprocess on rpi3, 1 MB -> 50 MB)")
    print("=" * 118)
    print(
        f"{'Payload':<8} | {'RR+Remote (s)':>14} | {'RR+Direct (s)':>14} | "
        f"{'DataAware+Direct (s)':>21} | {'Makespan+Direct (s)':>20} | {'Makespan Placement (prep->inf->post)'}"
    )
    print("-" * 118)

    for mb in payload_sizes_mb:
        size_bytes = mb * 1024 * 1024
        c_rrr, w_rrr = run_single_experiment(
            RoundRobinWorkflowScheduler(), DataPassingMode.REMOTE_STORAGE, size_bytes, pin_root_to_rpi=True
        )
        c_rrd, w_rrd = run_single_experiment(
            RoundRobinWorkflowScheduler(), DataPassingMode.DIRECT, size_bytes, pin_root_to_rpi=True
        )
        c_da, w_da = run_single_experiment(
            DataAwareWorkflowScheduler(), DataPassingMode.DIRECT, size_bytes, pin_root_to_rpi=True
        )
        c_ms, w_ms = run_single_experiment(
            MakespanWorkflowScheduler(base_compute_time=1.2), DataPassingMode.DIRECT, size_bytes, pin_root_to_rpi=True
        )

        scaling_summary["RoundRobin + RemoteStorage"].append((mb, c_rrr, w_rrr))
        scaling_summary["RoundRobin + Direct"].append((mb, c_rrd, w_rrd))
        scaling_summary["DataAware + Direct"].append((mb, c_da, w_da))
        scaling_summary["Makespan + Direct"].append((mb, c_ms, w_ms))

        ms_nodes = (
            f"{w_ms.task_records['preprocess'].node_name} -> "
            f"{w_ms.task_records['inference'].node_name} -> "
            f"{w_ms.task_records['postprocess'].node_name}"
        )
        print(
            f"{mb:>4} MB  | {w_rrr.makespan:>13.4f}s | {w_rrd.makespan:>13.4f}s | "
            f"{w_da.makespan:>20.4f}s | {w_ms.makespan:>19.4f}s | {ms_nodes}"
        )
    print("=" * 118)

    # 3. Parallel Fan-Out Benchmark (Video Analytics DAG: split_video -> K x classify_i -> aggregate)
    fanout_degrees = [1, 2, 4, 6, 8]
    fanout_summary: Dict[str, List[Tuple[int, WorkflowExecutionResult, WorkflowExecutionResult]]] = {
        "RoundRobin + RemoteStorage": [],
        "RoundRobin + Direct": [],
        "DataAware + Direct": [],
        "Makespan + Direct": [],
    }

    print("\n[Experiment 3] Parallel Video Analytics Fan-Out Scaling (split_video -> K x classify -> aggregate, 2 MB/chunk)")
    print("=" * 118)
    print(
        f"{'Fanout K':<8} | {'RR+Remote (s)':>14} | {'RR+Direct (s)':>14} | "
        f"{'DataAware+Direct (s)':>21} | {'Makespan+Direct (s)':>20} | {'Distinct Nodes (DA vs MS)'}"
    )
    print("-" * 118)

    for k in fanout_degrees:
        c_rrr, w_rrr = run_fanout_experiment(RoundRobinWorkflowScheduler(), DataPassingMode.REMOTE_STORAGE, k)
        c_rrd, w_rrd = run_fanout_experiment(RoundRobinWorkflowScheduler(), DataPassingMode.DIRECT, k)
        c_da, w_da = run_fanout_experiment(DataAwareWorkflowScheduler(), DataPassingMode.DIRECT, k)
        c_ms, w_ms = run_fanout_experiment(MakespanWorkflowScheduler(base_compute_time=1.2), DataPassingMode.DIRECT, k)

        fanout_summary["RoundRobin + RemoteStorage"].append((k, c_rrr, w_rrr))
        fanout_summary["RoundRobin + Direct"].append((k, c_rrd, w_rrd))
        fanout_summary["DataAware + Direct"].append((k, c_da, w_da))
        fanout_summary["Makespan + Direct"].append((k, c_ms, w_ms))

        da_distinct = len({r.node_name for name, r in w_da.task_records.items() if name.startswith("classify_")})
        ms_distinct = len({r.node_name for name, r in w_ms.task_records.items() if name.startswith("classify_")})
        print(
            f"   K = {k:<2} | {w_rrr.makespan:>13.4f}s | {w_rrd.makespan:>13.4f}s | "
            f"{w_da.makespan:>20.4f}s | {w_ms.makespan:>19.4f}s | DA: {da_distinct} node(s), MS: {ms_distinct} node(s)"
        )
    print("=" * 118)

    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    fig1_path, fig2_path, fig3_path = generate_plots(policy_summary, scaling_summary, fanout_summary, plots_dir)
    print(f"\nSaved comparison plots to:\n  - {fig1_path}\n  - {fig2_path}\n  - {fig3_path}")


if __name__ == "__main__":
    main()
