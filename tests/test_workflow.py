import pytest
import simpy

import examples.basic.main as basic
from sim.core import Environment
from sim.factory.flow import UninterruptingFlowFactory
from sim.workflow.model import WorkflowDAG, WorkflowTask, DataDependency, DataPassingMode
from sim.workflow.channel import (
    DirectDataChannel,
    RemoteStorageDataChannel,
    ColocatedDataChannel,
    DataTransferController,
)
from sim.workflow.engine import WorkflowEngine, WorkflowExecutionResult
from sim.workflow.scheduler import (
    RoundRobinWorkflowScheduler,
    DataAwareWorkflowScheduler,
    WorkflowSkippyScheduler,
)


class TestWorkflowDAG:

    def test_dag_construction_and_ordering(self):
        dag = WorkflowDAG("test-ml-pipeline")
        t1 = WorkflowTask(name="preprocess", function_name="data-prep")
        t2 = WorkflowTask(name="train", function_name="model-training")
        t3 = WorkflowTask(name="evaluate", function_name="model-eval")

        dag.add_task(t1).add_task(t2).add_task(t3)
        dag.add_dependency("preprocess", "train", size_bytes=10 * 1024 * 1024)
        dag.add_dependency("train", "evaluate", size_bytes=2 * 1024 * 1024)

        order = dag.get_execution_order()
        assert order == ["preprocess", "train", "evaluate"]
        assert dag.get_root_tasks() == [t1]
        assert dag.get_terminal_tasks() == [t3]

    def test_cycle_detection(self):
        dag = WorkflowDAG("cycle-test")
        dag.add_task(WorkflowTask(name="A", function_name="fn-a"))
        dag.add_task(WorkflowTask(name="B", function_name="fn-b"))
        dag.add_task(WorkflowTask(name="C", function_name="fn-c"))

        dag.add_dependency("A", "B", size_bytes=100)
        dag.add_dependency("B", "C", size_bytes=100)

        with pytest.raises(ValueError, match="contains a cycle"):
            dag.add_dependency("C", "A", size_bytes=100)

    def test_self_dependency_rejected(self):
        dag = WorkflowDAG("self-dep-test")
        dag.add_task(WorkflowTask(name="A", function_name="fn-a"))
        with pytest.raises(ValueError, match="Self-dependencies are not allowed"):
            dag.add_dependency("A", "A", size_bytes=50)

    def test_per_edge_data_passing_mode(self):
        dag = WorkflowDAG("hybrid-dag", default_mode=DataPassingMode.DIRECT)
        t1 = WorkflowTask(name="A", function_name="fn-a")
        t2 = WorkflowTask(name="B", function_name="fn-b")
        t3 = WorkflowTask(name="C", function_name="fn-c")

        dag.add_task(t1).add_task(t2).add_task(t3)
        dag.add_dependency("A", "B", size_bytes=1024, mode=DataPassingMode.COLOCATED)
        dag.add_dependency("B", "C", size_bytes=2048)  # Should inherit default DIRECT

        deps_from_a = dag.get_downstream_dependencies("A")
        assert deps_from_a[0].mode == DataPassingMode.COLOCATED

        deps_from_b = dag.get_downstream_dependencies("B")
        assert deps_from_b[0].mode == DataPassingMode.DIRECT


class TestDataChannels:

    def test_channel_comparison(self):
        env = Environment()
        env.topology = basic.example_topology()
        env.flow_factory = UninterruptingFlowFactory()

        nodes = list(env.topology.get_nodes())
        edge_nodes = [n for n in nodes if 'rpi' in n.name]
        cloud_nodes = [n for n in nodes if 'server' in n.name]

        node_a = edge_nodes[0].name
        node_b = edge_nodes[1].name
        cloud_storage = cloud_nodes[0].name

        payload_size = 5 * 1024 * 1024  # 5 MB
        results = {}

        def simulation_process():
            # 1. Colocated transfer on node_a
            colocated_ch = ColocatedDataChannel()
            rec_coloc = yield from colocated_ch.transfer(
                env, source_node=node_a, target_node=node_a, size_bytes=payload_size,
                source_task="taskA", target_task="taskB"
            )
            results["colocated"] = rec_coloc.duration

            # 2. Direct P2P transfer: node_a -> node_b
            direct_ch = DirectDataChannel()
            rec_direct = yield from direct_ch.transfer(
                env, source_node=node_a, target_node=node_b, size_bytes=payload_size,
                source_task="taskA", target_task="taskB"
            )
            results["direct"] = rec_direct.duration

            # 3. Remote storage transfer (Triangular): node_a -> cloud_storage -> node_b
            remote_ch = RemoteStorageDataChannel(storage_node=cloud_storage)
            rec_remote = yield from remote_ch.transfer(
                env, source_node=node_a, target_node=node_b, size_bytes=payload_size,
                source_task="taskA", target_task="taskB"
            )
            results["remote"] = rec_remote.duration

        env.process(simulation_process())
        env.run()

        assert results["colocated"] < results["direct"]
        assert results["direct"] < results["remote"]
        assert results["remote"] > results["direct"] * 1.5

    def test_data_transfer_controller_hybrid(self):
        env = Environment()
        env.topology = basic.example_topology()
        env.flow_factory = UninterruptingFlowFactory()

        nodes = list(env.topology.get_nodes())
        edge_nodes = [n for n in nodes if 'rpi' in n.name]
        cloud_nodes = [n for n in nodes if 'server' in n.name]

        node_a = edge_nodes[0].name
        node_b = edge_nodes[1].name
        storage = cloud_nodes[0].name

        controller = DataTransferController(storage_node=storage)
        results = []

        def hybrid_process():
            # Hop 1: Direct P2P
            rec1 = yield from controller.transfer(
                env=env,
                mode=DataPassingMode.DIRECT,
                source_node=node_a,
                target_node=node_b,
                size_bytes=1024 * 1024,
                source_task="task1",
                target_task="task2"
            )
            results.append(rec1)

            # Hop 2: Colocated
            rec2 = yield from controller.transfer(
                env=env,
                mode=DataPassingMode.COLOCATED,
                source_node=node_b,
                target_node=node_b,
                size_bytes=1024 * 1024,
                source_task="task2",
                target_task="task3"
            )
            results.append(rec2)

            # Hop 3: Remote storage
            rec3 = yield from controller.transfer(
                env=env,
                mode=DataPassingMode.REMOTE_STORAGE,
                source_node=node_b,
                target_node=storage,
                size_bytes=1024 * 1024,
                source_task="task3",
                target_task="task4"
            )
            results.append(rec3)

        env.process(hybrid_process())
        env.run()

        assert len(results) == 3
        all_records = controller.get_all_records()
        assert len(all_records) == 3
        assert [r.mode for r in all_records] == [
            DataPassingMode.DIRECT,
            DataPassingMode.COLOCATED,
            DataPassingMode.REMOTE_STORAGE
        ]


class TestWorkflowEngine:

    def test_workflow_engine_end_to_end(self):
        env = Environment()
        env.topology = basic.example_topology()
        env.flow_factory = UninterruptingFlowFactory()

        nodes = list(env.topology.get_nodes())
        edge_nodes = [n for n in nodes if 'rpi' in n.name]
        cloud_nodes = [n for n in nodes if 'server' in n.name]

        node1 = edge_nodes[0].name
        node2 = edge_nodes[1].name
        storage = cloud_nodes[0].name

        # Pipeline: A -> B -> C with 10MB payload between each step
        payload = 10 * 1024 * 1024
        task_duration = 0.2

        def build_dag(mode: DataPassingMode) -> WorkflowDAG:
            dag = WorkflowDAG(f"pipeline-{mode.value}", default_mode=mode)
            dag.add_task(WorkflowTask(name="prep", function_name="data-prep"))
            dag.add_task(WorkflowTask(name="train", function_name="model-train"))
            dag.add_task(WorkflowTask(name="eval", function_name="model-eval"))
            dag.add_dependency("prep", "train", size_bytes=payload)
            dag.add_dependency("train", "eval", size_bytes=payload)
            return dag

        # 1. Colocated execution (all on node1)
        engine_coloc = WorkflowEngine(env, storage_node=storage, default_task_duration=task_duration)
        dag_coloc = build_dag(DataPassingMode.COLOCATED)
        res_coloc: list = []

        def run_coloc():
            r = yield from engine_coloc.execute_workflow(
                dag_coloc, task_placements={"prep": node1, "train": node1, "eval": node1}
            )
            res_coloc.append(r)

        env.process(run_coloc())
        env.run()

        # 2. Direct P2P execution (distributed across node1 and node2)
        env2 = Environment()
        env2.topology = basic.example_topology()
        env2.flow_factory = UninterruptingFlowFactory()
        nodes2 = list(env2.topology.get_nodes())
        e_node1 = [n for n in nodes2 if 'rpi' in n.name][0].name
        e_node2 = [n for n in nodes2 if 'rpi' in n.name][1].name
        c_storage = [n for n in nodes2 if 'server' in n.name][0].name

        engine_direct = WorkflowEngine(env2, storage_node=c_storage, default_task_duration=task_duration)
        dag_direct = build_dag(DataPassingMode.DIRECT)
        res_direct: list = []

        def run_direct():
            r = yield from engine_direct.execute_workflow(
                dag_direct, task_placements={"prep": e_node1, "train": e_node2, "eval": e_node1}
            )
            res_direct.append(r)

        env2.process(run_direct())
        env2.run()

        # 3. Remote Storage execution (distributed across node1 and node2, data via cloud server)
        env3 = Environment()
        env3.topology = basic.example_topology()
        env3.flow_factory = UninterruptingFlowFactory()
        nodes3 = list(env3.topology.get_nodes())
        e3_node1 = [n for n in nodes3 if 'rpi' in n.name][0].name
        e3_node2 = [n for n in nodes3 if 'rpi' in n.name][1].name
        c3_storage = [n for n in nodes3 if 'server' in n.name][0].name

        engine_remote = WorkflowEngine(env3, storage_node=c3_storage, default_task_duration=task_duration)
        dag_remote = build_dag(DataPassingMode.REMOTE_STORAGE)
        res_remote: list = []

        def run_remote():
            r = yield from engine_remote.execute_workflow(
                dag_remote, task_placements={"prep": e3_node1, "train": e3_node2, "eval": e3_node1}
            )
            res_remote.append(r)

        env3.process(run_remote())
        env3.run()

        r_coloc = res_coloc[0]
        r_direct = res_direct[0]
        r_remote = res_remote[0]

        # Verify all 3 tasks completed in each experiment
        assert len(r_coloc.task_records) == 3
        assert len(r_direct.task_records) == 3
        assert len(r_remote.task_records) == 3

        # Makespan verification:
        # Colocated makespan should be approximately 3 * task_duration (almost 0 transfer time)
        assert r_coloc.makespan < r_direct.makespan
        # Direct passing should have significantly shorter makespan than Remote Storage
        assert r_direct.makespan < r_remote.makespan
        # Remote storage transfer time should be significantly higher due to 2-hop cloud overhead
        assert r_remote.total_data_transfer_time > r_direct.total_data_transfer_time

    def test_diamond_parallel_execution(self):
        r"""
        Tests diamond DAG:
               A (prep)
              /        \
          B (train1)   C (train2)   [Parallel execution]
              \        /
               D (eval)
        """
        env = Environment()
        env.topology = basic.example_topology()
        env.flow_factory = UninterruptingFlowFactory()

        nodes = list(env.topology.get_nodes())
        n1 = [n for n in nodes if 'rpi' in n.name][0].name
        n2 = [n for n in nodes if 'rpi' in n.name][1].name
        n3 = [n for n in nodes if 'nuc' in n.name][0].name

        dag = WorkflowDAG("diamond-dag", default_mode=DataPassingMode.DIRECT)
        dag.add_task(WorkflowTask(name="A", function_name="prep"))
        dag.add_task(WorkflowTask(name="B", function_name="train1"))
        dag.add_task(WorkflowTask(name="C", function_name="train2"))
        dag.add_task(WorkflowTask(name="D", function_name="aggregate"))

        dag.add_dependency("A", "B", size_bytes=1024 * 1024)
        dag.add_dependency("A", "C", size_bytes=1024 * 1024)
        dag.add_dependency("B", "D", size_bytes=512 * 1024)
        dag.add_dependency("C", "D", size_bytes=512 * 1024)

        engine = WorkflowEngine(env, default_task_duration=0.5)
        results = []

        def run_sim():
            res = yield from engine.execute_workflow(
                dag, task_placements={"A": n1, "B": n2, "C": n3, "D": n1}
            )
            results.append(res)

        env.process(run_sim())
        env.run()

        res: WorkflowExecutionResult = results[0]
        assert len(res.task_records) == 4

        # Tasks B and C should execute concurrently (their start times should be nearly identical)
        b_record = res.task_records["B"]
        c_record = res.task_records["C"]
        assert abs(b_record.start_time - c_record.start_time) < 0.2

        # Task D must start strictly after BOTH B and C have finished
        d_record = res.task_records["D"]
        assert d_record.start_time >= b_record.end_time
        assert d_record.start_time >= c_record.end_time


class TestWorkflowSchedulers:

    def test_round_robin_scheduler(self):
        env = Environment()
        env.topology = basic.example_topology()
        nodes = [n.name for n in env.topology.get_nodes() if not n.name.startswith("switch")]

        dag = WorkflowDAG("rr-pipeline")
        dag.add_task(WorkflowTask(name="T1", function_name="f1"))
        dag.add_task(WorkflowTask(name="T2", function_name="f2"))
        dag.add_task(WorkflowTask(name="T3", function_name="f3"))
        dag.add_task(WorkflowTask(name="T4", function_name="f4"))
        dag.add_dependency("T1", "T2")
        dag.add_dependency("T2", "T3")
        dag.add_dependency("T3", "T4")

        scheduler = RoundRobinWorkflowScheduler()
        placements = scheduler.schedule(dag, env)

        assert len(placements) == 4
        assert placements["T1"] == nodes[0]
        assert placements["T2"] == nodes[1 % len(nodes)]
        assert placements["T3"] == nodes[2 % len(nodes)]
        assert placements["T4"] == nodes[3 % len(nodes)]

    def test_data_aware_scheduler_places_coupled_tasks_together(self):
        env = Environment()
        env.topology = basic.example_topology()

        dag = WorkflowDAG("coupled-pipeline", default_mode=DataPassingMode.DIRECT)
        dag.add_task(WorkflowTask(name="extract", function_name="fn1"))
        dag.add_task(WorkflowTask(name="transform", function_name="fn2"))

        # Heavy data dependency between extract and transform (50 MB)
        dag.add_dependency("extract", "transform", size_bytes=50 * 1024 * 1024)

        scheduler = DataAwareWorkflowScheduler()
        placements = scheduler.schedule(dag, env)

        # To avoid moving 50 MB across the network, transform should be placed on the same node as extract
        assert placements["transform"] == placements["extract"]

    def test_workflow_engine_integrates_scheduler(self):
        """
        Tests that WorkflowEngine uses DataAwareWorkflowScheduler by default when no placements are given.
        """
        env = Environment()
        env.topology = basic.example_topology()
        flow_factory = UninterruptingFlowFactory()
        env.flow_factory = flow_factory

        dag = WorkflowDAG("auto-scheduled-dag", default_mode=DataPassingMode.DIRECT)
        dag.add_task(WorkflowTask(name="step1", function_name="fn1"))
        dag.add_task(WorkflowTask(name="step2", function_name="fn2"))
        dag.add_dependency("step1", "step2", size_bytes=10 * 1024 * 1024)

        # Default WorkflowEngine should use DataAwareWorkflowScheduler
        engine = WorkflowEngine(env, default_task_duration=0.2)
        results = []

        def run_sim():
            res = yield from engine.execute_workflow(dag)
            results.append(res)

        env.process(run_sim())
        env.run()

        res = results[0]
        assert len(res.task_records) == 2
        # Tasks step1 and step2 should be colocated by the data-aware scheduler
        assert res.task_records["step1"].node_name == res.task_records["step2"].node_name

    def test_data_aware_beats_round_robin_makespan(self):
        """
        Validates that DataAwareWorkflowScheduler outperforms RoundRobinWorkflowScheduler
        in data transfer overhead and makespan on network-intensive pipelines.
        """
        def run_pipeline(scheduler):
            env = Environment()
            env.topology = basic.example_topology()
            env.flow_factory = UninterruptingFlowFactory()

            dag = WorkflowDAG("comparison-dag", default_mode=DataPassingMode.DIRECT)
            dag.add_task(WorkflowTask(name="T1", function_name="fn1"))
            dag.add_task(WorkflowTask(name="T2", function_name="fn2"))
            dag.add_task(WorkflowTask(name="T3", function_name="fn3"))
            # 20 MB payloads between stages
            dag.add_dependency("T1", "T2", size_bytes=20 * 1024 * 1024)
            dag.add_dependency("T2", "T3", size_bytes=20 * 1024 * 1024)

            engine = WorkflowEngine(env, scheduler=scheduler, default_task_duration=0.1)
            results = []

            def sim():
                res = yield from engine.execute_workflow(dag)
                results.append(res)

            env.process(sim())
            env.run()
            return results[0]

        res_rr = run_pipeline(RoundRobinWorkflowScheduler())
        res_da = run_pipeline(DataAwareWorkflowScheduler())

        # Data-aware co-locates or selects optimal low-latency links, yielding lower transfer time
        assert res_da.total_data_transfer_time < res_rr.total_data_transfer_time
        assert res_da.makespan < res_rr.makespan

    def test_full_faassim_skippy_workflow_integration(self):
        """
        End-to-end integration test with faas-sim's full Simulation, DefaultFaasSystem,
        and WorkflowSkippyScheduler replacing Skippy (`sim.create_scheduler`).
        Verifies:
          1. When nothing is running (`scale_min=0`), triggering a workflow causes
             `WorkflowSkippyScheduler` to schedule pods onto the exact target nodes
             chosen by `DataAwareWorkflowScheduler`.
          2. `DefaultFaasSystem` deploys and executes the replicas on those target nodes.
          3. A second workflow arrival reuses the warm running replicas on those nodes.
        """
        from faas.system import ScalingConfiguration
        from sim.faassim import Simulation
        from sim.benchmark import Benchmark
        from sim.context.platform.deployment.model import SimScalingConfiguration

        topology = basic.example_topology()
        results = []
        replica_counts_after_runs = []

        class WorkflowBenchmark(Benchmark):
            def setup(self, env: Environment):
                basic.ExampleBenchmark().setup(env)

            def run(self, env: Environment):
                # Deploy functions with scale_min=0 (cold start: no pods running initially)
                eb = basic.ExampleBenchmark()
                pi_fd = eb.prepare_python_pi_deployment()
                pi_fd.scaling_config = SimScalingConfiguration(ScalingConfiguration(scale_min=0))
                resnet_fd = eb.prepare_resnet_inference_deployment()
                resnet_fd.scaling_config = SimScalingConfiguration(ScalingConfiguration(scale_min=0))

                yield from env.faas.deploy(pi_fd)
                yield from env.faas.deploy(resnet_fd)

                dag = WorkflowDAG("edge-ai-workflow", default_mode=DataPassingMode.DIRECT)
                dag.add_task(WorkflowTask(name="preprocess", function_name="python-pi"))
                dag.add_task(WorkflowTask(name="inference", function_name="resnet50-inference"))
                dag.add_dependency("preprocess", "inference", size_bytes=25 * 1024 * 1024)

                engine = WorkflowEngine(env)

                # First workflow request (triggers cold-start placement via WorkflowSkippyScheduler)
                res1 = yield from engine.execute_workflow(dag)
                results.append(res1)
                replica_counts_after_runs.append(
                    len(env.faas.get_replicas("python-pi")) + len(env.faas.get_replicas("resnet50-inference"))
                )

                # Second workflow request (reuses warm replicas on the same scheduled nodes)
                res2 = yield from engine.execute_workflow(dag)
                results.append(res2)
                replica_counts_after_runs.append(
                    len(env.faas.get_replicas("python-pi")) + len(env.faas.get_replicas("resnet50-inference"))
                )

        sim = Simulation(topology, WorkflowBenchmark())
        sim.create_scheduler = WorkflowSkippyScheduler.create()
        sim.run()

        assert len(results) == 2
        res1, res2 = results[0], results[1]

        node_prep = res1.task_records["preprocess"].node_name
        node_inf = res1.task_records["inference"].node_name

        # Verify actual faas-sim replicas were placed by WorkflowSkippyScheduler on the exact scheduled nodes
        pi_replicas = sim.env.faas.get_replicas("python-pi")
        resnet_replicas = sim.env.faas.get_replicas("resnet50-inference")
        assert len(pi_replicas) == 1
        assert len(resnet_replicas) == 1
        assert pi_replicas[0].node.name == node_prep
        assert resnet_replicas[0].node.name == node_inf

        # Verify each run isolated its own transfer records (no double-counting across runs)
        assert res1.total_data_bytes_transferred == 25 * 1024 * 1024
        assert res2.total_data_bytes_transferred == 25 * 1024 * 1024

        # Second run reused warm replicas without spawning extra pods, and ran faster (no image pull cold-start)
        assert replica_counts_after_runs[0] == 2
        assert replica_counts_after_runs[1] == 2
        assert res2.makespan < res1.makespan
