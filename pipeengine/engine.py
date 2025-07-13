import ray
import socket
import asyncio

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .task import Task


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)



class PipeEngine:
    def __init__(self, tasks_args):
        self.args = tasks_args[0]
        self.tasks_args = tasks_args[1:]
        self.num_tasks = len(self.tasks_args)

        self._pipeRLHF_validate_args()

        # get all gpus info
        num_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node + self.args.rollout_num_gpus
        self.pg, self.actor_pg_reordered_bundle_indices = self._create_placement_groups(num_gpus)
        self.tasks_init()


    def tasks_init(self):
        self.task_group = []
        for task_id in range(self.num_tasks):
            self.task_group.append(
                Task(
                    task_id,
                    self.tasks_args[task_id],
                    self.pg,
                    self.actor_pg_reordered_bundle_indices
                )
            )
        ray.get([task.init() for task in self.task_group])
            


    def _pipeRLHF_validate_args(self):
        # TODO: check pipeengine if there have any problem
        pass

    def _create_placement_groups(self, num_gpus):
        """Create a placement group with the specified number of GPUs."""
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
        pg = placement_group(bundles, strategy="PACK")
        num_bundles = len(bundles)

        ray.get(pg.ready())
        # use info actor to get the GPU id
        info_actors = []
        for i in range(num_bundles):
            info_actors.append(
                InfoActor.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                    )
                ).remote()
            )
        gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
        for actor in info_actors:
            ray.kill(actor)

        bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
        pg_reordered_bundle_indices = [bundle_info[0] for bundle_info in sorted(bundle_infos, key=sort_key)]
        for i in range(num_bundles):
            actual_bundle_index = pg_reordered_bundle_indices[i]
            print(
                f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
                f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
            )

        return pg, pg_reordered_bundle_indices
    

    async def run(self):
        await asyncio.gather(*[task.run() for task in self.task_group])
