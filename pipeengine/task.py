import os
import ray
import asyncio

from slime.ray.placement_group import create_actor_group, create_rollout_group


@ray.remote
class Task:
    def __init__(self, task_id, args, pg, actor_pg_reordered_bundle_indices):
        self.task_id = task_id
        self.args = args
        # allocate the GPUs
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
        self.pgs = {
            "actor": (pg, actor_pg_reordered_bundle_indices),
            "rollout": (pg, actor_pg_reordered_bundle_indices[:rollout_offset])
        }

        # create the actor model, with the megatron backend
        self.actor_model = create_actor_group(self.args, self.pgs["actor"])

        # create the rollout generator, with sglang engines inside.
        self.rollout_generator = create_rollout_group(self.args, self.pgs["rollout"])

        # calculate num_rollout from num_epoch
        self.num_rollout_per_epoch = None
        if self.args.num_rollout is None:
            self.num_rollout_per_epoch = self.rollout_generator.data_buffer.get_num_rollout_per_epoch.remote()
            self.args.num_rollout = self.num_rollout_per_epoch * self.args.num_epoch
        assert self.args.num_rollout > 0

        # sync the initialization (model initalization, load checkpoint, etc.)
        self.start_rollout_ids = ray.get(
            self.actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
        )
        assert len(set(self.start_rollout_ids)) == 1
        if self.args.start_rollout_id is None:
            self.args.start_rollout_id = self.start_rollout_ids[0]

        if self.args.rollout_global_dataset:
            ray.get(self.rollout_generator.data_buffer.load.remote(self.args.start_rollout_id - 1))

        # initialize the connection for weight update during training
        ray.get(self.actor_model.async_init_weight_update_connections(self.rollout_generator))

        if args.offload:
            ray.get(self.rollout_generator.async_onload())

        # always update weight first so that sglang has the loaded weights from training.
        ray.get(self.actor_model.async_update_weights())

    async def run(self):
        # train loop.
        # note that for async training, one can change the position of the sync operation(ray.get).
        # for rollout_id in range(self.args.start_rollout_id, self.args.num_rollout):
        #     if self.args.eval_interval is not None and rollout_id == 0:
        #         await self.rollout_generator.async_generate(rollout_id, evaluation=True)
        #         await asyncio.gather(*(self.actor_model.async_eval(rollout_id)))

        #     await self.rollout_generator.async_generate(rollout_id)

        #     if self.args.offload:
        #         await asyncio.gather(*(self.rollout_generator.async_offload()))

        #     await asyncio.gather(*(self.actor_model.async_train(rollout_id)))

        #     if self.args.save_interval is not None and (
        #         (rollout_id + 1) % self.args.save_interval == 0
        #         or (self.num_rollout_per_epoch is not None and (rollout_id + 1) % self.num_rollout_per_epoch == 0)
        #     ):
        #         await asyncio.gather(*(self.actor_model.async_save_model(rollout_id)))
        #         if self.args.rollout_global_dataset:
        #             await self.rollout_generator.data_buffer.save.remote(rollout_id)

        #     if self.args.offload:
        #         await asyncio.gather(*(self.actor_model.async_offload()))
        #         await asyncio.gether(*(self.rollout_generator.async_onload()))

        #     await asyncio.gather(*((self.actor_model.async_update_weights())))

        #     if self.args.eval_interval is not None and (
        #         (rollout_id + 1) % self.args.eval_interval == 0
        #         or (self.num_rollout_per_epoch is not None and (rollout_id + 1) % self.num_rollout_per_epoch == 0)
        #     ):
        #         await self.rollout_generator.async_generate(rollout_id, evaluation=True)
        #         await asyncio.gather(*(self.actor_model.async_eval(rollout_id)))
        print(f"task_id: {self.task_id} starting.....")
        await asyncio.sleep(3)
        print(f"task_id: {self.task_id} ending.....")

