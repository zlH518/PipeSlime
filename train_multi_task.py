import copy
import asyncio

from slime.utils.arguments import parse_args
from pipeengine import PipeEngine

NUM_TASKS=2

async def main():
    args = parse_args()
    tasks_args = []
    tasks_args.append(args)
    tasks_args = [copy.deepcopy(args) for _ in range(NUM_TASKS)]
    pipeEngine = PipeEngine(tasks_args)
    pipeEngine.run()


if __name__ == "__main__":
    asyncio.run(main())

