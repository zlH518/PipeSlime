import copy

from slime.utils.arguments import parse_args
from pipeengine import PipeEngine

NUM_TASKS=2

if __name__ == "__main__":
    args = parse_args()
    tasks_args = []
    tasks_args.append(args)
    tasks_args = [copy.deepcopy(args) for _ in NUM_TASKS]
    pipeEngine = PipeEngine(tasks_args)
    pipeEngine.run()
