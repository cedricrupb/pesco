
class Verifier:

    def __init__(self, verifier_name, runtime = 1):
        self.verifier_name = verifier_name
        self.runtime = runtime


class SequentialComposition:

    def __init__(self, verifiers):
        self._verifiers = verifiers


class AlgorithmSelection:

    def __init__(self, task_to_verifier):
        self.task_to_verifier = task_to_verifier

# Benchmark ----------------------------------------------------------------