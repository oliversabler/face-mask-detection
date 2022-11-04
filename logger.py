"""
Handles logging
"""
class EpochLogger:
    """
    Logger
    """
    def __init__(self, name, epoch, iteration=0):
        self.name = name
        self.epoch = epoch
        self.iteration = iteration
        self.metrics = {}

    def __str__(self):
        msg = f'{self.name} | Epoch [{self.epoch}] Iteration [{self.iteration}]'

        for key, value in sorted(self.metrics.items()):
            if isinstance(value, float):
                msg += f' {key}: {value:.4}'
            else:
                msg += f' {key}: {value}'

        return msg

    def increment(self):
        """
        Increment self
        """
        self.iteration += 1

    def update(self, **kwargs):
        """
        Update metrics
        """
        for key, value in kwargs.items():
            self.metrics[key] = value