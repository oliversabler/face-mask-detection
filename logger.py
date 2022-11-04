"""
Handles logging
"""
class EpochLogger:
    """
    Logger
    """
    def __init__(self, name, epoch, iteration, iteration_len):
        self.name = name
        self.epoch = epoch + 1
        self.iteration = iteration + 1
        self.iteration_len = iteration_len
        self.metrics = {}

    def __str__(self):
        msg = f'{self.name} | \
Epoch [{self.epoch}] \
Iteration [{self.iteration}/{self.iteration_len}]'

        for key, value in sorted(self.metrics.items()):
            if isinstance(value, float):
                msg += f' {key}: {value:.4}'
            else:
                msg += f' {key}: {value}'

        return msg

    def _increment(self):
        """
        Increment self
        """
        self.iteration += 1

    def log(self):
        """
        Log self
        """
        print(self)
        self._increment()

    def update(self, **kwargs):
        """
        Update metrics
        """
        for key, value in kwargs.items():
            self.metrics[key] = value
