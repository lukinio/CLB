class LinearScheduler:
    """
    class represent Linear Scheduler y = a * x
    """
    def __init__(self, start, end=None, coefficient=None):
        self.start = start
        self.end = end
        self.coefficient = coefficient
        self.current = start
        self.iteration = 0
        self.warm = 0

    def step(self):
        assert self.coefficient is not None, "coefficient is None"

        if self.warm:
            self.iteration += 1
            if self.iteration < self.warm:
                return self.current

        if self.iteration == self.warm and self.warm:
            print(f"=============== End of warm epochs =============")

        if self.end is None:
            self.current += self.coefficient
        else:
            if abs(self.current - self.end) > 1e-8:
                self.current += self.coefficient
            else:
                self.current = self.end

        return self.current

    def calc_coefficient(self, param_val, epoch, iter_on_epoch):
        self.coefficient = param_val / (epoch * iter_on_epoch)

    def warm_epoch(self, epoch, iter_on_epoch):
        self.iteration = 0
        self.warm = epoch * iter_on_epoch

    def set_end(self, end):
        self.end = None if end == 0 else end


class StepScheduler:
    def __init__(self, steps=None, coefficients=None, iter_on_batch=None):
        self.steps = steps
        self.coefficients = coefficients
        self.idx = 0
        self.iteration = 0
        self.current = None
        self.iter_on_batch = iter_on_batch

    def step(self, apply_fn=None, **kwargs):
        assert isinstance(self.steps, list) and isinstance(self.coefficients, list) and isinstance(self.iter_on_batch, int), "Missing input"
        if self.idx < len(self.steps) and self.iteration == self.steps[self.idx] * self.iter_on_batch:
            self.current = self.coefficients[self.idx]
            self.idx += 1
            if apply_fn is not None:
                apply_fn(**kwargs)
            print(f"{20 * '='} coefficient {self.current} {20 * '='}")
        self.iteration += 1
        return self.current


if __name__ == '__main__':
    ls = StepScheduler(steps=[0, 2], coefficients=[1, 0], iter_on_batch=5)
    for i in range(5 * 5):
        print(f"iter {i} - {ls.step()}")
    print(f"current: {ls.current}")
