import threading

import mlflow
from corax.utils.loggers import base
from mlflow.entities import Metric
from mlflow.utils.time import get_current_time_millis


class _Logger(base.Logger):
    def __init__(
        self,
        client: mlflow.MlflowClient,
        run_id: str,
        steps_key: str | None,
        label: str | None,
    ):
        self._client = client
        self._run_id = run_id
        self._steps_key = steps_key
        self._label = label
        self._auto_step_num = 0
        # TODO: This is resulting in duplicate metrics with the `learner/` prefix.
        # not sure why.

    def write(self, data: base.LoggingData) -> None:
        if self._steps_key:
            step = data[self._steps_key]
        else:
            step = self._auto_step_num
            self._auto_step_num += 1
        timestamp = get_current_time_millis()
        metrics = []
        for k, v in data.items():
            if k == self._steps_key:
                continue
            if self._label:
                k = f"{self._label}/{k}"
            metrics.append(Metric(k, float(v), timestamp, step))
        try:
            self._client.log_batch(self._run_id, metrics=metrics)
        except TypeError:
            breakpoint()
            raise

    def close(self) -> None:
        self._client.set_terminated(self._run_id)


# TODO: this doesn't work with multi-process
_RUN_ID = None
_RUN_ID_LOCK = threading.Lock()


def make_factory(exp_name: str) -> base.LoggerFactory:
    global _RUN_ID, _RUN_ID_LOCK
    with _RUN_ID_LOCK:
        if _RUN_ID is None:
            mlflow.set_tracking_uri("databricks")
            mlflow.set_experiment(f"/Shared/{exp_name}")
            run = mlflow.start_run()
            _RUN_ID = run.info.run_id

    def _factory(label, steps_key=None, instance=None):
        assert _RUN_ID
        return _Logger(mlflow.MlflowClient(), _RUN_ID, steps_key, label)

    return _factory
