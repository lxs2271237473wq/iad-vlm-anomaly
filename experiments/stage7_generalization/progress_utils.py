import shutil
import sys
import time

try:
    from lightning.pytorch.callbacks import Callback
except Exception:
    from pytorch_lightning.callbacks import Callback


def _format_seconds(seconds):
    seconds = max(0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(sec):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


class OneLineProgressCallback(Callback):
    def __init__(self, category="unknown", refresh_interval=0.5):
        super().__init__()
        self.category = category
        self.refresh_interval = refresh_interval
        self.stage = ""
        self.total = 0
        self.start_time = None
        self.last_update = 0.0

    def _get_width(self):
        return max(80, shutil.get_terminal_size((120, 20)).columns)

    def _clear_line(self):
        width = self._get_width()
        sys.stderr.write("\r" + " " * (width - 1) + "\r")
        sys.stderr.flush()

    def _print(self, done, total, force=False):
        now = time.time()
        if not force and (now - self.last_update) < self.refresh_interval:
            return

        self.last_update = now

        if self.start_time is None:
            self.start_time = now

        elapsed = now - self.start_time
        done = max(0, int(done))
        total = max(1, int(total))

        sec_per_batch = elapsed / max(done, 1)
        remaining = max(total - done, 0)
        eta = remaining * sec_per_batch
        percent = 100.0 * done / total

        msg = (
            f"[{self.category}] {self.stage} "
            f"{done}/{total} ({percent:5.1f}%) | "
            f"{sec_per_batch:.2f}s/batch | "
            f"elapsed {_format_seconds(elapsed)} | "
            f"ETA {_format_seconds(eta)}"
        )

        width = self._get_width()
        msg = msg[: width - 1]
        sys.stderr.write("\r" + msg.ljust(width - 1))
        sys.stderr.flush()

    def _start(self, stage, total):
        self.stage = stage
        self.total = max(1, int(total))
        self.start_time = time.time()
        self.last_update = 0.0
        self._clear_line()
        self._print(0, self.total, force=True)

    def _finish(self):
        if self.total:
            self._print(self.total, self.total, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def on_train_epoch_start(self, trainer, pl_module):
        total = getattr(trainer, "num_training_batches", 0)
        if isinstance(total, (list, tuple)):
            total = sum(total)
        if not total:
            total = 1
        self._start("fit", total)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._print(batch_idx + 1, self.total)

    def on_train_epoch_end(self, trainer, pl_module):
        self._finish()

    def on_predict_start(self, trainer, pl_module):
        total = getattr(trainer, "num_predict_batches", 0)
        if isinstance(total, (list, tuple)):
            total = sum(total)
        if not total:
            total = 1
        self._start("predict", total)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._print(batch_idx + 1, self.total)

    def on_predict_end(self, trainer, pl_module):
        self._finish()
