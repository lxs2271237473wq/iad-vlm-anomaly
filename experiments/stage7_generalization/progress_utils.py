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


def _safe_total(value):
    if isinstance(value, (list, tuple)):
        value = sum(value)
    try:
        value = int(value)
    except Exception:
        value = 1
    return max(1, value)


class OneLineProgressCallback(Callback):
    def __init__(
        self,
        category="unknown",
        category_index=1,
        total_categories=1,
        refresh_interval=1.0,
        run_start_time=None,
        fit_weight=0.85,
    ):
        super().__init__()
        self.category = category
        self.category_index = max(1, int(category_index))
        self.total_categories = max(1, int(total_categories))
        self.refresh_interval = float(refresh_interval)
        self.run_start_time = run_start_time or time.time()
        self.fit_weight = float(fit_weight)

        self.stage = ""
        self.total = 1
        self.start_time = None
        self.last_update = 0.0
        self.epoch_index = 1
        self.max_epochs = 1

    def _terminal_width(self):
        return max(100, shutil.get_terminal_size((140, 20)).columns)

    def _clear_line(self):
        width = self._terminal_width()
        sys.stderr.write("\r" + " " * (width - 1) + "\r")
        sys.stderr.flush()

    def _category_fraction(self, done, total):
        frac = max(0.0, min(1.0, float(done) / max(float(total), 1.0)))

        if self.stage == "fit":
            epoch_frac = (self.epoch_index - 1 + frac) / max(float(self.max_epochs), 1.0)
            return self.fit_weight * max(0.0, min(1.0, epoch_frac))

        if self.stage == "predict":
            return self.fit_weight + (1.0 - self.fit_weight) * frac

        return frac

    def _overall_eta(self, done, total):
        now = time.time()
        total_elapsed = now - self.run_start_time

        cat_frac = self._category_fraction(done, total)
        overall_done = (self.category_index - 1 + cat_frac) / float(self.total_categories)
        overall_done = max(1e-6, min(1.0, overall_done))

        total_estimated = total_elapsed / overall_done
        return max(0.0, total_estimated - total_elapsed)

    def _print(self, done, total, force=False):
        now = time.time()
        if not force and (now - self.last_update) < self.refresh_interval:
            return

        self.last_update = now

        if self.start_time is None:
            self.start_time = now

        done = max(0, int(done))
        total = _safe_total(total)

        elapsed = now - self.start_time
        sec_per_batch = elapsed / max(done, 1)
        stage_eta = max(total - done, 0) * sec_per_batch
        total_eta = self._overall_eta(done, total)
        percent = 100.0 * done / total

        if self.stage == "fit":
            stage_name = f"fit epoch {self.epoch_index}/{self.max_epochs}"
        else:
            stage_name = self.stage

        msg = (
            f"[{self.category_index}/{self.total_categories} {self.category}] "
            f"{stage_name} {done}/{total} ({percent:5.1f}%) | "
            f"{sec_per_batch:.2f}s/batch | "
            f"stage ETA {_format_seconds(stage_eta)} | "
            f"total ETA {_format_seconds(total_eta)}"
        )

        width = self._terminal_width()
        msg = msg[: width - 1]
        sys.stderr.write("\r" + msg.ljust(width - 1))
        sys.stderr.flush()

    def _start(self, stage, total):
        self.stage = stage
        self.total = _safe_total(total)
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
        self.epoch_index = int(getattr(trainer, "current_epoch", 0)) + 1
        self.max_epochs = int(getattr(trainer, "max_epochs", 1) or 1)

        total = getattr(trainer, "num_training_batches", 1)
        self._start("fit", total)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._print(batch_idx + 1, self.total)

    def on_train_epoch_end(self, trainer, pl_module):
        self._finish()

    def on_predict_start(self, trainer, pl_module):
        total = getattr(trainer, "num_predict_batches", 1)
        self._start("predict", total)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._print(batch_idx + 1, self.total)

    def on_predict_end(self, trainer, pl_module):
        self._finish()
