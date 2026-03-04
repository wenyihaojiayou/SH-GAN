import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

__all__ = ["Logger", "get_logger"]


class Logger:
    def __init__(
        self,
        log_dir: str,
        exp_name: Optional[str] = None,
        use_tensorboard: bool = True,
        is_main_process: bool = True
    ):
        self.is_main_process = is_main_process
        self.log_dir = log_dir
        self.exp_name = exp_name if exp_name is not None else f"SH-GAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join(log_dir, self.exp_name)

        if self.is_main_process:
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(os.path.join(self.exp_dir, "logs"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_dir, "tensorboard"), exist_ok=True)

        # File logger setup
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.is_main_process and not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # File handler
            log_file = os.path.join(self.exp_dir, "logs", "train.log")
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)

        # TensorBoard writer
        self.writer = None
        if use_tensorboard and SummaryWriter is not None and self.is_main_process:
            self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))

    def info(self, msg: str):
        if self.is_main_process:
            self.logger.info(msg)

    def warn(self, msg: str):
        if self.is_main_process:
            self.logger.warning(msg)

    def error(self, msg: str):
        if self.is_main_process:
            self.logger.error(msg)

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None and self.is_main_process:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self.writer is not None and self.is_main_process:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag: str, img_tensor, step: int, dataformats: str = "CHW"):
        if self.writer is not None and self.is_main_process:
            self.writer.add_image(tag, img_tensor, step, dataformats=dataformats)

    def log_config(self, config: Dict[str, Any]):
        if self.is_main_process:
            self.info("=" * 50 + " Config " + "=" * 50)
            for k, v in config.items():
                if isinstance(v, dict):
                    self.info(f"{k}:")
                    for sub_k, sub_v in v.items():
                        self.info(f"  {sub_k}: {sub_v}")
                else:
                    self.info(f"{k}: {v}")
            self.info("=" * 108)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()


def get_logger(
    log_dir: str = "./logs",
    exp_name: Optional[str] = None,
    use_tensorboard: bool = True,
    is_main_process: bool = True
) -> Logger:
    return Logger(log_dir, exp_name, use_tensorboard, is_main_process)


# Runtime verification
if __name__ == "__main__":
    test_logger = get_logger(log_dir="./test_logs", exp_name="test_run", use_tensorboard=False)
    test_logger.info("This is an info message")
    test_logger.warn("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.log_config({"model": "SH-GAN", "dataset": "Places2", "batch_size": 16})
    test_logger.close()
    print("Logger test completed.")
