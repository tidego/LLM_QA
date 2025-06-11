import logging
import inspect
from pathlib import Path
from config.Config import Base_DIR


class Logger:
    _initialized_loggers = {}  # 记录已初始化的 logger 名，防止重复配置
    _first_log_dir = None      # 只保留第一次传入的 log_dir

    def __init__(self, log_dir: Path = Base_DIR / "logs"):
        """
        初始化 Logger 实例，日志文件名为调用 Logger 的外部模块名（不含扩展名）
        :param log_dir: 日志目录（仅第一次设置有效）
        """

        # 如果是首次设置 log_dir，则记录下来，之后忽略新传入的值
        if Logger._first_log_dir is None:
            Logger._first_log_dir = log_dir
        log_dir = Logger._first_log_dir

        # 确保日志目录存在
        log_dir.mkdir(parents=True, exist_ok=True)

        # 获取调用 Logger 的模块文件名
        caller_frame = inspect.stack()[1]
        caller_path = Path(caller_frame.filename)
        log_name = caller_path.stem  # 不带 .py 的文件名
        log_file = log_dir / f"{log_name}.log"

        # 创建或获取 Logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # 防止重复添加 Handler
        if log_name not in Logger._initialized_loggers:
            # 文件输出
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)

            # 控制台输出
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 日志格式
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            # 标记为已初始化
            Logger._initialized_loggers[log_name] = True

    def __getattr__(self, name):
        """
        将 logger 的方法代理暴露出去（如 logger.info）
        """
        return getattr(self.logger, name)
