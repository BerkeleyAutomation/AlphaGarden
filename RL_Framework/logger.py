import logging
from torch.utils.tensorboard import SummaryWriter
import sys
import colorlog
import numpy as np

root_logger = logging.getLogger()
stdout_hdlr = logging.StreamHandler(sys.stdout)
stdout_frmtr = colorlog.ColoredFormatter(
    '%(processName)s %(purple)s%(name)-20s'
    ' %(log_color)s%(levelname)-8s%(reset)s'
    ' %(white)s%(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
stdout_hdlr.setFormatter(stdout_frmtr)
root_logger.addHandler(stdout_hdlr)
root_logger.setLevel(logging.INFO)

np.set_printoptions(suppress=True)


native_logger = logging.getLogger(__file__)

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir

        msg = '# Logging summaries to {}...'.format(log_dir)
        native_logger.info('#'*len(msg))
        native_logger.info(msg)
        native_logger.info('#'*len(msg))
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase),
                                      scalar_dict,
                                      step)

    def flush(self):
        self._summ_writer.flush()
