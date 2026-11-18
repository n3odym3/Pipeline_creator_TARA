from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.processing_base import ProcessingBase


class Processor_Template(ProcessingBase):
    """
    A template for building multiprocessing background worker processes.
    """

    # 1. Process Configuration (Class defaults - can also be overridden via kwargs)
    buffer_size = 10
    drop_policy = "drop_new"  # Options: 'drop_new', 'drop_oldest', or 'block'
    daemon = True

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Initialize the processor process.

        Args:
            params: Optional initial parameters dictionary passed to the worker.
            **kwargs: Extra configuration options forwarded to ProcessingBase.
        """
        super().__init__(params=params, **kwargs)

    def _process_data(self, data: Any, params: Dict[str, Any]) -> Any:
        """
        Actual data processing logic executed in the isolated worker process.

        Args:
            data: The incoming data packet from the submission queue.
            params: Dictionary containing current parameters/configuration.

        Returns:
            The processed result to be pushed to the out_queue.
        """
        # Simulated workload
        time.sleep(1.0)

        # Example processing using data and params:
        factor = params.get("multiplier", 1)
        if isinstance(data, (int, float)):
            return data * factor

        return True