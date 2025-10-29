from pathlib import Path
from typing import Protocol
import numpy as np
import onnx
from onnx import helper, numpy_helper
from .config import settings


class ScheduleLike(Protocol):
    phases: np.ndarray
    pump: np.ndarray
    coupling: np.ndarray


def export_stub(schedule: ScheduleLike, out_path: str | Path = "onnx/controller.onnx") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    phase_tensor = numpy_helper.from_array(schedule.phases.astype("float32"), name="phase_schedule")
    pump_tensor = numpy_helper.from_array(schedule.pump.astype("float32"), name="pump_schedule")
    coupling_tensor = numpy_helper.from_array(
        schedule.coupling.astype("float32"), name="coupling_map"
    )

    g = helper.make_graph(
        nodes=[],
        name="wavecore_nl_controller",
        inputs=[],
        outputs=[
            helper.make_tensor_value_info(
                "phase_schedule", onnx.TensorProto.FLOAT, list(schedule.phases.shape)
            ),
            helper.make_tensor_value_info(
                "pump_schedule", onnx.TensorProto.FLOAT, list(schedule.pump.shape)
            ),
            helper.make_tensor_value_info(
                "coupling_map", onnx.TensorProto.FLOAT, list(schedule.coupling.shape)
            ),
        ],
        initializer=[phase_tensor, pump_tensor, coupling_tensor],
    )
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", settings.onnx_opset)])
    onnx.save(m, out_path.as_posix())
    return out_path
