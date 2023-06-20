DEBUG=True
if DEBUG:
    from .petr3d_debug import Petr3D
else:
    from .petr3d import Petr3D
from .petr3d_pth import Petr3DPTH
from .petr3d_onnx import Petr3DONNX