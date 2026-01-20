# Lazy imports to avoid circular import issues when running as module
def __getattr__(name):
    if name == "Predictor":
        from inference.predictor import Predictor
        return Predictor
    elif name == "MeshSegmenter":
        from inference.mesh_segmenter import MeshSegmenter
        return MeshSegmenter
    elif name == "ComponentSplitter":
        from inference.component_splitter import ComponentSplitter
        return ComponentSplitter
    elif name == "GLBExporter":
        from inference.glb_exporter import GLBExporter
        return GLBExporter
    raise AttributeError(f"module 'inference' has no attribute '{name}'")

__all__ = [
    "Predictor",
    "MeshSegmenter",
    "ComponentSplitter",
    "GLBExporter",
]
