"""
Export segmented mesh to GLB format.

Creates a GLB file with:
- Each component as a separate mesh node
- Components named: Metal_01, Metal_02, Gem_01, etc.
- Optional material groups by class
"""

import numpy as np
import trimesh
import struct
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, BinaryIO
from io import BytesIO
from inference.component_splitter import SegmentedComponent
from utils.logging import get_logger

logger = get_logger(__name__)


# Material colors for each class
DEFAULT_MATERIALS = {
    "background": {"color": [0.5, 0.5, 0.5, 1.0]},  # Gray
    "metal": {"color": [0.83, 0.69, 0.22, 1.0]},  # Gold
    "gem": {"color": [0.15, 0.65, 0.85, 1.0]},  # Blue
}


class GLBExporter:
    """
    Export segmented components to GLB format.

    GLB is the binary version of glTF 2.0, a standard format
    for 3D models that is widely supported.

    Example:
        exporter = GLBExporter()
        glb_bytes = exporter.export(components)
        with open("output.glb", "wb") as f:
            f.write(glb_bytes)
    """

    def __init__(
        self,
        include_materials: bool = True,
        material_colors: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize GLB exporter.

        Args:
            include_materials: Add materials with colors
            material_colors: Custom material colors by class name
        """
        self.include_materials = include_materials
        self.material_colors = material_colors or DEFAULT_MATERIALS

    def export(
        self,
        components: List[SegmentedComponent],
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Export components to GLB format.

        Args:
            components: List of segmented components
            output_path: Optional file path to save

        Returns:
            GLB file as bytes
        """
        logger.info(f"Exporting {len(components)} components to GLB")

        # Create trimesh Scene
        scene = trimesh.Scene()

        for component in components:
            # Add mesh to scene with component name
            mesh = component.mesh.copy()

            # Set material/color based on class
            if self.include_materials:
                mat_config = self.material_colors.get(
                    component.class_name, DEFAULT_MATERIALS["background"]
                )
                color = mat_config["color"]

                # Set face colors
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh,
                    face_colors=np.array([
                        [int(c * 255) for c in color]
                        for _ in range(len(mesh.faces))
                    ], dtype=np.uint8),
                )

            # Add to scene
            scene.add_geometry(mesh, node_name=component.name)

        # Export to GLB
        glb_bytes = scene.export(file_type="glb")

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(glb_bytes)
            logger.info(f"Saved GLB to {output_path}")

        logger.info(f"GLB export complete: {len(glb_bytes)} bytes")
        return glb_bytes

    def export_with_hierarchy(
        self,
        components: List[SegmentedComponent],
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Export with hierarchical structure (class -> components).

        Creates structure:
        - Scene
          - Metal (empty node)
            - Metal_01
            - Metal_02
          - Gem (empty node)
            - Gem_01
        """
        scene = trimesh.Scene()

        # Group components by class
        by_class: Dict[str, List[SegmentedComponent]] = {}
        for comp in components:
            class_name = comp.class_name.capitalize()
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(comp)

        # Add each class group
        for class_name, class_components in by_class.items():
            for comp in class_components:
                mesh = comp.mesh.copy()

                # Set material
                if self.include_materials:
                    mat_config = self.material_colors.get(
                        comp.class_name, DEFAULT_MATERIALS["background"]
                    )
                    color = mat_config["color"]
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=mesh,
                        face_colors=np.array([
                            [int(c * 255) for c in color]
                            for _ in range(len(mesh.faces))
                        ], dtype=np.uint8),
                    )

                # Add to scene with parent structure
                # Note: trimesh's scene doesn't fully support hierarchy in export
                # but the node names will be preserved
                scene.add_geometry(mesh, node_name=comp.name)

        # Export
        glb_bytes = scene.export(file_type="glb")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(glb_bytes)
            logger.info(f"Saved GLB to {output_path}")

        return glb_bytes


def create_segmented_glb(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    output_path: str,
    min_volume_ratio: float = 0.001,
) -> Tuple[bytes, List[Dict]]:
    """
    Convenience function to create segmented GLB from mesh and labels.

    Args:
        mesh: Input mesh
        face_labels: Per-face class labels
        output_path: Output file path
        min_volume_ratio: Minimum component volume ratio

    Returns:
        Tuple of (glb_bytes, component_info)
    """
    from inference.component_splitter import ComponentSplitter

    # Split into components
    splitter = ComponentSplitter(min_volume_ratio=min_volume_ratio)
    components = splitter.split(mesh, face_labels)

    # Export to GLB
    exporter = GLBExporter(include_materials=True)
    glb_bytes = exporter.export(components, output_path)

    # Return component info
    component_info = [
        {
            "name": c.name,
            "class": c.class_name,
            "volume": c.volume,
            "face_count": c.face_count,
        }
        for c in components
    ]

    return glb_bytes, component_info


class GLBBuilder:
    """
    Low-level GLB builder for more control over output.

    Use this if you need custom materials, textures, or
    other advanced glTF features.
    """

    def __init__(self):
        self.buffers = []
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self._buffer_data = BytesIO()

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        material_idx: Optional[int] = None,
    ) -> int:
        """Add a mesh and return its index."""
        # Add vertex buffer
        vertices = vertices.astype(np.float32)
        vertex_bytes = vertices.tobytes()
        vertex_offset = self._buffer_data.tell()
        self._buffer_data.write(vertex_bytes)

        # Pad to 4-byte alignment
        while self._buffer_data.tell() % 4 != 0:
            self._buffer_data.write(b"\x00")

        # Add vertex buffer view
        vertex_view_idx = len(self.buffer_views)
        self.buffer_views.append({
            "buffer": 0,
            "byteOffset": vertex_offset,
            "byteLength": len(vertex_bytes),
            "target": 34962,  # ARRAY_BUFFER
        })

        # Add vertex accessor
        vertex_acc_idx = len(self.accessors)
        self.accessors.append({
            "bufferView": vertex_view_idx,
            "componentType": 5126,  # FLOAT
            "count": len(vertices),
            "type": "VEC3",
            "min": vertices.min(axis=0).tolist(),
            "max": vertices.max(axis=0).tolist(),
        })

        # Add index buffer
        faces = faces.astype(np.uint32)
        index_bytes = faces.flatten().tobytes()
        index_offset = self._buffer_data.tell()
        self._buffer_data.write(index_bytes)

        while self._buffer_data.tell() % 4 != 0:
            self._buffer_data.write(b"\x00")

        # Add index buffer view
        index_view_idx = len(self.buffer_views)
        self.buffer_views.append({
            "buffer": 0,
            "byteOffset": index_offset,
            "byteLength": len(index_bytes),
            "target": 34963,  # ELEMENT_ARRAY_BUFFER
        })

        # Add index accessor
        index_acc_idx = len(self.accessors)
        self.accessors.append({
            "bufferView": index_view_idx,
            "componentType": 5125,  # UNSIGNED_INT
            "count": len(faces) * 3,
            "type": "SCALAR",
        })

        # Add mesh
        mesh_idx = len(self.meshes)
        primitive = {
            "attributes": {"POSITION": vertex_acc_idx},
            "indices": index_acc_idx,
        }
        if material_idx is not None:
            primitive["material"] = material_idx

        self.meshes.append({
            "name": name,
            "primitives": [primitive],
        })

        # Add node
        node_idx = len(self.nodes)
        self.nodes.append({
            "name": name,
            "mesh": mesh_idx,
        })

        return node_idx

    def add_material(
        self,
        name: str,
        color: List[float],
        metallic: float = 0.0,
        roughness: float = 0.5,
    ) -> int:
        """Add a PBR material and return its index."""
        idx = len(self.materials)
        self.materials.append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": color,
                "metallicFactor": metallic,
                "roughnessFactor": roughness,
            },
        })
        return idx

    def build(self) -> bytes:
        """Build the GLB file."""
        # Finalize buffer
        buffer_data = self._buffer_data.getvalue()
        self.buffers = [{"byteLength": len(buffer_data)}]

        # Build JSON
        gltf = {
            "asset": {"version": "2.0", "generator": "mesh-segmentor"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": self.buffers,
        }

        if self.materials:
            gltf["materials"] = self.materials

        json_bytes = json.dumps(gltf).encode("utf-8")

        # Pad JSON to 4-byte alignment
        while len(json_bytes) % 4 != 0:
            json_bytes += b" "

        # Build GLB
        glb = BytesIO()

        # Header
        glb.write(b"glTF")  # Magic
        glb.write(struct.pack("<I", 2))  # Version
        total_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_data)
        glb.write(struct.pack("<I", total_length))

        # JSON chunk
        glb.write(struct.pack("<I", len(json_bytes)))
        glb.write(b"JSON")
        glb.write(json_bytes)

        # Binary chunk
        glb.write(struct.pack("<I", len(buffer_data)))
        glb.write(b"BIN\x00")
        glb.write(buffer_data)

        return glb.getvalue()
