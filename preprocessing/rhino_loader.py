"""
Load .3dm (Rhino) files and extract mesh geometry.

Uses rhino3dm library to parse Rhino files and extract:
- Mesh objects
- BREP objects (converted to mesh)
- Object names and layers for potential label inference
"""

import rhino3dm
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedMesh:
    """Container for extracted mesh data from Rhino file."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) or (M, 4) int64 - triangles or quads
    name: Optional[str] = None
    layer_name: Optional[str] = None
    object_id: Optional[str] = None

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def is_triangulated(self) -> bool:
        return self.faces.shape[1] == 3


class RhinoLoader:
    """
    Load and extract mesh geometry from .3dm files.

    The loader handles:
    1. Mesh objects - directly extract vertices and faces
    2. BREP objects - convert to mesh using Rhino's meshing
    3. Extrusion objects - convert to mesh

    Example:
        loader = RhinoLoader()
        meshes = loader.load("jewelry.3dm")
        for mesh in meshes:
            print(f"Mesh: {mesh.name}, {mesh.num_vertices} vertices")
    """

    def __init__(
        self,
        mesh_quality: int = 2,  # 0=low, 1=medium, 2=high
        triangulate: bool = True,
    ):
        """
        Initialize Rhino loader.

        Args:
            mesh_quality: Mesh quality for BREP conversion (0-2)
            triangulate: Convert quads to triangles
        """
        self.mesh_quality = mesh_quality
        self.triangulate = triangulate

        # Mesh parameters for BREP conversion
        self._mesh_params = self._create_mesh_params(mesh_quality)

    def _create_mesh_params(
        self, quality: int
    ) -> rhino3dm.MeshingParameters:
        """Create meshing parameters based on quality level."""
        params = rhino3dm.MeshingParameters()

        if quality == 0:  # Low
            params.RelativeTolerance = 0.5
            params.MinimumEdgeLength = 0.1
        elif quality == 1:  # Medium
            params.RelativeTolerance = 0.2
            params.MinimumEdgeLength = 0.05
        else:  # High
            params.RelativeTolerance = 0.1
            params.MinimumEdgeLength = 0.01

        return params

    def load(
        self,
        file_path: Union[str, Path],
    ) -> List[ExtractedMesh]:
        """
        Load a .3dm file and extract all mesh geometry.

        Args:
            file_path: Path to .3dm file

        Returns:
            List of ExtractedMesh objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".3dm":
            raise ValueError(f"Expected .3dm file, got: {file_path.suffix}")

        logger.info(f"Loading Rhino file: {file_path}")

        try:
            model = rhino3dm.File3dm.Read(str(file_path))
        except Exception as e:
            raise ValueError(f"Failed to parse Rhino file: {e}")

        if model is None:
            raise ValueError(f"Failed to load Rhino file: {file_path}")

        # Build layer name lookup
        layer_names = self._build_layer_lookup(model)

        # Extract meshes from all objects
        meshes = []
        for obj in model.Objects:
            extracted = self._extract_mesh_from_object(
                obj, layer_names, model
            )
            if extracted is not None:
                meshes.append(extracted)

        logger.info(
            f"Extracted {len(meshes)} meshes from {file_path.name}"
        )

        return meshes

    def load_from_bytes(
        self,
        data: bytes,
    ) -> List[ExtractedMesh]:
        """
        Load a .3dm file from bytes.

        Args:
            data: Raw bytes of .3dm file

        Returns:
            List of ExtractedMesh objects
        """
        logger.info(f"Loading Rhino file from bytes ({len(data)} bytes)")

        try:
            model = rhino3dm.File3dm.FromByteArray(data)
        except Exception as e:
            raise ValueError(f"Failed to parse Rhino file from bytes: {e}")

        if model is None:
            raise ValueError("Failed to load Rhino file from bytes")

        layer_names = self._build_layer_lookup(model)

        meshes = []
        for obj in model.Objects:
            extracted = self._extract_mesh_from_object(
                obj, layer_names, model
            )
            if extracted is not None:
                meshes.append(extracted)

        logger.info(f"Extracted {len(meshes)} meshes from bytes")

        return meshes

    def _build_layer_lookup(
        self, model: rhino3dm.File3dm
    ) -> Dict[int, str]:
        """Build layer index to name mapping."""
        layer_names = {}
        for i, layer in enumerate(model.Layers):
            layer_names[i] = layer.Name
        return layer_names

    def _extract_mesh_from_object(
        self,
        obj: rhino3dm.File3dmObject,
        layer_names: Dict[int, str],
        model: rhino3dm.File3dm,
    ) -> Optional[ExtractedMesh]:
        """Extract mesh from a Rhino object."""
        geometry = obj.Geometry

        if geometry is None:
            return None

        # Get object metadata
        name = obj.Attributes.Name if obj.Attributes.Name else None
        layer_idx = obj.Attributes.LayerIndex
        layer_name = layer_names.get(layer_idx, None)
        object_id = str(obj.Attributes.Id)

        # Handle different geometry types
        mesh = None

        if isinstance(geometry, rhino3dm.Mesh):
            mesh = self._extract_from_mesh(geometry)

        elif isinstance(geometry, rhino3dm.Brep):
            mesh = self._extract_from_brep(geometry)

        elif isinstance(geometry, rhino3dm.Extrusion):
            mesh = self._extract_from_extrusion(geometry)

        elif isinstance(geometry, rhino3dm.SubD):
            mesh = self._extract_from_subd(geometry)

        if mesh is None:
            return None

        vertices, faces = mesh

        # Validate
        if len(vertices) < 3 or len(faces) < 1:
            logger.debug(
                f"Skipping degenerate mesh: {len(vertices)} verts, {len(faces)} faces"
            )
            return None

        # Triangulate if needed
        if self.triangulate and faces.shape[1] == 4:
            faces = self._triangulate_quads(faces)

        return ExtractedMesh(
            vertices=vertices,
            faces=faces,
            name=name,
            layer_name=layer_name,
            object_id=object_id,
        )

    def _extract_from_mesh(
        self, mesh: rhino3dm.Mesh
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract vertices and faces from a Rhino Mesh."""
        # Get vertices
        vertices = []
        for v in mesh.Vertices:
            vertices.append([v.X, v.Y, v.Z])
        vertices = np.array(vertices, dtype=np.float64)

        # Get faces
        faces = []
        for f in mesh.Faces:
            if f.IsQuad:
                faces.append([f.A, f.B, f.C, f.D])
            else:
                faces.append([f.A, f.B, f.C, f.C])  # Pad to 4 for consistency

        if len(faces) == 0:
            return None

        faces = np.array(faces, dtype=np.int64)

        # Convert to triangles if all are triangles (C == D)
        if np.all(faces[:, 2] == faces[:, 3]):
            faces = faces[:, :3]

        return vertices, faces

    def _extract_from_brep(
        self, brep: rhino3dm.Brep
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convert BREP to mesh and extract geometry."""
        try:
            # Get mesh faces from BREP
            meshes = brep.Faces

            all_vertices = []
            all_faces = []
            vertex_offset = 0

            for face in meshes:
                mesh = face.GetMesh(rhino3dm.MeshType.Any)
                if mesh is None:
                    continue

                # Extract vertices
                for v in mesh.Vertices:
                    all_vertices.append([v.X, v.Y, v.Z])

                # Extract faces with offset
                for f in mesh.Faces:
                    if f.IsQuad:
                        all_faces.append([
                            f.A + vertex_offset,
                            f.B + vertex_offset,
                            f.C + vertex_offset,
                            f.D + vertex_offset,
                        ])
                    else:
                        all_faces.append([
                            f.A + vertex_offset,
                            f.B + vertex_offset,
                            f.C + vertex_offset,
                            f.C + vertex_offset,
                        ])

                vertex_offset = len(all_vertices)

            if len(all_vertices) == 0 or len(all_faces) == 0:
                return None

            vertices = np.array(all_vertices, dtype=np.float64)
            faces = np.array(all_faces, dtype=np.int64)

            # Convert to triangles if all are triangles
            if np.all(faces[:, 2] == faces[:, 3]):
                faces = faces[:, :3]

            return vertices, faces

        except Exception as e:
            logger.warning(f"Failed to extract mesh from BREP: {e}")
            return None

    def _extract_from_extrusion(
        self, extrusion: rhino3dm.Extrusion
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convert Extrusion to mesh and extract geometry."""
        try:
            mesh = extrusion.GetMesh(rhino3dm.MeshType.Any)
            if mesh is None:
                return None
            return self._extract_from_mesh(mesh)
        except Exception as e:
            logger.warning(f"Failed to extract mesh from Extrusion: {e}")
            return None

    def _extract_from_subd(
        self, subd: rhino3dm.SubD
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convert SubD to mesh and extract geometry."""
        try:
            # SubD needs to be converted to mesh
            # rhino3dm may not fully support this, try to get render mesh
            mesh = subd.ToMesh(rhino3dm.SubDDisplayParameters())
            if mesh is None:
                return None
            return self._extract_from_mesh(mesh)
        except Exception as e:
            logger.warning(f"Failed to extract mesh from SubD: {e}")
            return None

    def _triangulate_quads(self, faces: np.ndarray) -> np.ndarray:
        """Convert quad faces to triangles."""
        triangles = []
        for face in faces:
            if face[2] == face[3]:
                # Already a triangle
                triangles.append(face[:3])
            else:
                # Split quad into two triangles
                triangles.append([face[0], face[1], face[2]])
                triangles.append([face[0], face[2], face[3]])

        return np.array(triangles, dtype=np.int64)


def merge_meshes(
    meshes: List[ExtractedMesh],
) -> ExtractedMesh:
    """
    Merge multiple meshes into a single mesh.

    Args:
        meshes: List of ExtractedMesh objects

    Returns:
        Single merged ExtractedMesh
    """
    if len(meshes) == 0:
        raise ValueError("No meshes to merge")

    if len(meshes) == 1:
        return meshes[0]

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for mesh in meshes:
        all_vertices.append(mesh.vertices)
        all_faces.append(mesh.faces + vertex_offset)
        vertex_offset += mesh.num_vertices

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)

    return ExtractedMesh(
        vertices=vertices,
        faces=faces,
        name="merged",
    )
