"""
Convert Rhino mesh data to trimesh format.

Handles:
- Vertex/face array conversion
- Mesh validation and repair
- Normal computation
- Mesh cleaning (remove degenerates, duplicates)
"""

import trimesh
import numpy as np
from typing import List, Optional, Tuple, Union
from preprocessing.rhino_loader import ExtractedMesh
from utils.logging import get_logger

logger = get_logger(__name__)


class MeshConverter:
    """
    Convert extracted Rhino meshes to trimesh format.

    trimesh provides:
    - Robust mesh operations
    - Watertight mesh checking
    - Normal computation
    - Mesh repair utilities

    Example:
        converter = MeshConverter()
        rhino_meshes = loader.load("model.3dm")
        trimesh_meshes = converter.convert_all(rhino_meshes)
        merged = converter.merge(trimesh_meshes)
    """

    def __init__(
        self,
        repair: bool = True,
        remove_degenerates: bool = True,
        merge_vertices: bool = True,
        merge_threshold: float = 1e-8,
    ):
        """
        Initialize mesh converter.

        Args:
            repair: Attempt to repair broken meshes
            remove_degenerates: Remove degenerate faces
            merge_vertices: Merge duplicate vertices
            merge_threshold: Distance threshold for vertex merging
        """
        self.repair = repair
        self.remove_degenerates = remove_degenerates
        self.merge_vertices = merge_vertices
        self.merge_threshold = merge_threshold

    def convert(
        self,
        extracted_mesh: ExtractedMesh,
    ) -> Optional[trimesh.Trimesh]:
        """
        Convert a single ExtractedMesh to trimesh.Trimesh.

        Args:
            extracted_mesh: Mesh data from RhinoLoader

        Returns:
            trimesh.Trimesh or None if conversion fails
        """
        try:
            # Create trimesh from vertices and faces
            mesh = trimesh.Trimesh(
                vertices=extracted_mesh.vertices,
                faces=extracted_mesh.faces,
                process=False,  # Don't process yet
            )

            # Clean and repair
            mesh = self._process_mesh(mesh)

            if mesh is None or len(mesh.faces) == 0:
                logger.warning(
                    f"Mesh '{extracted_mesh.name}' has no valid faces after processing"
                )
                return None

            # Store metadata
            mesh.metadata["name"] = extracted_mesh.name
            mesh.metadata["layer"] = extracted_mesh.layer_name
            mesh.metadata["object_id"] = extracted_mesh.object_id

            logger.debug(
                f"Converted mesh '{extracted_mesh.name}': "
                f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
            )

            return mesh

        except Exception as e:
            logger.error(
                f"Failed to convert mesh '{extracted_mesh.name}': {e}"
            )
            return None

    def convert_all(
        self,
        extracted_meshes: List[ExtractedMesh],
    ) -> List[trimesh.Trimesh]:
        """
        Convert multiple extracted meshes to trimesh format.

        Args:
            extracted_meshes: List of ExtractedMesh objects

        Returns:
            List of valid trimesh.Trimesh objects
        """
        meshes = []
        for extracted in extracted_meshes:
            mesh = self.convert(extracted)
            if mesh is not None:
                meshes.append(mesh)

        logger.info(
            f"Converted {len(meshes)}/{len(extracted_meshes)} meshes"
        )

        return meshes

    def _process_mesh(
        self,
        mesh: trimesh.Trimesh,
    ) -> Optional[trimesh.Trimesh]:
        """Apply cleaning and repair operations to mesh."""
        try:
            # Remove degenerate faces
            if self.remove_degenerates:
                mesh.remove_degenerate_faces()

            # Merge duplicate vertices
            if self.merge_vertices:
                mesh.merge_vertices(merge_tex=True, merge_norm=True)

            # Remove duplicate faces
            mesh.remove_duplicate_faces()

            # Remove unreferenced vertices
            mesh.remove_unreferenced_vertices()

            # Fix face winding for consistent normals
            if self.repair:
                trimesh.repair.fix_winding(mesh)
                trimesh.repair.fix_normals(mesh)

            # Validate
            if not mesh.is_empty and len(mesh.faces) > 0:
                return mesh

            return None

        except Exception as e:
            logger.warning(f"Mesh processing failed: {e}")
            return mesh if not mesh.is_empty else None

    def merge(
        self,
        meshes: List[trimesh.Trimesh],
        keep_metadata: bool = True,
    ) -> trimesh.Trimesh:
        """
        Merge multiple meshes into a single mesh.

        Args:
            meshes: List of trimesh.Trimesh objects
            keep_metadata: Store original mesh boundaries in metadata

        Returns:
            Single merged trimesh.Trimesh
        """
        if len(meshes) == 0:
            raise ValueError("No meshes to merge")

        if len(meshes) == 1:
            return meshes[0]

        # Use trimesh's concatenation
        merged = trimesh.util.concatenate(meshes)

        if keep_metadata:
            # Store face ranges for each original mesh
            face_ranges = []
            vertex_ranges = []
            face_offset = 0
            vertex_offset = 0

            for mesh in meshes:
                face_ranges.append((face_offset, face_offset + len(mesh.faces)))
                vertex_ranges.append(
                    (vertex_offset, vertex_offset + len(mesh.vertices))
                )
                face_offset += len(mesh.faces)
                vertex_offset += len(mesh.vertices)

            merged.metadata["face_ranges"] = face_ranges
            merged.metadata["vertex_ranges"] = vertex_ranges
            merged.metadata["submesh_names"] = [
                m.metadata.get("name") for m in meshes
            ]

        logger.info(
            f"Merged {len(meshes)} meshes: "
            f"{len(merged.vertices)} vertices, {len(merged.faces)} faces"
        )

        return merged

    def validate(
        self,
        mesh: trimesh.Trimesh,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a mesh and return issues found.

        Args:
            mesh: Mesh to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if mesh.is_empty:
            issues.append("Mesh is empty")

        if len(mesh.faces) == 0:
            issues.append("Mesh has no faces")

        if not mesh.is_watertight:
            issues.append("Mesh is not watertight")

        if mesh.body_count > 1:
            issues.append(f"Mesh has {mesh.body_count} disconnected bodies")

        # Check for degenerate faces
        degenerate = trimesh.triangles.degenerate(
            mesh.vertices[mesh.faces]
        )
        if degenerate.any():
            issues.append(f"{degenerate.sum()} degenerate faces")

        # Check face normals
        if not np.all(np.isfinite(mesh.face_normals)):
            issues.append("Some face normals are invalid")

        return len(issues) == 0, issues


def from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    process: bool = True,
) -> trimesh.Trimesh:
    """
    Create a trimesh from numpy arrays.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        process: Whether to process/clean the mesh

    Returns:
        trimesh.Trimesh
    """
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=process,
    )


def compute_face_labels_from_points(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    point_labels: np.ndarray,
) -> np.ndarray:
    """
    Transfer point labels to mesh faces using nearest neighbor.

    For each face, find the closest sampled point and use its label.

    Args:
        mesh: The mesh to label
        points: (N, 3) sampled points
        point_labels: (N,) point labels

    Returns:
        (M,) face labels where M is number of faces
    """
    from scipy.spatial import cKDTree

    # Compute face centers
    face_centers = mesh.triangles_center

    # Build KD-tree of sampled points
    tree = cKDTree(points)

    # Query nearest point for each face center
    _, indices = tree.query(face_centers, k=1)

    # Get labels from nearest points
    face_labels = point_labels[indices]

    return face_labels
