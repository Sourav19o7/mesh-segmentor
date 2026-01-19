"""
Split labeled mesh into named connected components.

Naming convention:
- Metal_01, Metal_02, ... (ordered by volume, largest first)
- Gem_01, Gem_02, ... (ordered by volume, largest first)

Each connected component of the same class gets a unique index.
"""

import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentedComponent:
    """A single segmented mesh component."""

    name: str  # e.g., "Metal_01", "Gem_02"
    class_id: int  # 0=background, 1=metal, 2=gem
    class_name: str  # "background", "metal", "gem"
    mesh: trimesh.Trimesh  # The component mesh
    volume: float  # Component volume
    face_count: int  # Number of faces
    original_face_indices: np.ndarray  # Indices in original mesh


class ComponentSplitter:
    """
    Split a labeled mesh into named connected components.

    Algorithm:
    1. Group faces by label
    2. Find connected components within each label group
    3. Sort components by volume (largest first)
    4. Assign names: Metal_01, Metal_02, Gem_01, etc.

    Example:
        splitter = ComponentSplitter()
        components = splitter.split(mesh, face_labels)
        for comp in components:
            print(f"{comp.name}: {comp.volume:.2f} mm³")
    """

    # Class configuration
    CLASS_NAMES = {0: "background", 1: "metal", 2: "gem"}
    CLASS_PREFIXES = {0: "Background", 1: "Metal", 2: "Gem"}

    def __init__(
        self,
        min_volume_ratio: float = 0.001,
        include_background: bool = False,
        zero_pad: int = 2,
    ):
        """
        Initialize component splitter.

        Args:
            min_volume_ratio: Minimum component volume as fraction of total
            include_background: Whether to include background components
            zero_pad: Zero-padding for index (2 = Metal_01, 3 = Metal_001)
        """
        self.min_volume_ratio = min_volume_ratio
        self.include_background = include_background
        self.zero_pad = zero_pad

    def split(
        self,
        mesh: trimesh.Trimesh,
        face_labels: np.ndarray,
    ) -> List[SegmentedComponent]:
        """
        Split mesh into named components.

        Args:
            mesh: Input mesh
            face_labels: (num_faces,) array of class labels

        Returns:
            List of SegmentedComponent objects
        """
        logger.info(f"Splitting mesh with {len(mesh.faces)} faces")

        # Validate inputs
        if len(face_labels) != len(mesh.faces):
            raise ValueError(
                f"face_labels length ({len(face_labels)}) doesn't match "
                f"mesh faces ({len(mesh.faces)})"
            )

        # Calculate total volume for filtering
        total_volume = abs(mesh.volume) if mesh.is_watertight else mesh.area
        min_volume = total_volume * self.min_volume_ratio

        components = []

        # Process each class
        for class_id, class_name in self.CLASS_NAMES.items():
            # Skip background if not included
            if class_id == 0 and not self.include_background:
                continue

            # Get faces with this label
            class_face_mask = face_labels == class_id
            if not class_face_mask.any():
                continue

            # Find connected components within this class
            class_components = self._find_connected_components(
                mesh, class_face_mask
            )

            # Filter by minimum volume and sort by volume (descending)
            class_components = [
                (faces, vol)
                for faces, vol in class_components
                if vol >= min_volume
            ]
            class_components.sort(key=lambda x: x[1], reverse=True)

            # Create named components
            prefix = self.CLASS_PREFIXES[class_id]
            for idx, (face_indices, volume) in enumerate(class_components, 1):
                # Generate name with zero-padded index
                name = f"{prefix}_{idx:0{self.zero_pad}d}"

                # Extract submesh
                submesh = self._extract_submesh(mesh, face_indices)

                component = SegmentedComponent(
                    name=name,
                    class_id=class_id,
                    class_name=class_name,
                    mesh=submesh,
                    volume=volume,
                    face_count=len(face_indices),
                    original_face_indices=face_indices,
                )
                components.append(component)

                logger.info(
                    f"  {name}: {len(face_indices)} faces, volume={volume:.4f}"
                )

        logger.info(f"Split into {len(components)} components")
        return components

    def _find_connected_components(
        self,
        mesh: trimesh.Trimesh,
        face_mask: np.ndarray,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Find connected components in masked faces.

        Uses face adjacency to determine connectivity.

        Args:
            mesh: Full mesh
            face_mask: Boolean mask of faces to consider

        Returns:
            List of (face_indices, volume) tuples
        """
        face_indices = np.where(face_mask)[0]

        if len(face_indices) == 0:
            return []

        # Build adjacency for masked faces
        # Create a mapping from original face index to local index
        face_to_local = {f: i for i, f in enumerate(face_indices)}
        num_masked = len(face_indices)

        # Build local adjacency matrix
        adj_matrix = np.zeros((num_masked, num_masked), dtype=bool)

        for i, j in mesh.face_adjacency:
            if i in face_to_local and j in face_to_local:
                local_i = face_to_local[i]
                local_j = face_to_local[j]
                adj_matrix[local_i, local_j] = True
                adj_matrix[local_j, local_i] = True

        # Find connected components using label propagation
        num_components, labels = self._connected_components(adj_matrix)

        # Group faces by component
        components = []
        for comp_id in range(num_components):
            local_indices = np.where(labels == comp_id)[0]
            original_indices = face_indices[local_indices]

            # Calculate component volume
            volume = self._calculate_component_volume(mesh, original_indices)

            components.append((original_indices, volume))

        return components

    def _connected_components(
        self,
        adjacency: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """
        Find connected components in adjacency matrix.

        Args:
            adjacency: (N, N) boolean adjacency matrix

        Returns:
            Tuple of (num_components, labels array)
        """
        n = len(adjacency)
        labels = np.full(n, -1, dtype=np.int32)
        current_label = 0

        for start in range(n):
            if labels[start] >= 0:
                continue

            # BFS from this node
            queue = [start]
            labels[start] = current_label

            while queue:
                node = queue.pop(0)
                neighbors = np.where(adjacency[node])[0]

                for neighbor in neighbors:
                    if labels[neighbor] < 0:
                        labels[neighbor] = current_label
                        queue.append(neighbor)

            current_label += 1

        return current_label, labels

    def _calculate_component_volume(
        self,
        mesh: trimesh.Trimesh,
        face_indices: np.ndarray,
    ) -> float:
        """Calculate volume of a component (or area if not watertight)."""
        # Extract submesh
        submesh = self._extract_submesh(mesh, face_indices)

        # Try to get volume, fall back to area
        if submesh.is_watertight:
            return abs(submesh.volume)
        else:
            return submesh.area

    def _extract_submesh(
        self,
        mesh: trimesh.Trimesh,
        face_indices: np.ndarray,
    ) -> trimesh.Trimesh:
        """Extract a submesh containing only specified faces."""
        # Get faces
        faces = mesh.faces[face_indices]

        # Get unique vertices
        unique_vertices = np.unique(faces.flatten())
        vertex_map = {old: new for new, old in enumerate(unique_vertices)}

        # Remap faces
        new_faces = np.array(
            [[vertex_map[v] for v in face] for face in faces]
        )

        # Get vertices
        new_vertices = mesh.vertices[unique_vertices]

        # Create submesh
        submesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False,
        )

        # Copy vertex colors if present
        if mesh.visual.vertex_colors is not None:
            submesh.visual.vertex_colors = mesh.visual.vertex_colors[
                unique_vertices
            ]

        return submesh


def merge_small_components(
    components: List[SegmentedComponent],
    min_volume_ratio: float = 0.01,
) -> List[SegmentedComponent]:
    """
    Merge small components into larger neighbors of same class.

    Args:
        components: List of components
        min_volume_ratio: Minimum volume ratio to keep separate

    Returns:
        Filtered list of components
    """
    # Group by class
    by_class: Dict[int, List[SegmentedComponent]] = {}
    for comp in components:
        if comp.class_id not in by_class:
            by_class[comp.class_id] = []
        by_class[comp.class_id].append(comp)

    result = []
    for class_id, class_components in by_class.items():
        if len(class_components) == 0:
            continue

        # Calculate total volume for this class
        total_volume = sum(c.volume for c in class_components)
        min_volume = total_volume * min_volume_ratio

        # Keep components above threshold
        kept = [c for c in class_components if c.volume >= min_volume]

        # Re-index kept components
        for idx, comp in enumerate(kept, 1):
            comp.name = f"{ComponentSplitter.CLASS_PREFIXES[class_id]}_{idx:02d}"

        result.extend(kept)

    return result
