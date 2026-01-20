"""
Geometric heuristics for classifying mesh components as Metal or Gem.

Uses shape properties like convexity, compactness, and relative volume
to classify components when ML predictions are unreliable.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Optional
from dataclasses import dataclass
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GeometricFeatures:
    """Geometric features extracted from a mesh component."""
    volume: float
    surface_area: float
    convexity: float  # volume / convex_hull_volume
    compactness: float  # volume / bounding_sphere_volume
    aspect_ratio: float  # max_extent / min_extent
    face_count: int
    vertex_count: int
    is_watertight: bool


def extract_features(mesh: trimesh.Trimesh) -> GeometricFeatures:
    """Extract geometric features from a mesh."""
    # Basic properties
    volume = abs(mesh.volume) if mesh.is_watertight else mesh.convex_hull.volume * 0.8
    surface_area = mesh.area
    face_count = len(mesh.faces)
    vertex_count = len(mesh.vertices)
    is_watertight = mesh.is_watertight

    # Convexity: ratio of volume to convex hull volume
    try:
        convex_hull = mesh.convex_hull
        convex_volume = convex_hull.volume
        convexity = volume / convex_volume if convex_volume > 0 else 0
    except Exception:
        convexity = 0.5  # Default if convex hull fails

    # Compactness: how sphere-like the shape is
    # Sphere has compactness = 1, more complex shapes have lower
    extents = mesh.extents
    bounding_radius = np.max(extents) / 2
    bounding_sphere_volume = (4/3) * np.pi * (bounding_radius ** 3)
    compactness = volume / bounding_sphere_volume if bounding_sphere_volume > 0 else 0

    # Aspect ratio
    sorted_extents = np.sort(extents)
    aspect_ratio = sorted_extents[-1] / sorted_extents[0] if sorted_extents[0] > 0 else 1

    return GeometricFeatures(
        volume=volume,
        surface_area=surface_area,
        convexity=convexity,
        compactness=compactness,
        aspect_ratio=aspect_ratio,
        face_count=face_count,
        vertex_count=vertex_count,
        is_watertight=is_watertight,
    )


def classify_component(
    features: GeometricFeatures,
    total_volume: float,
    all_features: List[GeometricFeatures],
) -> Tuple[int, float]:
    """
    Classify a component as Metal (1) or Gem (2) based on geometric features.

    Heuristics:
    - Gems tend to be: small, highly convex, compact (sphere/faceted), isolated
    - Metal tends to be: larger, less convex (bands, settings), connected

    Args:
        features: Features of this component
        total_volume: Total volume of all components
        all_features: Features of all components for relative comparison

    Returns:
        (class_id, confidence) where class_id is 1 (metal) or 2 (gem)
    """
    volume_ratio = features.volume / total_volume if total_volume > 0 else 0

    # Score for being a gem (higher = more likely gem)
    gem_score = 0.0

    # 1. Size: Gems are typically small relative to the total piece
    if volume_ratio < 0.05:
        gem_score += 0.3  # Small component
    elif volume_ratio < 0.15:
        gem_score += 0.15
    elif volume_ratio > 0.4:
        gem_score -= 0.4  # Very large = likely metal

    # 2. Convexity: Gems are typically highly convex (faceted stones)
    if features.convexity > 0.85:
        gem_score += 0.35  # Very convex = likely gem
    elif features.convexity > 0.7:
        gem_score += 0.2
    elif features.convexity < 0.5:
        gem_score -= 0.3  # Low convexity = likely metal (bands, prongs)

    # 3. Compactness: Gems tend to be more compact (sphere-like or cube-like)
    if features.compactness > 0.3:
        gem_score += 0.2
    elif features.compactness < 0.1:
        gem_score -= 0.2  # Elongated/thin = likely metal

    # 4. Aspect ratio: Gems are typically more equidimensional
    if features.aspect_ratio < 2.0:
        gem_score += 0.15  # Roughly equal dimensions = gem-like
    elif features.aspect_ratio > 5.0:
        gem_score -= 0.25  # Very elongated = likely metal (band, prong)

    # 5. If this is the largest component by far, it's likely metal
    volumes = [f.volume for f in all_features]
    if len(volumes) > 1:
        max_volume = max(volumes)
        if features.volume == max_volume:
            second_max = sorted(volumes)[-2] if len(volumes) > 1 else 0
            if features.volume > 2 * second_max:
                gem_score -= 0.3  # Dominant component = metal

    # Decision threshold
    if gem_score > 0.2:
        class_id = 2  # Gem
        confidence = min(0.5 + gem_score, 0.95)
    else:
        class_id = 1  # Metal
        confidence = min(0.5 - gem_score, 0.95)

    return class_id, confidence


def reclassify_components(components: List) -> List:
    """
    Reclassify components using geometric heuristics.

    Args:
        components: List of SegmentedComponent objects

    Returns:
        Components with updated class_id, class_name, and name
    """
    from inference.component_splitter import SegmentedComponent, ComponentSplitter

    if not components:
        return components

    # Extract features for all components
    all_features = []
    for comp in components:
        features = extract_features(comp.mesh)
        all_features.append(features)

    total_volume = sum(f.volume for f in all_features)

    # Reclassify each component
    metal_components = []
    gem_components = []

    for comp, features in zip(components, all_features):
        class_id, confidence = classify_component(features, total_volume, all_features)

        # Update component
        comp.class_id = class_id
        comp.class_name = "metal" if class_id == 1 else "gem"

        if class_id == 1:
            metal_components.append((comp, features.volume))
        else:
            gem_components.append((comp, features.volume))

        logger.debug(
            f"Component reclassified: convexity={features.convexity:.3f}, "
            f"compactness={features.compactness:.3f}, vol_ratio={features.volume/total_volume:.3f} "
            f"-> {comp.class_name} (conf={confidence:.2f})"
        )

    # Sort by volume and rename
    metal_components.sort(key=lambda x: x[1], reverse=True)
    gem_components.sort(key=lambda x: x[1], reverse=True)

    for idx, (comp, _) in enumerate(metal_components, 1):
        comp.name = f"Metal {idx:02d}"

    for idx, (comp, _) in enumerate(gem_components, 1):
        comp.name = f"Gem {idx:02d}"

    # Combine and sort by name
    result = [c for c, _ in metal_components] + [c for c, _ in gem_components]

    logger.info(f"Reclassified: {len(metal_components)} metal, {len(gem_components)} gem components")

    return result
