import numpy as np
from torch_robotics.environments.primitives import MultiSphereField, ObjectField, MultiBoxField


def sample_non_overlapping_boxes_2d(
        n_boxes: int,
        min_size: float = 0.12,
        max_size: float = 0.28,
        margin: float = 0.15,
        gap: float = 0.02,
        max_global_attempts: int = 50_000,
        rng: np.random.Generator | None = None,
):
    """
    Returns:
        centers: (N, 2) float array
        sizes:   (N, 2) float array
    Raises:
        RuntimeError if packing fails within max_global_attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample sizes first
    sizes = rng.uniform(low=min_size, high=max_size, size=(n_boxes, 2))
    # Place larger boxes first (descending by max side)
    order = np.argsort(-np.max(sizes, axis=1))
    sizes_sorted = sizes[order]

    # Workspace bounds
    lo = -1.0 + margin
    hi = 1.0 - margin

    centers_sorted = np.zeros((n_boxes, 2), dtype=float)
    placed = 0
    attempts = 0

    def fits(cent, k):
        """Check non-overlap vs. already placed [0..k-1] with a gap."""
        if k == 0:
            return True
        cj = centers_sorted[:k]  # (k, 2)
        sj = sizes_sorted[:k]  # (k, 2)
        dx = np.abs(cent[0] - cj[:, 0])
        dy = np.abs(cent[1] - cj[:, 1])
        req_x = (sj[:, 0] + sizes_sorted[k, 0]) * 0.5 + gap
        req_y = (sj[:, 1] + sizes_sorted[k, 1]) * 0.5 + gap
        # overlap if dx < req_x AND dy < req_y
        overlap = (dx < req_x) & (dy < req_y)
        return not np.any(overlap)

    while placed < n_boxes and attempts < max_global_attempts:
        w, h = sizes_sorted[placed]
        # centers must keep the box inside [lo, hi] with half-sizes
        x = rng.uniform(low=lo + 0.5 * w, high=hi - 0.5 * w)
        y = rng.uniform(low=lo + 0.5 * h, high=hi - 0.5 * h)
        cand = np.array([x, y], dtype=float)
        if fits(cand, placed):
            centers_sorted[placed] = cand
            placed += 1
        attempts += 1

    if placed < n_boxes:
        raise RuntimeError(
            f"Failed to place {n_boxes} non-overlapping boxes with gap={gap} "
            f"and margin={margin}. Placed {placed}. Try fewer boxes, smaller sizes, "
            f"smaller gap/margin, or increase max_global_attempts."
        )

    # Undo the sort so caller sees per-box sizes in original order
    centers = np.zeros_like(centers_sorted)
    centers[order] = centers_sorted
    return centers, sizes


def create_grid_spheres(rows=5, cols=5, heights=0, radius=0.1, distance_from_border=0.1, tensor_args=None):
    # Generates a grid (rows, cols, heights) of circles
    # if heights = 0, creates a 2d grid, else a 3d grid
    dim = 2 if heights == 0 else 3
    centers_x = np.linspace(-1 + distance_from_border, 1 - distance_from_border, cols)
    centers_y = np.linspace(-1 + distance_from_border, 1 - distance_from_border, rows)
    z_flat = None
    if dim == 3:
        centers_z = np.linspace(-1 + distance_from_border, 1 - distance_from_border, heights)
        X, Y, Z = np.meshgrid(centers_x, centers_y, centers_z)
        z_flat = Z.flatten()
    else:
        X, Y = np.meshgrid(centers_x, centers_y)

    flats = [X.flatten(), Y.flatten()]
    if z_flat:
        flats.append(z_flat)
    centers = np.array(flats).T
    radii = np.ones(flats[0].shape[0]) * radius

    spheres = MultiSphereField(centers, radii, tensor_args=tensor_args)
    obj_field = ObjectField([spheres], 'grid-of-spheres')
    obj_list = [obj_field]
    return obj_list
