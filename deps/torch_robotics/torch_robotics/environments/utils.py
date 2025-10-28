import numpy as np
from torch_robotics.environments.primitives import MultiSphereField, ObjectField, MultiBoxField


def circle_overlaps_box(center, r, box_center, box_size, gap=0.0):
    """
    center: (2,) circle center
    r:      float circle radius
    box_center: (2,) box center
    box_size:   (2,) box (w,h)
    gap:    safety clearance

    Returns True if circle intersects (or is within gap of) the AABB.
    """
    cx, cy = float(center[0]), float(center[1])
    bx, by = float(box_center[0]), float(box_center[1])
    hw, hh = float(box_size[0]) * 0.5, float(box_size[1]) * 0.5

    # distance from circle center to rectangle center along each axis
    dx = abs(cx - bx) - hw
    dy = abs(cy - by) - hh

    # if center projects inside the box on both axes, it's overlapping
    if dx <= 0.0 and dy <= 0.0:
        return True

    # clamp to rectangle edge and measure corner distance
    dx_clamped = max(dx, 0.0)
    dy_clamped = max(dy, 0.0)
    return (dx_clamped * dx_clamped + dy_clamped * dy_clamped) < (r + gap) ** 2


def circle_overlaps_any_box(center, r, box_centers, box_sizes, gap=0.0):
    if box_centers is None or len(box_centers) == 0:
        return False
    for bc, bs in zip(box_centers, box_sizes):
        if circle_overlaps_box(center, r, bc, bs, gap=gap):
            return True
    return False


def sample_non_overlapping_boxes_2d(
        n_boxes: int,
        min_size: float = 0.12,
        max_size: float = 0.28,
        margin: float = 0.15,
        gap: float = 0.02,
        max_global_attempts: int = 50_000,
        map_size=1,  # assume it is a square
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
    lo = -map_size + margin
    hi = map_size - margin

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


def sample_non_overlapping_spheres_2d(
        n_spheres: int,
        r_min: float = 0.06,
        r_max: float = 0.12,
        margin: float = 0.10,
        gap: float = 0.02,
        max_global_attempts: int = 100_000,
        rng: np.random.Generator | None = None,
        # NEW: avoid existing boxes
        avoid_box_centers: np.ndarray | None = None,
        avoid_box_sizes: np.ndarray | None = None,
        avoid_box_gap: float = 0.0,
        map_size = 1,
):
    """
    Random non-overlapping circles in [-1,1]^2 w/ margin and *no overlap with boxes*.
    Returns centers (N,2), radii (N,). Raises if not feasible within attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    radii = rng.uniform(low=r_min, high=r_max, size=(n_spheres,))
    order = np.argsort(-radii)  # place larger first
    radii_sorted = radii[order]

    lo = -map_size + margin
    hi = map_size - margin

    centers_sorted = np.zeros((n_spheres, 2), dtype=float)
    placed = 0
    attempts = 0

    def fits_pairwise(cent, k):
        if k == 0:
            return True
        cj = centers_sorted[:k]
        rj = radii_sorted[:k]
        d2 = np.sum((cj - cent[None, :]) ** 2, axis=1)
        req = (rj + radii_sorted[k] + gap)
        if np.any(d2 < (req * req)):
            return False
        return True

    while placed < n_spheres and attempts < max_global_attempts:
        r = radii_sorted[placed]
        x = rng.uniform(low=lo + r, high=hi - r)
        y = rng.uniform(low=lo + r, high=hi - r)
        cand = np.array([x, y], dtype=float)

        # 1) no collision w/ already-placed spheres
        if not fits_pairwise(cand, placed):
            attempts += 1
            continue

        # 2) no collision w/ any boxes (with its own gap)
        if circle_overlaps_any_box(cand, r, avoid_box_centers, avoid_box_sizes, gap=avoid_box_gap):
            attempts += 1
            continue

        centers_sorted[placed] = cand
        placed += 1
        attempts += 1

    if placed < n_spheres:
        raise RuntimeError(
            f"Failed to place {n_spheres} non-overlapping spheres (and avoiding boxes). "
            f"Placed {placed}. Try fewer spheres, smaller radii, smaller gaps/margins, "
            f"or increase max_global_attempts."
        )

    centers = np.zeros_like(centers_sorted)
    radii_out = np.zeros_like(radii_sorted)
    centers[order] = centers_sorted
    radii_out[order] = radii_sorted
    return centers, radii_out


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
