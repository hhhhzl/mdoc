import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================
# Config
# ============================
DT = 0.08
H = 20  # horizon steps
NS = 64  # MPPI samples
LAMBDA = 1.0  # temperature
SEED = 42
np.random.seed(SEED)

# Court
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
OUTER_WIDTH = 15.0
OUTER_LENGTH = 30.0
NET_Y = COURT_LENGTH / 2.0
NET_THICK = 0.5

# Robot & sensing
R_ROBOT = 0.65
V_MAX = 2.0
W_MAX = 2.0
A_V_MAX = 2.0
A_W_MAX = 4.0
FOV_RADIUS = 5.0

# Dynamic obstacles
DYN_OBS = [
    {"c": np.array([5.0, 10.0]), "r": 0.6, "v": np.array([0.08, 0.1]), "is_human": True},
    {"c": np.array([10.0, 15.0]), "r": 0.6, "v": np.array([-0.05, 0.07])},  # not human
]

# Static obstacles
STAT_OBS = [
    {"c": np.array([OUTER_WIDTH / 2, OUTER_LENGTH - 2.0]), "r": 0.8},
    {"c": np.array([OUTER_WIDTH / 2, 2.0]), "r": 0.8},
    {"c": np.array([2.0, OUTER_LENGTH / 2]), "r": 0.6},
    {"c": np.array([OUTER_WIDTH - 2.0, OUTER_LENGTH / 2]), "r": 0.6},
]


# ============================
# Utils
# ============================
def clamp(v, lo, hi): return np.minimum(np.maximum(v, lo), hi)


def wrap_angle(a):
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return a


def in_bounds(p):
    return (
            R_ROBOT <= p[0] <= OUTER_WIDTH - R_ROBOT and
            R_ROBOT <= p[1] <= OUTER_LENGTH - R_ROBOT
    )


def court_net_params():
    y_off = (OUTER_LENGTH - COURT_LENGTH) / 2.0
    net_y = y_off + NET_Y
    x_min = (OUTER_WIDTH - COURT_WIDTH) / 2.0
    x_max = (OUTER_WIDTH + COURT_WIDTH) / 2.0
    return net_y, x_min, x_max


def collide_point_circles(p, circles):
    for obj in circles:
        if np.linalg.norm(p - obj["c"]) <= (obj["r"] + R_ROBOT):
            return True
    return False


def collide_state(p):
    if not in_bounds(p): return True
    # Net as an obstacle strip (unless later "allow_cross" needed)
    net_y, x_min, x_max = court_net_params()
    if (x_min - R_ROBOT) <= p[0] <= (x_max + R_ROBOT):
        if (net_y - NET_THICK / 2 - R_ROBOT) <= p[1] <= (net_y + NET_THICK / 2 + R_ROBOT):
            return True
    if collide_point_circles(p, STAT_OBS): return True
    if collide_point_circles(p, DYN_OBS):  return True
    return False


def visible_balls(robot_p, balls):
    # Simple FOV: within radius (no occlusion modeling for demo)
    vis_idx = []
    for i, b in enumerate(balls):
        if np.linalg.norm(b - robot_p) <= FOV_RADIUS:
            vis_idx.append(i)
    return vis_idx


# ============================
# Reference path (boundary spline - simple rectangle inner offset polyline)
# ============================
def boundary_polyline(offset=1.5, n_per_side=50):
    x0, x1 = offset, OUTER_WIDTH - offset
    y0, y1 = offset, OUTER_LENGTH - offset
    xs = np.concatenate([
        np.linspace(x0, x1, n_per_side),
        np.full(n_per_side, x1),
        np.linspace(x1, x0, n_per_side),
        np.full(n_per_side, x0)
    ])
    ys = np.concatenate([
        np.full(n_per_side, y0),
        np.linspace(y0, y1, n_per_side),
        np.full(n_per_side, y1),
        np.linspace(y1, y0, n_per_side)
    ])
    path = np.stack([xs, ys], axis=1)
    return path


# ========== Global  ==========
GRID_RES = 0.25
NOV_SIGMA = 0.75
W_NOV = 20.0
visited_grid = None
novelty_field = None


def world_to_grid(p):
    gx = int(np.clip(p[0] / GRID_RES, 0, OUTER_WIDTH / GRID_RES - 1))
    gy = int(np.clip(p[1] / GRID_RES, 0, OUTER_LENGTH / GRID_RES - 1))
    return gx, gy


def ensure_visited_grid():
    global visited_grid
    if visited_grid is None:
        Wg = int(np.ceil(OUTER_WIDTH / GRID_RES))
        Hg = int(np.ceil(OUTER_LENGTH / GRID_RES))
        visited_grid = np.zeros((Hg, Wg), dtype=np.float32)


def splat_visitation(p, w=1.0):
    ensure_visited_grid()
    gx, gy = world_to_grid(p)
    visited_grid[gy, gx] += w


def precompute_novelty_field():
    ensure_visited_grid()
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(visited_grid, sigma=NOV_SIGMA / GRID_RES, mode='nearest')
    nf = np.exp(-smoothed)
    return nf


def novelty_at(p):
    ensure_visited_grid()
    x = np.clip(p[0] / GRID_RES, 0, visited_grid.shape[1] - 1)
    y = np.clip(p[1] / GRID_RES, 0, visited_grid.shape[0] - 1)
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, visited_grid.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, visited_grid.shape[0] - 1)
    fx = x - x0
    fy = y - y0
    v00 = novelty_field[y0, x0]
    v10 = novelty_field[y0, x1]
    v01 = novelty_field[y1, x0]
    v11 = novelty_field[y1, x1]
    return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11


# ============================
# Serpentine (Boustrophedon) coverage path
# ============================
USE_SERPENTINE_REF = True
LANE_SPACING = 1.0
MARGIN = 1.0
SAMPLES_PER_LANE = 60


def serpentine_polyline():
    net_y, x_min, x_max = court_net_params()
    y0 = MARGIN
    y1a = net_y - (NET_THICK / 2 + R_ROBOT + MARGIN)
    y0b = net_y + (NET_THICK / 2 + R_ROBOT + MARGIN)
    y1b = OUTER_LENGTH - MARGIN

    x0 = MARGIN
    x1 = OUTER_WIDTH - MARGIN

    def build_half(y_lo, y_hi, reverse=False):
        if y_hi <= y_lo + 1e-6:
            return []
        lanes = []
        ys = np.arange(y_lo, y_hi, LANE_SPACING)
        if len(ys) == 0 or ys[-1] < y_hi - 0.5 * LANE_SPACING:
            ys = np.append(ys, y_hi)
        for i, y in enumerate(ys):
            xs = np.linspace(x0, x1, SAMPLES_PER_LANE)
            if i % 2 == 1:
                xs = xs[::-1]
            seg = np.stack([xs, np.full_like(xs, y)], axis=1)
            lanes.append(seg)
        poly = np.concatenate(lanes, axis=0) if lanes else np.zeros((0, 2))
        return poly[::-1] if reverse else poly

    lower = build_half(y0, y1a, reverse=False)
    upper = build_half(y0b, y1b, reverse=True)

    if lower.size == 0 and upper.size == 0:
        return boundary_polyline(offset=min(2.0, FOV_RADIUS * 0.9))

    if lower.size == 0:
        return upper
    if upper.size == 0:
        return lower

    connector = np.linspace(lower[-1], upper[0], 20)
    return np.concatenate([lower, connector, upper], axis=0)


# ============================
# Lightweight RRT* for reference path (optional)
# ============================
USE_RRT_REF = False  # RRT* as REF_PATH
RRT_MAX_ITER = 2000
RRT_STEP = 0.6
RRT_NEIGHBOR = 1.5


def collides_segment(p, q):
    n = max(2, int(np.ceil(np.linalg.norm(q - p) / 0.1)))
    for i in range(n + 1):
        s = i / n
        x = p * (1 - s) + q * s
        if collide_state(x):
            return True
    return False


def rrt_star(start, goal, max_iter=RRT_MAX_ITER, step=RRT_STEP, r_neigh=RRT_NEIGHBOR):
    start = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)
    nodes = [start]
    parents = [-1]
    costs = [0.0]

    def nearest(z):
        d = [np.sum((n - z) ** 2) for n in nodes]
        return int(np.argmin(d))

    def steer(a, b, eta=step):
        v = b - a
        d = np.linalg.norm(v)
        if d <= eta:
            return b
        return a + v / d * eta

    for it in range(max_iter):
        # goal bias
        if np.random.rand() < 0.12:
            z = goal
        else:
            z = np.array([
                np.random.uniform(R_ROBOT, OUTER_WIDTH - R_ROBOT),
                np.random.uniform(R_ROBOT, OUTER_LENGTH - R_ROBOT)
            ])
        idx = nearest(z)
        new = steer(nodes[idx], z)
        if collides_segment(nodes[idx], new):
            continue

        # choose parent by rewiring
        parent = idx
        best_cost = costs[idx] + np.linalg.norm(new - nodes[idx])
        # neighbor set
        neigh_idx = [j for j, n in enumerate(nodes) if np.linalg.norm(n - new) <= r_neigh]
        for j in neigh_idx:
            if not collides_segment(nodes[j], new):
                c = costs[j] + np.linalg.norm(new - nodes[j])
                if c < best_cost:
                    best_cost, parent = c, j

        nodes.append(new)
        parents.append(parent)
        costs.append(best_cost)

        # rewire neighbors
        j_new = len(nodes) - 1
        for j in neigh_idx:
            if collides_segment(nodes[j], new):
                continue
            c = costs[j_new] + np.linalg.norm(nodes[j] - new)
            if c + 1e-9 < costs[j]:
                parents[j] = j_new
                costs[j] = c

        # check reach goal
        if not collides_segment(new, goal) and np.linalg.norm(new - goal) <= step:
            nodes.append(goal)
            parents.append(j_new)
            costs.append(costs[j_new] + np.linalg.norm(goal - new))
            # backtrack
            path = []
            cur = len(nodes) - 1
            while cur != -1:
                path.append(nodes[cur])
                cur = parents[cur]
            path = path[::-1]
            # densify to polyline
            poly = [path[0]]
            for i in range(1, len(path)):
                seg = np.linspace(0, 1, 8)
                for s in seg[1:]:
                    poly.append(path[i - 1] * (1 - s) + path[i] * s)
            return np.array(poly)
    return None


REF_PATH = serpentine_polyline() if USE_SERPENTINE_REF else boundary_polyline(offset=min(2.0, FOV_RADIUS * 0.9))


# REF_PATH = boundary_polyline(offset=min(2.0, FOV_RADIUS * 0.9))


def nearest_on_polyline(p, poly):
    d = np.sum((poly - p[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d))
    return poly[idx], idx


# ============================
# Dynamics (unicycle, control u=[a_v, a_w], state x=[x,y,theta,v,w])
# ============================
def f_step(x, u, dt=DT):
    # x: [x,y,theta,v,w]; u: [a_v, a_w]
    a_v = clamp(u[0], -A_V_MAX, A_V_MAX)
    a_w = clamp(u[1], -A_W_MAX, A_W_MAX)

    v = clamp(x[3] + a_v * dt, 0.0, V_MAX)
    w = clamp(x[4] + a_w * dt, -W_MAX, W_MAX)

    th = wrap_angle(x[2] + w * dt)
    x_next = np.array([
        x[0] + v * np.cos(th) * dt,
        x[1] + v * np.sin(th) * dt,
        th, v, w
    ])
    return x_next


# ============================
# CBF / Human helpers (add-on)
# ============================

CBF_ALPHA = 0.5  # alpha in h_{k+1} - (1 - alpha) h_k >= 0
CBF_W = 600 # CBF
CBF_SAFE_MARGIN = 0.05
HUMAN_FRONT_BETA = 1.5
HUMAN_UNCERT_PER_S = 0.10
USE_CBF = False
USE_HUMAN_SOCIAL = True


def circle_dist_margin(p, c, r_total):
    return np.linalg.norm(p - c) - r_total


def cbf_violation(h_k, h_next, alpha=CBF_ALPHA):
    return max(0.0, (1.0 - alpha) * h_k - h_next)


def frontness_weight(p_robot, c_human, v_human):
    to_robot = p_robot - c_human
    nr = np.linalg.norm(to_robot) + 1e-9
    nv = np.linalg.norm(v_human) + 1e-9
    cosang = np.dot(to_robot / nr, v_human / nv)  # >0 表示机器人在人的前方锥形区
    return 1.0 + HUMAN_FRONT_BETA * max(0.0, cosang)


def predict_dyn_obstacles_at(k, dt=DT):
    preds = []
    t = k * dt
    for obj in DYN_OBS:
        c = obj["c"] + obj["v"] * t
        r = obj["r"]
        if obj.get("is_human", False):
            r = r + HUMAN_UNCERT_PER_S * t
        preds.append({"c": c, "r": r, "v": obj["v"], "is_human": obj.get("is_human", False)})
    return preds


# ============================
# Cost terms
# ============================
def cost_terms_traj(X, U, target=None, w_ref=1.0, w_tar=0.0):
    # X: [H+1,5], U: [H,2]
    J = 0.0
    for k in range(X.shape[0]):
        p = X[k, :2]
        pr, idx = nearest_on_polyline(p, REF_PATH)
        e_pos = p - pr
        J += w_ref * (0.5 * np.dot(e_pos, e_pos))
        # heading
        t_dir = REF_PATH[(idx + 1) % len(REF_PATH)] - REF_PATH[idx]
        t_dir = t_dir / (np.linalg.norm(t_dir) + 1e-6)
        e_th = wrap_angle(X[k, 2] - math.atan2(t_dir[1], t_dir[0]))
        J += w_ref * (0.2 * e_th * e_th)

    # -------- attractive --------
    if target is not None:
        for k in range(X.shape[0]):
            p = X[k, :2]
            d = np.linalg.norm(p - target)
            J += w_tar * 0.5 * min(d, 1.5) ** 2
        pH = X[-1, :2]
        vH = X[-1, 3]
        dH = np.linalg.norm(pH - target)
        if dH > 0.3:
            J += w_tar * 8.0 * (dH - 0.3) ** 2
        else:
            J += w_tar * 2.0 * (vH ** 2)

    # -------- smoothness --------
    J += 0.1 * np.sum(U ** 2)

    # -------- CBF --------
    if USE_CBF:
        net_y, x_min, x_max = court_net_params()
        for k in range(X.shape[0] - 1):
            p_k = X[k, :2]
            p_n = X[k + 1, :2]
            if not in_bounds(p_k) or not in_bounds(p_n):
                J += 1e4
            else:
                dmin_k = min(p_k[0] - R_ROBOT,
                             OUTER_WIDTH - R_ROBOT - p_k[0],
                             p_k[1] - R_ROBOT,
                             OUTER_LENGTH - R_ROBOT - p_k[1])
                dmin_n = min(p_n[0] - R_ROBOT,
                             OUTER_WIDTH - R_ROBOT - p_n[0],
                             p_n[1] - R_ROBOT,
                             OUTER_LENGTH - R_ROBOT - p_n[1])
                h_k = dmin_k - CBF_SAFE_MARGIN
                h_n = dmin_n - CBF_SAFE_MARGIN
                v = cbf_violation(h_k, h_n)
                if v > 0:
                    J += CBF_W * v * v

            if (x_min - R_ROBOT) <= p_k[0] <= (x_max + R_ROBOT):
                dnet_k = min(abs(p_k[1] - (net_y - NET_THICK / 2 - R_ROBOT)),
                             abs(p_k[1] - (net_y + NET_THICK / 2 + R_ROBOT)))
                dnet_n = min(abs(p_n[1] - (net_y - NET_THICK / 2 - R_ROBOT)),
                             abs(p_n[1] - (net_y + NET_THICK / 2 + R_ROBOT)))
                h_k = dnet_k - CBF_SAFE_MARGIN
                h_n = dnet_n - CBF_SAFE_MARGIN
                v = cbf_violation(h_k, h_n)
                if v > 0:
                    J += (0.8 * CBF_W) * v * v

            all_obs_now = [{"c": o["c"], "r": o["r"], "is_human": False, "v": np.zeros(2)} for o in STAT_OBS]
            dyn_now = predict_dyn_obstacles_at(k)
            dyn_next = predict_dyn_obstacles_at(k + 1)
            cur_obs = all_obs_now + dyn_now

            for idx_o, o in enumerate(cur_obs):
                c_k, r_k = o["c"], o["r"]
                if idx_o < len(all_obs_now):
                    c_n, r_n = c_k, r_k
                else:
                    c_n, r_n = dyn_next[idx_o - len(all_obs_now)]["c"], dyn_next[idx_o - len(all_obs_now)]["r"]

                d_k = circle_dist_margin(p_k, c_k, r_k + R_ROBOT)
                d_n = circle_dist_margin(p_n, c_n, r_n + R_ROBOT)
                h_k = d_k - CBF_SAFE_MARGIN
                h_n = d_n - CBF_SAFE_MARGIN
                v = cbf_violation(h_k, h_n)
                if v > 0:
                    w = CBF_W
                    if USE_HUMAN_SOCIAL and o.get("is_human", False):
                        w *= frontness_weight(p_k, c_k, o.get("v", np.zeros(2)))
                    J += w * v * v

    for k in range(X.shape[0]):
        p = X[k, :2]
        # net soft barrier
        net_y, x_min, x_max = court_net_params()
        if (x_min - R_ROBOT) <= p[0] <= (x_max + R_ROBOT):
            dnet = min(abs(p[1] - (net_y - NET_THICK / 2 - R_ROBOT)),
                       abs(p[1] - (net_y + NET_THICK / 2 + R_ROBOT)))
            if dnet < 0.5:
                J += 200 * (0.5 - dnet + 1e-6) ** 2

        # static + dynamic (human)
        for obj in STAT_OBS:
            d = np.linalg.norm(p - obj["c"]) - (obj["r"] + R_ROBOT)
            if d < 0.3:
                J += 50 * (0.3 - d + 1e-6) ** 2
        dyn = predict_dyn_obstacles_at(k)
        for obj in dyn:
            d = np.linalg.norm(p - obj["c"]) - (obj["r"] + R_ROBOT)
            if d < 0.3:
                J += 200.0 * (0.3 - d + 1e-6) ** 2

    for k in range(X.shape[0]):
        J -= W_NOV * novelty_at(X[k, :2])

    return J


# ============================
# MPPI planner
# ============================
class LocalMPPI:
    def __init__(self, H, dt, nsamples, lam, u_lo, u_hi, sigma_u):
        self.H = H
        self.dt = dt
        self.N = nsamples
        self.lam = lam
        self.u_lo = np.array(u_lo)
        self.u_hi = np.array(u_hi)
        self.sigma = np.array(sigma_u)
        self.U = np.zeros((H, 2))  # mean control seq

    def command(self, x0, target, w_ref, w_tar):
        eps = np.random.randn(self.N, self.H, 2) * self.sigma
        costs = np.zeros(self.N)
        Xs = np.zeros((self.N, self.H + 1, 5))

        for n in range(self.N):
            u_seq = self.U + eps[n]
            u_seq = clamp(u_seq, self.u_lo, self.u_hi)
            x = np.copy(x0)
            Xs[n, 0] = x
            for k in range(self.H):
                x = f_step(x, u_seq[k], DT)
                Xs[n, k + 1] = x
            costs[n] = cost_terms_traj(Xs[n], u_seq, target=target, w_ref=w_ref, w_tar=w_tar)

        beta = np.min(costs)
        w = np.exp(-(costs - beta) / self.lam + 1e-9)
        w = w / (np.sum(w) + 1e-12)

        dU = np.tensordot(w, eps, axes=(0, 0))
        self.U = clamp(self.U + dU, self.u_lo, self.u_hi)

        # shift for receding horizon
        u0 = np.copy(self.U[0])
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0
        return u0, np.mean(Xs[np.argmax(w)], axis=0)


# Try importing external MPPI if present
def make_planner():
    try:
        from mp_baselines.planners.mppi import MPPI as ExtMPPI
        class Wrapper:
            def __init__(self):
                self.inner = ExtMPPI(
                    horizon=H, num_samples=NS, lambda_=LAMBDA, dt=DT,
                    u_min=np.array([-A_V_MAX, -A_W_MAX]),
                    u_max=np.array([A_V_MAX, A_W_MAX]),
                    sigma=np.array([0.8 * A_V_MAX, 0.8 * A_W_MAX])
                )
                self.U_cache = np.zeros((H, 2))

            def command(self, x0, target, w_ref, w_tar):
                # Define a callback cost for the external MPPI version
                def rollout_cost(u_seq):
                    x = np.copy(x0)
                    X = [x]
                    for k in range(u_seq.shape[0]):
                        x = f_step(x, u_seq[k], DT)
                        X.append(x)
                    X = np.stack(X, axis=0)
                    return cost_terms_traj(X, u_seq, target=target, w_ref=w_ref, w_tar=w_tar)

                u, traj = self.inner.control(rollout_cost)  # API may differ; we provide a fallback below
                if u is None:
                    # fallback to local if API mismatch
                    raise RuntimeError("External MPPI API mismatch")
                return u, traj

        return Wrapper(), True
    except Exception as e:
        # Fallback
        planner = LocalMPPI(
            H=H, dt=DT, nsamples=NS, lam=LAMBDA,
            u_lo=np.array([-A_V_MAX, -A_W_MAX]),
            u_hi=np.array([A_V_MAX, A_W_MAX]),
            sigma_u=np.array([0.7 * A_V_MAX, 0.7 * A_W_MAX])
        )
        return planner, False


# ============================
# Scenario & animation
# ============================
def draw_court(ax):
    # outer
    ax.plot([0, OUTER_WIDTH, OUTER_WIDTH, 0, 0],
            [0, 0, OUTER_LENGTH, OUTER_LENGTH, 0])
    # inner court
    y_off = (OUTER_LENGTH - COURT_LENGTH) / 2.0
    x_min = (OUTER_WIDTH - COURT_WIDTH) / 2.0
    x_max = (OUTER_WIDTH + COURT_WIDTH) / 2.0
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_off, y_off, y_off + COURT_LENGTH, y_off + COURT_LENGTH, y_off])
    # net
    net_y, x_min, x_max = court_net_params()
    ax.plot([x_min, x_max], [net_y, net_y], linestyle='--')


def update_dynamic_obstacles():
    for obj in DYN_OBS:
        obj["c"] = obj["c"] + obj["v"]
        if obj["c"][0] <= 0 or obj["c"][0] >= OUTER_WIDTH:  obj["v"][0] *= -1
        if obj["c"][1] <= 0 or obj["c"][1] >= OUTER_LENGTH: obj["v"][1] *= -1


def make_balls(n=15):
    balls = []
    tries = 0
    while len(balls) < n and tries < 5000:
        x = np.random.uniform(0.6, OUTER_WIDTH - 0.6)
        y = np.random.uniform(0.6, OUTER_LENGTH - 0.6)
        p = np.array([x, y])
        if not collide_state(p):
            balls.append(p)
        tries += 1
    return np.array(balls)


def run_demo(save_path="tennisbuddy_mppi_demo.gif", max_steps=2000):
    balls = make_balls(18)
    x = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # [x,y,th,v,w]
    planner, used_ext = make_planner()

    # figure
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, OUTER_WIDTH)
    ax.set_ylim(0, OUTER_LENGTH)
    ax.set_title("TennisBuddy MPPI demo")

    draw_court(ax)
    # reference path
    ax.plot(REF_PATH[:, 0], REF_PATH[:, 1], linewidth=1)

    # obstacles
    dyn_patches = []
    for o in DYN_OBS:
        circ = plt.Circle(tuple(o["c"]), o["r"], alpha=0.5)
        dyn_patches.append(circ)
        ax.add_patch(circ)
    for o in STAT_OBS:
        circ = plt.Circle(tuple(o["c"]), o["r"], alpha=0.5)
        ax.add_patch(circ)

    # balls
    scat_balls = ax.scatter(balls[:, 0], balls[:, 1], s=30, marker='o')

    # robot and FOV
    robot_dot, = ax.plot([], [], marker='o')
    fov_circle = plt.Circle((x[0], x[1]), FOV_RADIUS, fill=False)
    ax.add_patch(fov_circle)

    # planned horizon line
    plan_line, = ax.plot([], [], linewidth=2)

    # history
    traj_x, traj_y = [], []

    # weight scheduling
    w_ref, w_tar = 5, 3

    def pick_target(x, balls):
        vis = visible_balls(x[:2], balls)
        if len(vis) == 0:
            return None, -1
        dists = [np.linalg.norm(balls[i] - x[:2]) for i in vis]
        idx_local = int(np.argmin(dists))
        return balls[vis[idx_local]], vis[idx_local]

    def animate(frame):
        global novelty_field
        novelty_field = precompute_novelty_field()
        nonlocal x, balls, w_ref, w_tar

        # update dynamic obstacles
        update_dynamic_obstacles()
        for circ, o in zip(dyn_patches, DYN_OBS):
            circ.center = tuple(o["c"])

        # select target if any visible
        target, tidx = pick_target(x, balls)
        if target is not None:
            # weight increase towards target
            w_tar = 0.7 * w_tar + 0.3 * 1.0
        else:
            w_tar = 0.7 * w_tar + 0.3 * 0.0
        w_ref = 1.0 - w_tar

        # MPPI step: sample sequence, get first control and nominal traj (approx)
        u, _ = planner.command(x0=x, target=target, w_ref=w_ref, w_tar=w_tar)
        x = f_step(x, u, DT)

        splat_visitation(x[:2], w=1.0)

        # if capture ball
        if tidx >= 0 and np.linalg.norm(x[:2] - balls[tidx]) <= 0.35:
            balls = np.delete(balls, tidx, axis=0)
            scat_balls.set_offsets(balls)

        # store
        traj_x.append(x[0])
        traj_y.append(x[1])

        # simple rollout of current mean plan for visualization
        # (recompute by propagating last planner.U)
        try:
            U_vis = getattr(planner, "U", np.zeros((H, 2)))
        except Exception:
            U_vis = np.zeros((H, 2))
        xv = np.copy(x)
        pred = [xv[:2].copy()]
        for k in range(H):
            xv = f_step(xv, U_vis[k] if k < len(U_vis) else np.zeros(2), DT)
            pred.append(xv[:2].copy())
        pred = np.stack(pred, axis=0)

        # draw
        robot_dot.set_data([x[0]], [x[1]])
        fov_circle.center = (x[0], x[1])
        plan_line.set_data(pred[:, 0], pred[:, 1])

        return [robot_dot, scat_balls, fov_circle, plan_line] + dyn_patches

    ani = FuncAnimation(fig, animate, frames=max_steps, interval=10, blit=True)
    plt.show()
    # ani = FuncAnimation(fig, animate, frames=max_steps, interval=60, blit=True)
    # out = os.path.abspath(save_path)
    # ani.save(out, writer=PillowWriter(fps=15))
    # return out


if __name__ == "__main__":
    run_demo()
