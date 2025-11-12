# Multi-agent kinodynamic CBS-style planner with:
# (1) Conflict time-window cropping -> path constraints only on [t_s, t_e]
# (2) Meta-robot merge & joint kinodynamic RRT in combined space
# (3) Animated visualization (GIF) of the final solution
#
# Notes:
# - Robots are discs with same radius; state includes time explicitly.
# - Low-level: RRT for single agent; JointRRT for merged groups (m agents).
# - World has static circular obstacles and one linear moving circular obstacle.
#
# This is a compact research skeleton meant for extension.

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation


# --------------------------
# Geometry / Obstacles
# --------------------------
@dataclass
class StaticCircle:
    center: np.ndarray  # (2,)
    radius: float

@dataclass
class StaticBox:
    center: np.ndarray  # (2,)
    size: np.ndarray    # (2,) width & height


@dataclass
class MovingCircle:
    center0: np.ndarray  # (2,)
    vel: np.ndarray  # (2,)
    radius: float

    def center_at(self, t: float) -> np.ndarray:
        return self.center0 + self.vel * t


@dataclass
class PathObstacle:
    ts: np.ndarray  # (L,)
    pts: np.ndarray  # (L,2)
    radius: float

    def center_at(self, t: float) -> np.ndarray:
        if t <= self.ts[0]:
            return self.pts[0]
        if t >= self.ts[-1]:
            return self.pts[-1]
        k = np.searchsorted(self.ts, t, side="right") - 1
        k = max(0, min(k, len(self.ts) - 2))
        t0, t1 = self.ts[k], self.ts[k + 1]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (1 - a) * self.pts[k] + a * self.pts[k + 1]


@dataclass
class World:
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    static_obstacles: List[StaticCircle]
    moving_obstacles: List[MovingCircle]
    robot_radius: float = 0.25

# --------------------------
# Trees / RRT parameters
# --------------------------
@dataclass
class Node:
    pos: np.ndarray  # (d,) or concatenated (2*m,)
    t: float
    parent: Optional[int]


@dataclass
class Tree:
    nodes: List[Node]

    def add(self, node: Node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def nearest_index(self, point: np.ndarray) -> int:
        pts = np.array([n.pos for n in self.nodes])
        d2 = np.sum((pts - point) ** 2, axis=1)
        return int(np.argmin(d2))


# @dataclass
# class RRTParams:
#     dt: float = 0.1
#     v_max: float = 0.04
#     K_min: int = 6
#     K_max: int = 16
#     n_iter: int = 7000
#     goal_radius: float = 0.02
#     prob_goal_bias: float = 0.2
#     dt_check: float = 0.03
#     t_max: float = 60.0

@dataclass
class RRTParams:
    dt: float = 0.12
    v_max: float = 0.06
    K_min: int = 4
    K_max: int = 12
    n_iter: int = 20000
    goal_radius: float = 0.02
    prob_goal_bias: float = 0.35
    dt_check: float = 0.05
    t_max: float = 100


# --------------------------
# Utility
# --------------------------
def interp_path(pos: np.ndarray, ts: np.ndarray, t: float) -> np.ndarray:
    if t <= ts[0]: return pos[0]
    if t >= ts[-1]: return pos[-1]
    k = np.searchsorted(ts, t, side="right") - 1
    k = max(0, min(k, len(ts) - 2))
    t0, t1 = ts[k], ts[k + 1]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return (1 - a) * pos[k] + a * pos[k + 1]


def slice_polyline_by_time(pos: np.ndarray, ts: np.ndarray, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return a path clipped to [t0,t1], including interpolated endpoints."""
    t0c = max(t0, float(ts[0]))
    t1c = min(t1, float(ts[-1]))
    if t1c < t0c:
        # no overlap; return degenerate 2-point segment at nearest boundary
        p = interp_path(pos, ts, (t0 + t1) / 2.0)
        return np.stack([p, p], axis=0), np.array([t0, t1])
    # collect sample knots inside (including new endpoints)
    ks = [t0c]
    for t in ts[1:-1]:
        if t0c < t < t1c:
            ks.append(float(t))
    ks.append(t1c)
    ks = np.array(ks, dtype=float)
    pts = np.stack([interp_path(pos, ts, t) for t in ks], axis=0)
    return pts, ks


# --------------------------
# Single-agent validity
# --------------------------
def is_valid_segment_single(
        pts: np.ndarray, ts: np.ndarray, world: World, dt_check: float,
        path_obstacles: List[PathObstacle]
) -> bool:
    assert pts.shape[0] == ts.shape[0] and pts.shape[1] == 2
    t0, t1 = float(ts[0]), float(ts[-1])
    if t1 < t0 or dt_check <= 0: return False
    T = np.arange(t0, t1 + 1e-9, dt_check)
    if T[-1] < t1: T = np.append(T, t1)

    seg_starts = ts[:-1]
    seg_durs = ts[1:] - ts[:-1]
    seg_durs[seg_durs == 0.0] = 1e-9

    for tau in T:
        if tau >= ts[-1]:
            k = len(ts) - 2
            a = 1.0
        else:
            k = np.searchsorted(ts, tau, side="right") - 1
            k = max(0, min(k, len(seg_starts) - 1))
            a = float((tau - seg_starts[k]) / seg_durs[k])
            a = float(np.clip(a, 0.0, 1.0))
        p = (1 - a) * pts[k] + a * pts[k + 1]

        if not (world.bounds[0][0] <= p[0] <= world.bounds[0][1] and
                world.bounds[1][0] <= p[1] <= world.bounds[1][1]):
            return False
        for obs in world.static_obstacles:
            if isinstance(obs, StaticCircle):
                if np.linalg.norm(p - obs.center) <= (obs.radius + world.robot_radius):
                    return False
            elif isinstance(obs, StaticBox):
                dx = abs(p[0] - obs.center[0])
                dy = abs(p[1] - obs.center[1])
                if dx <= (obs.size[0] / 2 + world.robot_radius) and dy <= (obs.size[1] / 2 + world.robot_radius):
                    return False
        for mobs in world.moving_obstacles:
            c = mobs.center_at(tau)
            if np.linalg.norm(p - c) <= (mobs.radius + world.robot_radius):
                return False
        for pob in path_obstacles:
            c = pob.center_at(tau)
            if np.linalg.norm(p - c) <= (pob.radius + world.robot_radius):
                return False
    return True


# --------------------------
# Joint (meta-robot) validity
# --------------------------
def is_valid_segment_multi(
        pts_seq: np.ndarray,  # (L, m, 2)
        ts: np.ndarray,  # (L,)
        world: World,
        dt_check: float,
        path_obstacles: List[PathObstacle]
) -> bool:
    L, m, _ = pts_seq.shape
    assert ts.shape[0] == L
    t0, t1 = float(ts[0]), float(ts[-1])
    if t1 < t0 or dt_check <= 0: return False
    T = np.arange(t0, t1 + 1e-9, dt_check)
    if T[-1] < t1: T = np.append(T, t1)

    seg_starts = ts[:-1]
    seg_durs = ts[1:] - ts[:-1]
    seg_durs[seg_durs == 0.0] = 1e-9

    for tau in T:
        if tau >= ts[-1]:
            k = len(ts) - 2
            a = 1.0
        else:
            k = np.searchsorted(ts, tau, side="right") - 1
            k = max(0, min(k, len(seg_starts) - 1))
            a = float((tau - seg_starts[k]) / seg_durs[k])
            a = float(np.clip(a, 0.0, 1.0))

        P = (1 - a) * pts_seq[k] + a * pts_seq[k + 1]  # (m,2)
        # Bounds & static/moving obstacles
        for r in range(m):
            p = P[r]
            if not (world.bounds[0][0] <= p[0] <= world.bounds[0][1] and
                    world.bounds[1][0] <= p[1] <= world.bounds[1][1]):
                return False
            for obs in world.static_obstacles:
                if isinstance(obs, StaticCircle):
                    if np.linalg.norm(p - obs.center) <= (obs.radius + world.robot_radius):
                        return False
                elif isinstance(obs, StaticBox):
                    dx = abs(p[0] - obs.center[0])
                    dy = abs(p[1] - obs.center[1])
                    if dx <= (obs.size[0] / 2 + world.robot_radius) and dy <= (obs.size[1] / 2 + world.robot_radius):
                        return False
            for mobs in world.moving_obstacles:
                c = mobs.center_at(tau)
                if np.linalg.norm(p - c) <= (mobs.radius + world.robot_radius):
                    return False
            for pob in path_obstacles:
                c = pob.center_at(tau)
                if np.linalg.norm(p - c) <= (pob.radius + world.robot_radius):
                    return False
        # Intra-group separation
        for a_i in range(m):
            for b_i in range(a_i + 1, m):
                if np.linalg.norm(P[a_i] - P[b_i]) <= 2 * world.robot_radius:
                    return False
    return True


# --------------------------
# RRT: single agent
# --------------------------
def steer_and_propagate_single(p_from: np.ndarray, t_from: float, p_to_hint: np.ndarray, params: RRTParams):
    d = p_to_hint - p_from
    n = np.linalg.norm(d)
    heading = np.array([1.0, 0.0]) if n < 1e-9 else d / n
    theta = random.uniform(-math.pi / 6, math.pi / 6)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    heading = (R @ heading.reshape(2, 1)).ravel()
    if np.linalg.norm(heading) < 1e-9:
        heading = np.array([1.0, 0.0])
    v = params.v_max
    u = heading * v
    K = random.randint(params.K_min, params.K_max)
    pts = [p_from.copy()]
    ts = [t_from]
    p = p_from.copy()
    t = t_from
    for _ in range(K):
        p = p + u * params.dt
        t = t + params.dt
        pts.append(p.copy())
        ts.append(t)
    return np.array(pts), np.array(ts)


def rrt_plan_single(
        start: np.ndarray,
        goal: np.ndarray,
        world: World,
        params: RRTParams,
        path_constraints: List[PathObstacle],
        seed: int = 0
):
    random.seed(seed)
    np.random.seed(seed)
    tree = Tree(nodes=[Node(pos=start.copy(), t=0.0, parent=None)])

    if not is_valid_segment_single(np.array([start, start]), np.array([0.0, 1e-3]), world, params.dt_check,
                                   path_constraints):
        return None, None, tree

    for _ in range(params.n_iter):
        p_rand = goal.copy() if random.random() < params.prob_goal_bias else np.array([
            random.uniform(world.bounds[0][0], world.bounds[0][1]),
            random.uniform(world.bounds[1][0], world.bounds[1][1])
        ])
        idx = tree.nearest_index(p_rand)
        node = tree.nodes[idx]
        pts, ts = steer_and_propagate_single(node.pos, node.t, p_rand, params)
        if ts[-1] > params.t_max:
            continue
        if not is_valid_segment_single(pts, ts, world, params.dt_check, path_constraints):
            continue
        new_node = Node(pos=pts[-1], t=float(ts[-1]), parent=idx)
        tree.add(new_node)

        if np.linalg.norm(new_node.pos - goal) <= params.goal_radius:
            # reconstruct
            path_pos = [new_node.pos]
            path_t = [new_node.t]
            cur = new_node
            while cur.parent is not None:
                cur = tree.nodes[cur.parent]
                path_pos.append(cur.pos)
                path_t.append(cur.t)
            return np.array(list(reversed(path_pos))), np.array(list(reversed(path_t))), tree
    return None, None, tree


# --------------------------
# Joint RRT: meta-robot with m agents
# --------------------------
def steer_and_propagate_multi(P_from: np.ndarray, t_from: float, goals: np.ndarray, params: RRTParams):
    # P_from: (m,2); goals: (m,2)
    m = P_from.shape[0]
    K = random.randint(params.K_min, params.K_max)
    Pts = [P_from.copy()]
    Ts = [t_from]
    P = P_from.copy()
    t = t_from
    for _ in range(K):
        V = []  # controls per agent
        for r in range(m):
            d = goals[r] - P[r]
            n = np.linalg.norm(d)
            heading = np.array([1.0, 0.0]) if n < 1e-9 else d / n
            theta = random.uniform(-math.pi / 6, math.pi / 6)
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            heading = (R @ heading.reshape(2, 1)).ravel()
            if np.linalg.norm(heading) < 1e-9: heading = np.array([1.0, 0.0])
            V.append(heading * params.v_max)
        V = np.stack(V, axis=0)  # (m,2)
        P = P + V * params.dt
        t = t + params.dt
        Pts.append(P.copy())
        Ts.append(t)
    return np.stack(Pts, axis=0), np.array(Ts)


def rrt_plan_joint(
        starts: np.ndarray,
        goals: np.ndarray,
        world: World,
        params: RRTParams,
        path_constraints: List[PathObstacle],
        seed: int = 0
):
    # starts: (m,2), goals: (m,2)
    random.seed(seed)
    np.random.seed(seed)
    m = starts.shape[0]
    tree = Tree(nodes=[Node(pos=starts.flatten().copy(), t=0.0, parent=None)])

    # small precheck: stationary
    if not is_valid_segment_multi(np.stack([starts, starts], axis=0), np.array([0.0, 1e-3]), world, params.dt_check,
                                  path_constraints):
        return None, None, tree

    for _ in range(params.n_iter):
        # goal bias: towards goals centroid (for nearest selection); propagate using per-agent goals
        centroid_goal = goals.mean(axis=0)
        p_rand = centroid_goal if random.random() < params.prob_goal_bias else np.array([
            random.uniform(world.bounds[0][0], world.bounds[0][1]),
            random.uniform(world.bounds[1][0], world.bounds[1][1])
        ])
        idx = tree.nearest_index(np.tile(p_rand, m))  # nearest in concatenated space by repeating hint
        node = tree.nodes[idx]
        P_from = node.pos.reshape(m, 2)
        Pts, Ts = steer_and_propagate_multi(P_from, node.t, goals, params)
        if Ts[-1] > params.t_max:
            continue
        if not is_valid_segment_multi(Pts, Ts, world, params.dt_check, path_constraints):
            continue
        new_node = Node(pos=Pts[-1].flatten(), t=float(Ts[-1]), parent=idx)
        tree.add(new_node)
        # goal check: all within radius
        if np.all(np.linalg.norm(Pts[-1] - goals, axis=1) <= params.goal_radius):
            # reconstruct
            cur = new_node
            seq = [cur.pos.reshape(m, 2)]
            tseq = [cur.t]
            while cur.parent is not None:
                cur = tree.nodes[cur.parent]
                seq.append(cur.pos.reshape(m, 2))
                tseq.append(cur.t)
            seq = list(reversed(seq))
            tseq = list(reversed(tseq))
            return np.stack(seq, axis=0), np.array(tseq), tree  # (L,m,2), (L,)
    return None, None, tree


# --------------------------
# CBS-style high level with merging
# --------------------------
@dataclass
class AgentSpec:
    start: np.ndarray
    goal: np.ndarray


@dataclass
class EntityPlan:
    # For single: pos:(L,2); ts:(L,). For group: pos_multi:(L,m,2); ts:(L,)
    pos: Optional[np.ndarray]
    ts: Optional[np.ndarray]
    is_group: bool
    members: List[int]  # original agent indices
    tree: Tree


@dataclass
class HLNode:
    plans: Dict[int, EntityPlan]  # ent_id -> plan
    constraints: Dict[int, List[PathObstacle]]  # ent_id -> PathObstacle list
    grouping: Dict[int, List[int]]  # ent_id -> member agent ids
    cost: float
    conflict_count: Dict[Tuple[int, int], int]  # pairwise conflict hits


def make_entity_plan_single(
        agent_id: int,
        start: np.ndarray,
        goal: np.ndarray,
        world: World,
        params: RRTParams,
        constraints: List[PathObstacle],
        seed: int
):
    pos, ts, tree = rrt_plan_single(start, goal, world, params, path_constraints=constraints, seed=seed)
    return EntityPlan(pos=pos, ts=ts, is_group=False, members=[agent_id], tree=tree)


def make_entity_plan_group(
        members: List[int],
        starts: np.ndarray,
        goals: np.ndarray,
        world: World,
        params: RRTParams,
        constraints: List[PathObstacle],
        seed: int
):
    P, T, tree = rrt_plan_joint(starts, goals, world, params, path_constraints=constraints, seed=seed)
    return EntityPlan(pos=P, ts=T, is_group=True, members=list(members), tree=tree)


def detect_first_conflict_window(
        plans: Dict[int, EntityPlan],
        robot_radius: float,
        dt_check: float
):
    # returns (a_ent, b_ent, t_s, t_e) or None
    ent_ids = list(plans.keys())
    for i_idx in range(len(ent_ids)):
        ei = ent_ids[i_idx]
        Pi = plans[ei]
        if Pi.pos is None: continue
        for j_idx in range(i_idx + 1, len(ent_ids)):
            ej = ent_ids[j_idx]
            Pj = plans[ej]
            if Pj.pos is None: continue

            # unify to (pos(t): (2,) for single; or set of members for group)
            # We'll discretize time overlap and check min distance across any member pairs
            ti0, ti1 = float(Pi.ts[0]), float(Pi.ts[-1])
            tj0, tj1 = float(Pj.ts[0]), float(Pj.ts[-1])
            t0, t1 = max(ti0, tj0), min(ti1, tj1)
            if t1 < t0: continue
            T = np.arange(t0, t1 + 1e-9, dt_check)
            if T[-1] < t1: T = np.append(T, t1)

            def interp_multi(Plan: EntityPlan, t: float) -> np.ndarray:
                if Plan.is_group:
                    # interpolate each member
                    if t <= Plan.ts[0]: return Plan.pos[0]
                    if t >= Plan.ts[-1]: return Plan.pos[-1]
                    k = np.searchsorted(Plan.ts, t, side="right") - 1
                    k = max(0, min(k, len(Plan.ts) - 2))
                    ta, tb = Plan.ts[k], Plan.ts[k + 1]
                    a = 0.0 if tb == ta else (t - ta) / (tb - ta)
                    return (1 - a) * Plan.pos[k] + a * Plan.pos[k + 1]
                else:
                    p = interp_path(Plan.pos, Plan.ts, t)
                    return p.reshape(1, 2)

            # find contiguous conflict intervals
            in_conflict = False
            t_s = None
            for tau in T:
                PMi = interp_multi(Pi, tau)  # (mi,2)
                PMj = interp_multi(Pj, tau)  # (mj,2)
                # check min pairwise distance
                collide = False
                for a in range(PMi.shape[0]):
                    for b in range(PMj.shape[0]):
                        if np.linalg.norm(PMi[a] - PMj[b]) <= 2 * robot_radius:
                            collide = True
                            break
                    if collide: break

                if collide and not in_conflict:
                    in_conflict = True
                    t_s = float(tau)
                elif (not collide) and in_conflict:
                    # end conflict
                    return (ei, ej, t_s, float(tau))
            if in_conflict:
                return (ei, ej, t_s, float(T[-1]))
    return None


def total_cost(
        plans: Dict[int, EntityPlan]
) -> float:
    c = 0.0
    for _, P in plans.items():
        if P.pos is None:
            return float("inf")
        c += float(P.ts[-1])
    return c


def kcbs_with_merge(
        agents: List[AgentSpec],
        world: World, params: RRTParams,
        merge_threshold: int = 2,
        max_nodes: int = 80,
        seed: int = 0
):
    random.seed(seed)

    # ent_id -> members
    grouping = {i: [i] for i in range(len(agents))}
    constraints = {i: [] for i in grouping.keys()}
    plans: Dict[int, EntityPlan] = {}

    # initial independent plans
    for i in grouping.keys():
        ag = agents[i]
        plans[i] = make_entity_plan_single(i, ag.start, ag.goal, world, params, constraints[i],
                                           seed=random.randint(0, 10 ** 6))

    root = HLNode(plans=plans, constraints=constraints, grouping=grouping,
                  cost=total_cost(plans), conflict_count={})
    open_list = [root]

    expansions = 0
    while open_list and expansions < max_nodes:
        open_list.sort(key=lambda n: n.cost)
        cur = open_list.pop(0)

        confl = detect_first_conflict_window(cur.plans, world.robot_radius, params.dt_check)
        if confl is None:
            return cur.plans  # solved

        a_ent, b_ent, t_s, t_e = confl
        pair = tuple(sorted((a_ent, b_ent)))
        cc = dict(cur.conflict_count)
        cc[pair] = cc.get(pair, 0) + 1

        # MERGE if exceeded threshold
        if cc[pair] >= merge_threshold:
            # new entity id
            new_ent = max(cur.grouping.keys()) + 1
            members = cur.grouping[a_ent] + cur.grouping[b_ent]
            # assemble starts/goals
            starts = np.stack([agents[m].start for m in members], axis=0)
            goals = np.stack([agents[m].goal for m in members], axis=0)
            # constraints for the merged entity are union of previous constraints (converted)
            new_constraints = []
            for e_id, P in cur.plans.items():
                if e_id in (a_ent, b_ent): continue
                if P.pos is None: continue
                if P.is_group:
                    for r_idx in range(P.pos.shape[1]):
                        new_constraints.append(
                            PathObstacle(ts=P.ts.copy(), pts=P.pos[:, r_idx, :].copy(), radius=world.robot_radius))
                else:
                    new_constraints.append(PathObstacle(ts=P.ts.copy(), pts=P.pos.copy(), radius=world.robot_radius))
            # plan joint
            Pm, Tm, Ttree = rrt_plan_joint(starts, goals, world, params, path_constraints=new_constraints,
                                           seed=random.randint(0, 10 ** 6))
            new_plans = {k: v for k, v in cur.plans.items() if k not in (a_ent, b_ent)}
            new_plans[new_ent] = EntityPlan(pos=Pm, ts=Tm, is_group=True, members=members, tree=Ttree)

            new_constraints_map = {k: v for k, v in cur.constraints.items() if k not in (a_ent, b_ent)}
            new_constraints_map[new_ent] = new_constraints

            new_grouping = {k: v for k, v in cur.grouping.items() if k not in (a_ent, b_ent)}
            new_grouping[new_ent] = members

            open_list.append(HLNode(plans=new_plans, constraints=new_constraints_map,
                                    grouping=new_grouping, cost=total_cost(new_plans),
                                    conflict_count=cc))
            expansions += 1
            continue

        # Otherwise: branch with time-window cropped constraints
        # Build path obstacles sliced to [t_s, t_e]
        def build_windowed_obstacles(plan: EntityPlan, t_s: float, t_e: float) -> List[PathObstacle]:
            obs = []
            if plan.is_group:
                for r_idx in range(plan.pos.shape[1]):
                    pts, ts = slice_polyline_by_time(plan.pos[:, r_idx, :], plan.ts, t_s, t_e)
                    obs.append(PathObstacle(ts=ts, pts=pts, radius=world.robot_radius))
            else:
                pts, ts = slice_polyline_by_time(plan.pos, plan.ts, t_s, t_e)
                obs.append(PathObstacle(ts=ts, pts=pts, radius=world.robot_radius))
            return obs

        for (constrain_ent, other_ent) in [(a_ent, b_ent), (b_ent, a_ent)]:
            new_constraints_map = {k: list(v) for k, v in cur.constraints.items()}
            new_constraints_map[constrain_ent] = list(new_constraints_map[constrain_ent]) + build_windowed_obstacles(
                cur.plans[other_ent], t_s, t_e)

            # replan constrained entity (single or group)
            if len(cur.grouping[constrain_ent]) == 1:
                aid = cur.grouping[constrain_ent][0]
                ag = agents[aid]
                EP = make_entity_plan_single(aid, ag.start, ag.goal, world, params,
                                             new_constraints_map[constrain_ent], seed=random.randint(0, 10 ** 6))
            else:
                members = cur.grouping[constrain_ent]
                starts = np.stack([agents[m].start for m in members], axis=0)
                goals = np.stack([agents[m].goal for m in members], axis=0)
                EP = make_entity_plan_group(members, starts, goals, world, params,
                                            new_constraints_map[constrain_ent], seed=random.randint(0, 10 ** 6))

            new_plans = {k: v for k, v in cur.plans.items()}
            new_plans[constrain_ent] = EP

            open_list.append(
                HLNode(
                    plans=new_plans,
                    constraints=new_constraints_map,
                    grouping=cur.grouping,
                    cost=total_cost(new_plans),
                    conflict_count=cc)
            )
        expansions += 1

    return None


# --------------------------
# Demo world & agents
# --------------------------
def build_world() -> World:
    bounds = ((0.0, 20.0), (0.0, 20.0))
    static_obs = [
        StaticCircle(center=np.array([6.0, 7.0]), radius=1.2),
        StaticCircle(center=np.array([12.0, 12.0]), radius=1.5),
        StaticCircle(center=np.array([9.0, 4.0]), radius=1.0),
    ]
    moving_obs = [MovingCircle(center0=np.array([4.0, 15.0]), vel=np.array([0.08, -0.05]), radius=1.0)]
    return World(bounds=bounds, static_obstacles=static_obs, moving_obstacles=moving_obs, robot_radius=0.25)


def build_agents_three() -> List[AgentSpec]:
    A = AgentSpec(start=np.array([1.0, 1.2]), goal=np.array([18.5, 18.0]))
    B = AgentSpec(start=np.array([18.0, 1.0]), goal=np.array([1.5, 18.0]))
    C = AgentSpec(start=np.array([2.0, 18.5]), goal=np.array([17.5, 2.0]))
    return [A, B, C]


# --------------------------
# Visualization: static and animated
# --------------------------
def draw_world(ax, world: World):
    ax.set_xlim(world.bounds[0][0], world.bounds[0][1])
    ax.set_ylim(world.bounds[1][0], world.bounds[1][1])
    ax.set_aspect('equal', adjustable='box')
    for obs in world.static_obstacles:
        ax.add_patch(Circle(obs.center, obs.radius, fill=False))
    # show moving obstacle path (sampled)
    ts_vis = np.linspace(0.0, 20.0, 25)
    m = world.moving_obstacles[0]
    centers = np.stack([m.center_at(t) for t in ts_vis], axis=0)
    ax.plot(centers[:, 0], centers[:, 1], linestyle='--')
    for c in centers[::5]:
        ax.add_patch(Circle(c, m.radius, fill=False, linestyle=':'))


# Animated GIF
def build_all_member_paths(plans: Dict[int, EntityPlan]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Flatten entity plans into a list of (pos, ts) per original agent index order."""
    # Gather by original member index to keep consistent order
    flat: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for EP in plans.values():
        if EP.pos is None: continue
        if EP.is_group:
            m = EP.pos.shape[1]
            for idx, agent_id in enumerate(EP.members):
                flat[agent_id] = (EP.pos[:, idx, :], EP.ts)
        else:
            flat[EP.members[0]] = (EP.pos, EP.ts)
    return [flat[i] for i in sorted(flat.keys())]


def animate_solution(
        world: World,
        agents: List[AgentSpec],
        plans: Dict[int, EntityPlan],
        out_path: str
):
    flat = build_all_member_paths(plans)
    # time horizon
    t_end = max(ts[-1] for (_, ts) in flat)
    dt = 0.12
    T = np.arange(0.0, t_end + 1e-9, dt)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_title("KCBS Solution (animated)")
    draw_world(ax, world)
    # robot artists (circles)
    patches = []
    for _ in flat:
        c = Circle((0.0, 0.0), world.robot_radius, fill=False)
        ax.add_patch(c)
        patches.append(c)

    def update(frame_idx):
        t = T[frame_idx]
        for k, (pos, ts) in enumerate(flat):
            p = interp_path(pos, ts, t)
            patches[k].center = (p[0], p[1])
        return patches

    ani = animation.FuncAnimation(fig, update, frames=len(T), interval=50, blit=True)
    # Use PillowWriter for wide compatibility
    ani.save(out_path, writer=animation.PillowWriter(fps=int(1.0 / dt)))
    plt.close(fig)