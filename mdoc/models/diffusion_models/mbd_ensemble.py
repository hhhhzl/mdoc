import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from mdoc.config import MDOCParams as mparams
import einops
from torch_robotics.robots.robot_base import RobotState
from mp_baselines.planners.costs.cost_functions import CostConstraint
import torch._dynamo
import torch, contextlib

torch._dynamo.config.suppress_errors = True

import torch

torch.set_num_threads(16)
torch.set_num_interop_threads(1)


class ModelBasedDiffusionEnsemble(nn.Module):
    """
    Unlike Model-free diffusion,
    model-based diffusion interact with model information,
    and analytically calculate score function through Monte Carlo Approximation
    """

    def __init__(
            self,
            robot,
            env_models: Dict[int, Any],
            transforms: Dict[int, torch.Tensor],
            start_state_pos: torch.tensor,
            goal_state_pos: torch.tensor,
            seed: int = 0,
            enable_demo: bool = False,
            context_model=None,
            disable_recommended_params: bool = True,
            device: str = 'cpu',
            **kwargs
    ):
        """

        Args:
            env_models: model_environments {model_id: env_model}
            env_name: model_env name
            seed: random seed
            enable_demo: whether to use demonstration
            disable_recommended_params: whether to use recommended parames
            device: device
        """
        super().__init__(**kwargs)
        self.device = device
        self.robot = robot
        self.env_models = env_models
        self.enable_demo = enable_demo
        self.disable_recommended_params = disable_recommended_params

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.diffusion_params = {
            'temp_sample': mparams.temp_sample,
            'n_samples': mparams.n_samples,
            'n_diffusion_step': mparams.n_diffusion_steps,
            'horizon': 64,
            'beta0': 1e-5,
            'betaT': 1e-2
        }

        self.recommended_params = {
            "temp_sample": 0.5,
            "n_diffusion_step": 100,
            'n_samples': 64,
            "horizon": 64
        }

        # Initialization
        self.models = {}
        self.soft_constraints = {}
        self.start_state_pos = start_state_pos
        self.goal_state_pos = goal_state_pos
        for model_id, env_model in env_models.items():
            self._setup_model_environment(model_id, env_model)
            self.soft_constraints[model_id] = []
            # self._add_endpoint_soft_constraints(model_id)
        self.transforms = transforms
        self.context_model = context_model

        # for cache (ECBS calculate collision cost)
        self._centers = None
        self._radii2 = None
        self._time_mask = None

        # for non empty environment
        sdf_cell_size = 0.05
        self.sdf_cell_size = float(sdf_cell_size)
        self._sdf_tex = None
        self._sdf_tex_meta = None

    def _prepare_soft_constraints(
            self,
            model_id: int,
            H: int,
            device: torch.device
    ):
        """
        Cache for diffusion step：
            _centers    [1,1,N,q_dim]
            _radii2     [1,1,N]
            _time_mask  [1,H,N,1]
        """
        qs_l, radii_l, t0_l, t1_l, w_l = [], [], [], [], []
        for c in self.soft_constraints[model_id]:
            qs_l.append(c.qs.to(device))  # [K,q_dim]
            radii_l.append(torch.as_tensor(c.radii, device=device))
            tr = torch.as_tensor(c.traj_ranges, device=device)  # [K,2]
            t0_l.append(tr[:, 0])
            t1_l.append(tr[:, 1])
            if isinstance(c.priority_weight, torch.Tensor):
                w_l.append(c.priority_weight.to(device))
            else:
                # w_l.append(torch.full((len(c.qs),), c.priority_weight, device=device))
                w_l.append(1)

        centers = torch.cat(qs_l, dim=0)  # [N,q_dim]
        radii = torch.cat(radii_l, dim=0)  # [N]
        self._t0 = torch.cat(t0_l, dim=0)  # [N]
        self._t1 = torch.cat(t1_l, dim=0)  # [N]
        self._N = centers.shape[0]
        self._centers = centers.view(1, 1, -1, self.robot.q_dim)  # [1,1,N,q_dim]
        self._radii = radii.view(1, 1, self._N)  # [1,1,N]
        self._centers_flat = centers  # [N,q_dim]  (contiguous)
        self._radii_flat = radii  # [N]
        # self._weights = torch.cat(w_l).view(1, 1, -1, 1)  # (1,1,N,1)

    def _setup_model_environment(self, model_id: int, env_model: Any):
        params = self.diffusion_params.copy()
        if not self.disable_recommended_params:
            for param_name in ["temp_sample", "n_diffusion_step", "n_samples", "horizon"]:
                if param_name in self.recommended_params:
                    params[param_name] = self.recommended_params[param_name]

        setattr(self, f'model_{model_id}_params', params)
        self.state_inits = RobotState(
            q=self.start_state_pos,
            q_dot=torch.Tensor([0, 0]),
            cost=0,
            collision_cost=0,
            control_cost=0,
            pipeline_state=torch.cat([self.start_state_pos, self.start_state_pos])
        )
        self._setup_diffusion_params(model_id, params)

    def _setup_diffusion_params(self, model_id: int, params: dict):
        """set parameter for each model"""
        betas = torch.linspace(params['beta0'], params['betaT'], params['n_diffusion_step'], device=self.device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        sigmas = torch.sqrt(1 - alphas_bar)

        # Conditional parameters
        shifted_alphas_bar = torch.roll(alphas_bar, 1)
        shifted_alphas_bar[0] = 1.0
        sigmas_cond = ((1 - alphas) * (1 - torch.sqrt(shifted_alphas_bar)) / (1 - alphas_bar))
        sigmas_cond = torch.sqrt(sigmas_cond)
        sigmas_cond[0] = 0.0

        self.models[model_id] = {
            'betas': betas,
            'alphas': alphas,
            'alphas_bar': alphas_bar,
            'sigmas': sigmas,
            'sigmas_cond': sigmas_cond,
            'params': params
        }

    @torch.no_grad()
    def _normalize_goal_xy(self, goal_store, model_id, B_expected, device, dtype, fallback_y: torch.Tensor):
        """
        返回 (B,2) 的 (x,y) Tensor。
        允许 goal_store 是：
          - list/tuple/字典/张量集合：先取 index/键，再解析
          - 单个对象/张量/标量：直接解析
        能识别的字段：.xy/.pos/.goal/.q/.state，或 .x/.y；dict 可用 {"x":..,"y":..}
        对于“标量/单元素”的情况：视为只给了 x，y 用 fallback_y（通常为 state0[:,1]）。
        """
        # 先取出当前 model_id 的那一项
        g = goal_store
        if isinstance(g, (list, tuple)):
            g = g[model_id]
        elif isinstance(g, dict):
            g = g.get(model_id, g.get("goal", g))
        else:
            try:
                g = g[model_id]
            except Exception:
                pass

        # dict: {"x":..., "y":...}
        if isinstance(g, dict):
            x = g.get("x", None)
            y = g.get("y", None)
            if x is None and y is None:
                # 退回到下面的通用路径
                pass
            else:
                if x is None:
                    raise ValueError("goal dict 缺少 'x'")
                x_t = torch.as_tensor(x, device=device, dtype=dtype).view(1)
                if y is None:
                    # 只给了 x：y 用 fallback
                    y_t = fallback_y
                else:
                    y_t = torch.as_tensor(y, device=device, dtype=dtype).view(1).expand(B_expected)
                xy = torch.stack((x_t.expand(B_expected), y_t), dim=-1)
                return xy

        # 尝试从对象字段拿 tensor
        t = None
        if isinstance(g, torch.Tensor):
            t = g.to(device=device, dtype=dtype)
        elif isinstance(g, (list, tuple)):
            t = torch.tensor(g, device=device, dtype=dtype)
        else:
            for name in ("xy", "pos", "goal", "q", "state", "data"):
                if hasattr(g, name):
                    cand = getattr(g, name)
                    if isinstance(cand, torch.Tensor):
                        t = cand.to(device=device, dtype=dtype)
                        break
                    elif isinstance(cand, (list, tuple)):
                        t = torch.tensor(cand, device=device, dtype=dtype)
                        break
            # 支持 .x/.y
            if t is None and (hasattr(g, "x") or hasattr(g, "y")):
                x = getattr(g, "x", None)
                y = getattr(g, "y", None)
                if x is None:
                    raise ValueError("goal 对象只有 y 没有 x，无法构造 (x,y)")
                x_t = torch.as_tensor(x, device=device, dtype=dtype).view(1).expand(B_expected)
                if y is None:
                    y_t = fallback_y
                else:
                    y_t = torch.as_tensor(y, device=device, dtype=dtype).view(1).expand(B_expected)
                return torch.stack((x_t, y_t), dim=-1)

        if t is None:
            # 可能是纯标量（int/float/np scalar）
            try:
                x_scalar = float(g)
                x_t = torch.tensor(x_scalar, device=device, dtype=dtype).view(1).expand(B_expected)
                y_t = fallback_y
                return torch.stack((x_t, y_t), dim=-1)
            except Exception:
                raise TypeError(f"无法从 goal 中解析出坐标，类型={type(g)}")

        # 到这里 t 是 tensor
        if t.dim() == 0:
            # 标量 → 视为 x，y=fallback
            x_t = t.view(1).expand(B_expected)
            y_t = fallback_y
            return torch.stack((x_t, y_t), dim=-1)

        if t.dim() == 1:
            if t.numel() == 2:
                # (2,) → (1,2) 再广播
                t = t.view(1, 2)
            elif t.numel() == 1:
                # 单元素 → x，y=fallback
                x_t = t.view(1).expand(B_expected)
                y_t = fallback_y
                return torch.stack((x_t, y_t), dim=-1)
            else:
                raise ValueError(f"无法解析 goal: shape={tuple(t.shape)}，需要 1 或 2 个元素")

        if t.dim() > 2:
            t = t.view(t.size(0), -1)

        # 现在 t 至少是 (N, >=2)
        if t.size(-1) < 2:
            # 只有一个元素 → x，y=fallback
            x_t = t[:, 0]
            if x_t.numel() == 1:
                x_t = x_t.view(1).expand(B_expected)
                y_t = fallback_y
                return torch.stack((x_t, y_t), dim=-1)
            if x_t.numel() == B_expected:
                return torch.stack((x_t, fallback_y), dim=-1)
            raise ValueError(f"goal 最后一维 < 2，且无法对齐 batch 大小：shape={tuple(t.shape)}")

        # 对齐 batch
        if t.size(0) == 1 and B_expected > 1:
            t = t[:, :2].expand(B_expected, 2)
        elif t.size(0) != B_expected:
            t = t[:1, :2].expand(B_expected, 2)

        return t[:, :2]
    @torch.no_grad()
    def _normalize_state_xy(self, state_init, B_expected, device, dtype):
        """
        返回 (B,2) 的 float Tensor，表示世界系的 (x,y) 初始位置。
        兼容：
          - torch.Tensor: 形如 (B, D) 或 (B,2) 或 (2,)
          - RobotState-like: 具有 .q / .pos / .xy / .state 等属性
        若只有单个状态而 us 是 (B, ...)，则会自动重复到 B。
        """
        # 1) 从对象里取出底层张量/数组
        t = None
        if isinstance(state_init, torch.Tensor):
            t = state_init
        else:
            # 常见容器字段名尝试（按你项目实际改一下就行）
            for name in ("q", "pos", "xy", "state", "data"):
                if hasattr(state_init, name):
                    candidate = getattr(state_init, name)
                    if isinstance(candidate, torch.Tensor):
                        t = candidate
                        break
            # 还找不到，可以尝试 __array__ / 转成张量
            if t is None and hasattr(state_init, "__array__"):
                import numpy as np
                t = torch.as_tensor(np.asarray(state_init.__array__()))

        if t is None:
            raise TypeError(
                f"state_init 的类型不受支持：{type(state_init)}，"
                f"请传 torch.Tensor，或在对象里提供 .q/.pos/.xy/.state 等张量属性。"
            )

        # 2) 规范形状到 (B, >=2)
        if t.dim() == 1:
            # (D,) → (1,D)
            t = t.view(1, -1)
        elif t.dim() > 2:
            # 尝试把多余维度摊平到 (B, D)
            t = t.view(t.size(0), -1)

        if t.size(0) == 1 and B_expected > 1:
            # 单个起点，批量动作 → 重复到 B
            t = t.expand(B_expected, t.size(1))
        elif t.size(0) != B_expected:
            raise ValueError(f"state_init batch 大小 {t.size(0)} 与 us 的 {B_expected} 不一致")

        if t.size(-1) < 2:
            raise ValueError(f"state_init 的最后一维 < 2，无法提取 (x,y)：shape={tuple(t.shape)}")

        # 3) 提取前两维 (x,y) 并转 dtype/device
        xy = t[:, :2].to(device=device, dtype=dtype)
        return xy

    # --------------------------------------------------------------------- #
    # Soft "safety ball" around start / goal poses
    # --------------------------------------------------------------------- #
    def _add_endpoint_soft_constraints(self, model_id, radius=0.05, window=10, weight=1.0):
        """
        Create and register two CostConstraint objects that keep the first
         <window> and last <window> timesteps within a ball of <radius>
        around the (potentially very wall‑adjacent) start and goal poses.
        """

        horizon = self.models[model_id]['params']['horizon']
        qs = [self.start_state_pos.clone(), self.goal_state_pos.clone()]
        traj_ranges = [(0, window - 1), (horizon - window, horizon - 1)]
        radii = [radius, radius]
        constraint = CostConstraint(
            self.robot,
            horizon,
            q_l=qs,
            traj_range_l=traj_ranges,
            radius_l=radii,
            is_soft=True,
            tensor_args={'device': self.device},
            priority_weight=weight
        )
        self.soft_constraints[model_id].append(constraint)

    def _project_to_free_space(self, q_ws, env_model, robot_radius, max_iter=1, grad_clip=5.0):
        """hard constraint"""
        B, H, _ = q_ws.shape
        q_flat = q_ws.reshape(-1, self.robot.q_dim).detach().clone()
        eps = 1e-6

        for _ in range(max_iter):
            # Compute SDF values
            sdf_val = env_model.compute_sdf(q_flat)  # [B*H]

            # Compute gradients using finite differences
            grad = torch.zeros_like(q_flat)
            for dim in range(q_flat.shape[-1]):
                q_perturbed = q_flat.clone()
                q_perturbed[:, dim] += eps
                sdf_perturbed = env_model.compute_sdf(q_perturbed)
                grad[:, dim] = (sdf_perturbed - sdf_val) / eps

            sdf_grad = grad.clamp(-grad_clip, grad_clip)
            # Compute clearance and collision mask
            clearance = sdf_val - (robot_radius + 1e-8)
            inside = clearance < 0

            if not inside.any():
                break

            # Project points out of obstacles
            grad_inside = sdf_grad[inside]
            grad_norm2 = (grad_inside ** 2).sum(dim=1, keepdim=True).clamp_min(1e-6)
            step = clearance[inside].unsqueeze(-1) * grad_inside / grad_norm2
            q_flat[inside] -= step

        return q_flat.view(B, H, self.robot.q_dim)

    def _project_traj_with_segment_check(
            self,
            q_ws,
            env_model,
            robot_radius,
            step=0.03,
            max_project_iter=1,
            grad_clip=5.0
    ):
        """
        """
        B, H0, _ = q_ws.shape
        flag = False
        q_ws = self._project_to_free_space(
            q_ws,
            env_model,
            robot_radius,
            max_iter=max_project_iter,
            grad_clip=grad_clip
        )

        new_trajs = []
        for b in range(B):
            pts = q_ws[b]  # [H,2]
            i = 0
            while i < pts.shape[0] - 1:
                p0, p1 = pts[i], pts[i + 1]
                seg_len = torch.norm(p1 - p0).item()
                n_samples = max(int(seg_len / step), 1)
                if n_samples == 1:
                    i += 1
                    continue
                alphas = torch.linspace(0, 1, n_samples + 1, device=pts.device)[1:]
                inter = p0 * (1 - alphas).unsqueeze(1) + p1 * alphas.unsqueeze(1)  # [n,2]
                sdf = env_model.compute_sdf(inter) - robot_radius
                if (sdf < 0).any():
                    flag = True
                    mid = (p0 + p1) * 0.5
                    pts = torch.cat([pts[:i + 1],
                                     mid.unsqueeze(0),
                                     pts[i + 1:]], dim=0)
                    pts[i + 1:i + 2] = self._project_to_free_space(
                        pts[i + 1:i + 2].unsqueeze(0), env_model, robot_radius
                    ).squeeze(0)
                else:
                    i += 1
            new_trajs.append(pts)

        max_H = max(t.shape[0] for t in new_trajs)
        padded = torch.full((B, max_H, 2), float('nan'), device=q_ws.device)
        for b, t in enumerate(new_trajs):
            padded[b, :t.shape[0]] = t

        return padded, flag

    def _segment_free_mask(self, q_ws, env_model, robot_radius, step=0.03):
        """
        Return Bool mask indicating whether each trajectory in the batch is
        entirely collision‑free when each poly‑line segment is densely sampled
        every <step> meters.
        """

        B, H, _ = q_ws.shape
        device = q_ws.device
        free_mask = torch.ones(B, dtype=torch.bool, device=device)
        for b in range(B):
            pts = q_ws[b]
            for i in range(H - 1):
                p0, p1 = pts[i], pts[i + 1]
                seg_len = torch.norm(p1 - p0).item()
                n_samples = max(int(seg_len / step), 1)
                if n_samples == 1:
                    continue
                alphas = torch.linspace(0.0, 1.0, n_samples + 1, device=device)[1:]
                inter = p0 * (1 - alphas).unsqueeze(1) + p1 * alphas.unsqueeze(1)
                sdf = env_model.compute_sdf(inter) - robot_radius
                if (sdf < 0).any():
                    free_mask[b] = False
                    break

        return free_mask

    # ---------- CUDA Graph compile ----------
    def rollout_ultrafast(self, model_id, state_init, us, use_amp: bool = False):
        """
        Rollout 入口（数学与 new2 等价）：
          - 自动 compile；CUDA 下固定形状时自动用 CUDA Graph 重放
          - 返回: cost_sum:(B,), q_seq:(B,H,2), free_mask:(B,)
        """
        device = us.device
        B, H, Udim = us.shape
        key = (str(device), us.dtype, B, H, model_id)

        if not hasattr(self, "_ultra_cache"):
            self._ultra_cache = {}
            self._ultra_flags = {"compile": True, "cuda_graph": True}

        entry = self._ultra_cache.get(key)

        if entry is None:
            def _core(state_init_, us_):
                return self._rollout_single_batch_new2(model_id, state_init_, us_)

            enable_compile = self._ultra_flags["compile"] and (device.type != "cpu")
            runner = _core
            if enable_compile:
                try:
                    runner = torch.compile(runner, mode="max-autotune", fullgraph=True)
                except Exception:
                    runner = _core

            enable_cudagraph = self._ultra_flags["cuda_graph"] and (device.type == "cuda")
            graph = None
            if enable_cudagraph:
                try:
                    static_in_state = state_init.detach().clone()
                    static_in_us = us.detach().clone()
                    with torch.cuda.amp.autocast(enabled=False):
                        warm_cost, warm_q, warm_free = runner(static_in_state, static_in_us)
                    static_out_cost = torch.empty_like(warm_cost)
                    static_out_q = torch.empty_like(warm_q)
                    static_out_free = torch.empty_like(warm_free)

                    g = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(g):
                        c, q, f = runner(static_in_state, static_in_us)
                        static_out_cost.copy_(c)
                        static_out_q.copy_(q)
                        static_out_free.copy_(f)
                    graph = (g, static_in_state, static_in_us, static_out_cost, static_out_q, static_out_free)
                except Exception:
                    graph = None

            entry = {"runner": runner, "graph": graph}
            self._ultra_cache[key] = entry

        if entry["graph"] is not None:
            g, static_in_state, static_in_us, static_out_cost, static_out_q, static_out_free = entry["graph"]
            static_in_state.copy_(state_init)
            static_in_us.copy_(us)
            g.replay()
            return static_out_cost.clone(), static_out_q.clone(), static_out_free.clone()

        # 普通编译 runner（AMP：CUDA bf16 / MPS fp16）
        if device.type == "cuda":
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=bool(use_amp))
        elif device.type == "mps":
            ctx = torch.autocast(device_type="mps", dtype=torch.float16, enabled=bool(use_amp))
        else:
            ctx = torch.no_grad() if not torch.is_grad_enabled() else torch.enable_grad()

        with ctx:
            return entry["runner"](state_init, us)

    @torch.no_grad()
    def _rollout_ultrafast_core(self, model_id, state_init, us):
        """
        UltraFast (Single-Integrator 对齐版)
        输出：
          cost_sum: (B,)
          q_ws    : (B,H,2) 世界系
          free_mk : (B,)
        说明：
          - us 被视为速度输入（single integrator）：dq = us * dt；q = q0 + cumsum(dq)
          - 代价对齐 single_batch：cost_t = ||u||^2 + 500 * ||q - goal||
          - 自由掩码使用 mparams.robot_planar_disk_radius + 1e-8，与 single_batch 一致
          - 保留 ultrafast 的预分配缓冲/纹理 SDF fast-path/transform 等快速结构
          - 关闭 v_bar 跟踪、进度项、终端额外项、路径 SDF 罚等（以保证数值对齐）
        """
        device, dtype = us.device, us.dtype
        B, H, Udim = us.shape
        dt = float(self.robot.dt)

        # 半径取 mparams（与 single_batch 对齐），并且不加额外 margin（只在 free mask 扣 1e-8）
        R = float(mparams.robot_planar_disk_radius)

        # ===== 缓冲区（保持 ultrafast 结构） =====
        if device.type == "cpu":
            bufs = self._get_cpu_bufs(B, H, device, dtype)
            q, dq, ddq_unused, q_ws = bufs["q"], bufs["dq"], bufs["ddq"], bufs["q_ws"]
            runtime_cost, tmp2, _vbar_unused = bufs["rc"], bufs["tmp2"], bufs["vbar"]
        else:
            q = torch.empty((B, H, 2), device=device, dtype=dtype)
            dq = torch.empty((B, H, 2), device=device, dtype=dtype)
            q_ws = torch.empty((B, H, 2), device=device, dtype=dtype)
            runtime_cost = torch.zeros((B, H), device=device, dtype=dtype)
            tmp2 = torch.empty((B, H, 2), device=device, dtype=dtype)

        # ===== 初值（严格与 single_batch 的 q0 一致）=====
        # 归一化/防切片写入，形状 (B,2)
        state0 = self._normalize_state_xy(state_init, B_expected=B, device=device, dtype=dtype)
        q.copy_(state0.view(B, 1, 2).expand(B, H, 2))  # 按时间轴填起点
        dq.copy_(us).mul_(dt)  # dq = u * dt   （SI）
        torch.cumsum(dq, dim=1, out=tmp2)  # 累计位移
        q.add_(tmp2)  # q = q0 + cumsum(dq)

        # ===== 目标（与 single_batch 一致：直接用 goal_state_pos，而非标量 fallback 分支）=====
        # 注意：此处避免 _normalize_goal_xy 的“标量→fallback_y”路径，确保与 single_batch 一致
        goal = self.goal_state_pos.to(device=device, dtype=dtype).view(1, 1, 2)  # (1,1,2)
        goal = goal.expand(B, 1, 2)  # (B,1,2)

        # ===== 世界坐标（保留 ultrafast 的 transform）=====
        if getattr(self, "transforms", None) is not None and self.transforms[model_id] is not None:
            offset = self.transforms[model_id].to(device=device, dtype=dtype).view(1, 1, 2)
            q_ws.copy_(q).add_(offset)
        else:
            q_ws.copy_(q)

        # ===== 代价（对齐 single_batch）=====
        # ctrl: ||u||^2 （逐时刻）
        runtime_cost.add_((us * us).sum(-1))  # (B,H)

        # pos_err: 500 * ||q - goal|| （逐时刻）
        tmp2.copy_(q).sub_(goal)  # (B,H,2)
        # 使用稳定的范数实现
        runtime_cost.add_(torch.sqrt((tmp2 * tmp2).sum(-1) + 1e-12), alpha=500.0)

        # ===== Free mask（对齐 single_batch：仅用于判定，不加到 cost）=====
        # 注意：使用 R + 1e-8 的清除距离，且取整段最小值 >= 0 作为 free
        sdf_seq, _ = self.sample_sdf_and_grad(q_ws, model_id=model_id)  # (B,H), (B,H,2)
        sdf_clear = sdf_seq - (R + 1e-8)
        free_mask = (sdf_clear.min(dim=1).values >= 0)

        # ===== 聚合 =====
        cost_sum = runtime_cost.sum(dim=1)  # 无单独终端罚项
        return cost_sum, q_ws, free_mask

    # ---------- SDF：纹理快路径 + 一次性梯度 ----------
    @torch.no_grad()
    def sample_sdf_and_grad(self, q_ws, model_id=0):
        """
        批量 SDF 采样（优先纹理；失败则回退 compute_sdf）
          输入: q_ws (B,H,2)
          输出: sdf (B,H), grad (B,H,2)
        """
        device = q_ws.device
        B, H, _ = q_ws.shape

        if device.type == "cpu":
            env = self.env_models[model_id]
            q_flat = q_ws.reshape(-1, 2)
            sdf = env.compute_sdf(q_flat, reshape_shape=(B, H))  # (B,H)

            grad = torch.zeros(B, H, 2, device=q_ws.device, dtype=q_ws.dtype)
            tau = 0.05
            band = sdf.abs() < tau
            if band.any():
                eps = 1e-3
                qx = q_flat.clone()
                qx[:, 0] += eps
                qy = q_flat.clone()
                qy[:, 1] += eps
                sdf_xp = env.compute_sdf(qx, reshape_shape=(B, H))
                sdf_yp = env.compute_sdf(qy, reshape_shape=(B, H))
                gx = (sdf_xp - sdf) / eps
                gy = (sdf_yp - sdf) / eps
                grad[..., 0][band] = gx[band]
                grad[..., 1][band] = gy[band]
            return sdf, grad

        # 1) 优先尝试纹理（single grid_sample 得到 sdf/gx/gy）
        try:
            self.build_sdf_texture(model_id=model_id)  # 已构建时直接返回
            if getattr(self, "_sdf_tex", None) is not None:
                tex = self._sdf_tex  # [1,3,Hm,Wm], float32, CUDA/MPS/CPU
                meta = self._sdf_tex_meta
                x0, y0, res = float(meta["x0"]), float(meta["y0"]), float(meta["res"])
                Hm, Wm = int(meta["H"]), int(meta["W"])

                x = q_ws[..., 0]
                y = q_ws[..., 1]
                # align_corners=True 的 [-1,1] 归一化
                xn = (x - x0) / (res * (Wm - 1) + 1e-12) * 2.0 - 1.0
                yn = (y - y0) / (res * (Hm - 1) + 1e-12) * 2.0 - 1.0
                grid = torch.stack((xn, yn), dim=-1).view(B, H, 1, 2).to(tex.device)

                texB = tex.expand(B, -1, -1, -1)  # [B,3,Hm,Wm]
                out = F.grid_sample(texB, grid, mode="bilinear", padding_mode="border", align_corners=True)  # [B,3,H,1]
                out = out.squeeze(-1)  # [B,3,H]
                sdf = out[:, 0, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                gx = out[:, 1, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                gy = out[:, 2, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                grad = torch.stack((gx, gy), dim=-1)  # (B,H,2)
                return sdf, grad
        except Exception:
            pass

        # 2) 回退：环境 compute_sdf + 窄带差分
        env = self.env_models[model_id]
        q_flat = q_ws.reshape(-1, 2)
        sdf = env.compute_sdf(q_flat, reshape_shape=(B, H))  # (B,H)

        grad = torch.zeros(B, H, 2, device=q_ws.device, dtype=q_ws.dtype)
        tau = 0.05
        band = sdf.abs() < tau
        if band.any():
            eps = 1e-3
            qx = q_flat.clone()
            qx[:, 0] += eps
            qy = q_flat.clone()
            qy[:, 1] += eps
            sdf_xp = env.compute_sdf(qx, reshape_shape=(B, H))
            sdf_yp = env.compute_sdf(qy, reshape_shape=(B, H))
            gx = (sdf_xp - sdf) / eps
            gy = (sdf_yp - sdf) / eps
            grad[..., 0][band] = gx[band]
            grad[..., 1][band] = gy[band]
        return sdf, grad

    def _get_cpu_bufs(self, B, H, device, dtype):
        if not hasattr(self, "_cpu_bufs"):
            self._cpu_bufs = {}
        key = (B, H, device, dtype)
        bufs = self._cpu_bufs.get(key)
        if bufs is None:
            bufs = {
                "q": torch.empty((B, H, 2), device=device, dtype=dtype),
                "dq": torch.empty((B, H, 2), device=device, dtype=dtype),
                "ddq": torch.empty((B, H, 2), device=device, dtype=dtype),
                "q_ws": torch.empty((B, H, 2), device=device, dtype=dtype),
                "rc": torch.zeros((B, H), device=device, dtype=dtype),
                "tmp2": torch.empty((B, H, 2), device=device, dtype=dtype),
                "vbar": torch.empty((B, H, 2), device=device, dtype=dtype),
            }
            self._cpu_bufs[key] = bufs
        else:
            bufs["rc"].zero_()  # 需要清零的只清零
        return bufs

    # ---------- SDF 纹理构建（一次性） ----------
    @torch.no_grad()
    def build_sdf_texture(self, model_id=0, force_rebuild: bool = False, res: float = 0.01):
        """
        把环境 SDF 预烘焙为纹理（sdf, gx, gy） → [1,3,Hm,Wm] float32
        - 若已构建且非 force_rebuild：直接返回
        - env 需能在矩形边界上 compute_sdf
        """
        if hasattr(self, "_sdf_tex") and (self._sdf_tex is not None) and (not force_rebuild):
            return

        env0 = self.env_models[model_id]
        # 边界/网格分辨率：优先来自 env，找不到则用默认
        try:
            x_min, x_max, y_min, y_max = env0.bounds  # 期望 env0.bounds = (xmin,xmax,ymin,ymax)
        except Exception:
            # 默认边界（请按你的地图调整）
            x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0

        Wm = max(2, int(round((x_max - x_min) / res)) + 1)
        Hm = max(2, int(round((y_max - y_min) / res)) + 1)

        # 构网格坐标
        xs = torch.linspace(x_min, x_max, Wm)
        ys = torch.linspace(y_min, y_max, Hm)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Hm,Wm]
        P = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)  # [Hm*Wm,2]

        # 计算 SDF
        sdf = env0.compute_sdf(P, reshape_shape=(Hm, Wm)).to(torch.float32)  # [Hm,Wm], float32 CPU/GPU
        # 中心差分梯度（边界单侧）
        gx = torch.zeros_like(sdf)
        gy = torch.zeros_like(sdf)
        gx[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2 * res)
        gy[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2 * res)
        gx[:, 0] = (sdf[:, 1] - sdf[:, 0]) / res
        gx[:, -1] = (sdf[:, -1] - sdf[:, -2]) / res
        gy[0, :] = (sdf[1, :] - sdf[0, :]) / res
        gy[-1, :] = (sdf[-1, :] - sdf[-2, :]) / res

        tex = torch.stack((sdf, gx, gy), dim=0).unsqueeze(0)  # [1,3,Hm,Wm]

        # 放到与模型一致的设备（优先 CUDA / MPS）
        prefer = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tex = tex.to(device=prefer, dtype=torch.float32)

        self._sdf_tex = tex  # [1,3,Hm,Wm]
        self._sdf_tex_meta = {"x0": x_min, "y0": y_min, "res": float(res), "H": int(Hm), "W": int(Wm)}
        return
    @torch.no_grad()
    def get_sdf_grid(self, model_id=0, res: float = 0.01):
        env0 = self.env_models[model_id]
        try:
            x_min, x_max, y_min, y_max = env0.bounds
        except Exception:
            x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0

        Wm = max(2, int(round((x_max - x_min) / res)) + 1)
        Hm = max(2, int(round((y_max - y_min) / res)) + 1)

        xs = torch.linspace(x_min, x_max, Wm)
        ys = torch.linspace(y_min, y_max, Hm)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        P = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        sdf = env0.compute_sdf(P, reshape_shape=(Hm, Wm))
        return sdf, (x_min, x_max, y_min, y_max), res

    @torch.no_grad()
    def _reverse_diffusion_step(
            self,
            model_id: int,
            i: int,
            traj_bars_i: torch.Tensor,
            max_resample: int = 500,
            free_ratio_target: float = 1,
            enable_k_selection=True,
    ):
        """
        Reverse process for single model
        """
        model_params = self.models[model_id]
        env_model = self.env_models[model_id]

        sigma_i = model_params['sigmas'][i].item()
        # temp_step = np.interp(sigma_i, [model_params['sigmas'][-1].item(), model_params['sigmas'][0].item()], [model_params['params']['temp_sample'] * 0.5, model_params['params']['temp_sample']])
        temp_step = model_params['params']['temp_sample']
        n_samples = model_params['params']['n_samples']
        attempt = 0

        while attempt < max_resample:
            # recover the trajectory
            if traj_bars_i.shape[0] != n_samples:
                traj_bars_i = traj_bars_i[:1].repeat(n_samples, 1, 1)
            traj_i = traj_bars_i.clone()
            # traj_i = traj_bars_i * torch.sqrt(model_params['alphas_bar'][i])

            #  Monte Carlo Sample from q_i velocity
            eps_u = torch.randn((n_samples, model_params['params']['horizon'], self.robot.q_dim * 2),
                                device=self.device)
            traj_0s = torch.clamp(eps_u * model_params['sigmas'][i] + traj_bars_i, -1, 1)

            # evaluate samples
            qs = []
            actions = traj_0s[..., self.robot.q_dim:]
            costs, q_seq, free_mask = self._rollout_single_batch_new2_ultrafast(model_id, self.state_inits, actions)
            # costs, q_seq, free_mask = self.rollout_ultrafast(
            #     model_id, self.state_inits, actions, use_amp=False
            # )
            traj_0s[..., :self.robot.q_dim] = q_seq
            free_ratio = free_mask.float().mean().item()
            if i >= 2 and free_ratio > 0:
                break
            elif i < 2 and free_ratio == free_ratio_target:
                break

            attempt += 1

        if not free_mask.any():
            raise RuntimeError(f"No collision-free sample after {attempt} resamples")

        cost_mean = costs.mean()
        cost_std = costs.std().clamp(min=1e-4)
        logp0 = -(costs - cost_mean) / cost_std / temp_step
        logp0[~free_mask] = -torch.inf

        # soft_constraint_grad = torch.zeros(traj_i.shape[0], traj_i.shape[1], self.robot.q_dim, device=traj_i.device)
        # if len(self.soft_constraints[model_id]) > 0:
        #     lambda_c = 1e-5
        #     k = 10
        #     # Reshape for proper broadcasting
        #     traj_pos = traj_i[..., :self.robot.q_dim].unsqueeze(2)
        #     if self._N > k and enable_k_selection:
        #         # select k
        #         B, H, q_dim = traj_pos.shape[0], traj_pos.shape[1], self.robot.q_dim
        #         traj_flat = traj_pos.reshape(-1, q_dim)  # [(B·H), q_dim]
        #         dist2_full = torch.cdist(traj_flat, self._centers_flat)  # [(B·H), N]
        #
        #         dmin, _ = dist2_full.min(dim=0)
        #         _, idx = torch.topk(dmin, k=k, largest=False)
        #         centers = self._centers_flat[idx].view(1, 1, k, -1)
        #         radii = self._radii_flat[idx].view(1, 1, k)
        #         t0 = self._t0[idx]
        #         t1 = self._t1[idx]
        #     else:
        #         centers = self._centers_flat.view(1, 1, self._N, -1)
        #         radii = self._radii_flat.view(1, 1, self._N)
        #         t0 = self._t0
        #         t1 = self._t1
        #
        #     diff = traj_pos - centers  # [B, H, N, q_dim]
        #     dist = diff.norm(dim=-1, keepdim=True)  # [B, H, N, 1]
        #
        #     idx = torch.arange(traj_pos.shape[1], device=self.device).view(1, traj_pos.shape[1], 1)  # [1, H, 1]
        #     mask_time = ((idx >= t0) & (idx <= t1)).unsqueeze(-1)
        #     mask_dist = (dist < radii.unsqueeze(-1))  # [B, H, N, 1]
        #     mask = mask_time & mask_dist  # [B, H, N, 1]
        #
        #     grad = -diff / (dist + 1e-6) * mask  # [B, H, N, q_dim]
        #     soft_constraint_grad = grad.sum(dim=2) / lambda_c  # [B, H, q_dim]

        # --- ALM ---
        # eps_deadzone = 1e-3
        # g_all_raw = dist - radii.unsqueeze(-1)  # [B,H,N,1]
        # g_all = torch.relu(g_all_raw - eps_deadzone) * mask_time  # [B,H,N,1]
        # viol_mask = (g_all > 0).float()
        # grad_g_all = (diff / (dist + 1e-6)) * mask_time
        #
        # progress = i / (len(model_params['alphas_bar']) - 1 + 1e-9)
        # rho_min, rho_max = 0.5, 2.0
        # rho_alm = rho_min + (rho_max - rho_min) * (0.5 - 0.5 * torch.cos(torch.tensor(progress * 3.14159265, device=traj_i.device)))
        #
        # # dual variable
        # if model_id not in self._alm_mu or self._alm_mu[model_id].shape[1:3] != g_all.shape[1:3]:
        #     self._alm_mu[model_id] = torch.zeros((1, g_all.shape[1], g_all.shape[2], 1), device=traj_i.device, dtype=traj_i.dtype)
        # mu = self._alm_mu[model_id]
        #
        # if viol_mask.any():
        #     term_pen = (mu * grad_g_all + rho_alm * g_all * grad_g_all) * viol_mask  # [B,H,N,q]
        #     active_cnt = viol_mask.sum(dim=2, keepdim=True).clamp_min(1.0)  # [B,H,1,1]
        #     soft_constraint_grad = term_pen.sum(dim=2) / active_cnt.squeeze(-1)  # [B,H,q]
        #     max_gnorm = 0.25
        #     gnorm = soft_constraint_grad.norm(dim=-1, keepdim=True).clamp_min(1e-9)  # [B,H,1]
        #     scale = (max_gnorm / gnorm).clamp(max=1.0)
        #     soft_constraint_grad = soft_constraint_grad * scale
        # else:
        #     soft_constraint_grad = torch.zeros(traj_i.shape[0], traj_i.shape[1], self.robot.q_dim, device=traj_i.device)
        #
        # # --- dual update ---
        # mu_max = 10.0
        # if viol_mask.any():
        #     mean_pos = (g_all * viol_mask).sum(dim=0, keepdim=True) / viol_mask.sum(dim=0, keepdim=True).clamp(min=1.0)
        #     mu_new = (mu + rho_alm * mean_pos).clamp(min=0.0, max=mu_max)
        # else:
        #     decay = 0.9
        #     mu_new = mu * decay
        # self._alm_mu[model_id] = mu_new.detach()

        # use demonstration data
        if self.enable_demo:
            xref_logpds = torch.tensor([env_model.eval_xref_logpd(q) for q in qs], device=self.device)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = -(xref_logpds + env_model.rew_xref - cost_mean) / cost_std / model_params['params'][
                'temp_sample']
            demo_mask = logpdemo > logp0
            logp0 = torch.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / model_params['params']['temp_sample']

        # weighted trajectory
        weights = torch.nn.functional.softmax(logp0, dim=0)
        traj_bar = torch.einsum("n,nij->ij", weights, traj_0s)

        # calculate score function
        score = 1 / (1.0 - model_params['alphas_bar'][i]) * (
                - traj_i + torch.sqrt(model_params['alphas_bar'][i]) * traj_bar)

        # expanded_grad = torch.zeros_like(score)
        # expanded_grad[..., :self.robot.q_dim] = soft_constraint_grad
        # score += expanded_grad  # add soft constraints

        traj_i_m1 = (1 / torch.sqrt(model_params['alphas'][i]) * (
                traj_i + (1.0 - model_params['alphas_bar'][i]) * score)) / torch.sqrt(
            model_params['alphas_bar'][i - 1])

        best_idx = torch.where(free_mask)[0][costs[free_mask].argmin()]
        best_traj_sample = traj_0s[best_idx]

        if self.transforms is not None and model_id in self.transforms:
            best_traj_sample[..., :self.robot.q_dim] += self.transforms[model_id]

        return traj_i_m1, best_traj_sample, costs.mean().item()

    # def get_sdf_grid(self):
    #     if self._sdf_tex_meta is not None:
    #         return None, (self._sdf_tex_meta["x0"], self._sdf_tex_meta["y0"]), self._sdf_tex_meta["res"]
    #
    #     x0, y0 = float(self.limits[0, 0]), float(self.limits[0, 1])
    #     x1, y1 = float(self.limits[1, 0]), float(self.limits[1, 1])
    #     res = float(self.sdf_cell_size)
    #     Hm = int(np.ceil((y1 - y0) / res)) + 1
    #     Wm = int(np.ceil((x1 - x0) / res)) + 1
    #
    #     xs = torch.linspace(x0, x1, steps=Wm, **self.tensor_args)
    #     ys = torch.linspace(y0, y1, steps=Hm, **self.tensor_args)
    #     X, Y = torch.meshgrid(xs, ys, indexing='xy')
    #     P = torch.stack((X.T.reshape(-1), Y.T.reshape(-1)), dim=-1).view(-1, 1, 2)  # (Hm*Wm,1,2) 按 [i(row),j(col)] 顺序
    #     with torch.no_grad():
    #         sdf = self.compute_sdf(P, reshape_shape=(Hm, Wm))
    #
    #     return sdf, (x0, y0), res
    #
    # @torch.no_grad()
    # def build_sdf_texture(self):
    #     if self._sdf_tex is not None:
    #         return
    #
    #     sdf_grid, origin, res = self.get_sdf_grid()
    #     if sdf_grid is None:
    #         return
    #
    #     gy = 0.5 * (torch.roll(sdf_grid, -1, 0) - torch.roll(sdf_grid, 1, 0)) / res  # d/dy  -> 行方向
    #     gx = 0.5 * (torch.roll(sdf_grid, -1, 1) - torch.roll(sdf_grid, 1, 1)) / res  # d/dx  -> 列方向
    #     tex = torch.stack([sdf_grid, gx, gy], dim=0).unsqueeze(0).contiguous()  # [1,3,Hm,Wm]
    #
    #     self._sdf_tex = tex.to(self.tensor_args['device'])
    #     self._sdf_tex_meta = {"x0": origin[0], "y0": origin[1], "res": res,
    #                           "H": int(sdf_grid.shape[0]), "W": int(sdf_grid.shape[1])}
    #
    # @torch.no_grad()
    # def sample_sdf_and_grad(self, q_ws, model_id=0):
    #     """
    #     batch sample SDF
    #     q_ws: (B,H,2
    #     return:
    #       sdf  : (B,H)
    #       grad : (B,H,2)
    #     """
    #     device = q_ws.device
    #     try:
    #         self.build_sdf_texture()
    #         if self._sdf_tex is not None:
    #             tex = self._sdf_tex  # [1,3,Hm,Wm]
    #             meta = self._sdf_tex_meta
    #             x0, y0, res, Hm, Wm = meta["x0"], meta["y0"], meta["res"], meta["H"], meta["W"]
    #
    #             j = (q_ws[..., 0] - x0) / res
    #             i = (q_ws[..., 1] - y0) / res
    #             u = 2.0 * (j / (Wm - 1)) - 1.0
    #             v = 2.0 * (i / (Hm - 1)) - 1.0
    #             grid = torch.stack([u, v], dim=-1).unsqueeze(0)  # [1,B,H,2]
    #
    #             samp = torch.nn.functional.grid_sample(tex, grid, mode="bilinear", align_corners=True)  # [1,3,1,B,H]
    #             samp = samp.squeeze(2).squeeze(0).permute(1, 2, 0)  # -> (B,H,3)
    #             sdf = samp[..., 0]
    #             gx = samp[..., 1]
    #             gy = samp[..., 2]
    #             grad = torch.stack([gx, gy], dim=-1)
    #             return sdf, grad
    #     except Exception:
    #         pass
    #
    #     env_model = self.env_models[0]
    #     B, H, _ = q_ws.shape
    #     q_flat = q_ws.reshape(-1, 2)
    #     sdf = env_model.compute_sdf(q_flat, reshape_shape=(B, H))
    #     tau = 0.05
    #     band = sdf.abs() < tau
    #     grad = torch.zeros(B, H, 2, device=device)
    #     if band.any():
    #         eps = 1e-3
    #         qx = q_flat.clone()
    #         qx[:, 0] += eps
    #         qy = q_flat.clone()
    #         qy[:, 1] += eps
    #         sdf_xp = env_model.compute_sdf(qx, reshape_shape=(B, H))
    #         sdf_yp = env_model.compute_sdf(qy, reshape_shape=(B, H))
    #         gx = (sdf_xp - sdf) / eps
    #         gy = (sdf_yp - sdf) / eps
    #         grad[..., 0][band] = gx[band]
    #         grad[..., 1][band] = gy[band]
    #     return sdf, grad

    def _rollout_single_batch_new2(self, model_id, state_init, us):
        env = self.env_models[model_id]
        B, H, _ = us.shape
        device = us.device
        dt = self.robot.dt

        q_init = state_init.q.view(1, 1, 2).to(device).expand(B, 1, 2)
        us_flat = us.reshape(B * H, 2)  # (B*H, 2)
        q_init_expanded = q_init.expand(B, H, 2).reshape(B * H, 2)  # (B*H, 2)

        q_ws_flat = q_init_expanded
        if self.transforms is not None and model_id in self.transforms:
            q_ws_flat = q_init_expanded + self.transforms[model_id].to(device).view(1, -1)

        sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat.view(B * H, 1, 2), model_id=model_id)
        sdf_flat = sdf_flat.view(B * H)
        grad_flat = grad_flat.view(B * H, 2)

        R = mparams.robot_planar_disk_radius
        margin = 0.01
        tau = 0.05
        eta = 0.8

        h = sdf_flat - (R + margin)
        A = grad_flat
        u_flat = us_flat.clone()

        goodA = (A.norm(dim=-1) > 1e-9)
        need = (h < tau) & goodA

        b = - (eta / dt) * h
        lhs = (A * u_flat).sum(dim=-1)
        den = (A.pow(2).sum(dim=-1) + 1e-9)

        alpha = torch.zeros(B * H, device=device)
        alpha[need] = torch.clamp((b[need] - lhs[need]) / den[need], min=0.0)
        u_flat = u_flat + alpha.unsqueeze(-1) * A

        lhs2 = (A * u_flat).sum(dim=-1)
        bad = goodA & (lhs2 < b)
        alpha2 = torch.zeros(B * H, device=device)
        alpha2[bad] = torch.clamp(((b - lhs2) / den)[bad], min=0.0)
        u_flat = u_flat + alpha2.unsqueeze(-1) * A

        still_in = h < 0.0
        if still_in.any():
            step = (0.5 * (-h[still_in])).clamp_max(0.10)
            n = torch.nn.functional.normalize(A[still_in], dim=-1)
            q_ws_flat_new = q_ws_flat.clone()
            q_ws_flat_new[still_in] = q_ws_flat_new[still_in] + step.unsqueeze(-1) * n
            if self.transforms is not None and model_id in self.transforms:
                q_init_expanded[still_in] = q_ws_flat_new[still_in] - self.transforms[model_id].to(device).view(1, -1)
            else:
                q_init_expanded[still_in] = q_ws_flat_new[still_in]

        u_safe = u_flat.reshape(B, H, 2)
        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)

        # ========= Vertex Constraints ======== #

        C_list, R_list, TM_list = [], [], []
        if hasattr(self, "soft_constraints") and (model_id in self.soft_constraints):
            for sph in (self.soft_constraints[model_id] or []):
                C_cc = sph.qs.to(device)[:, :2]
                R_cc = sph.radii.to(device)
                TR = sph.traj_ranges.to(device).long()  # (n,2)
                n_cc = C_cc.shape[0]
                t_idx_full = torch.arange(H, device=device).view(1, H).expand(n_cc, -1)  # (n,H)
                TM_cc = (t_idx_full >= TR[:, 0:1]) & (t_idx_full < TR[:, 1:2])  # (n,H)

                if n_cc > 0:
                    C_list.append(C_cc)  # (n,2)
                    R_list.append(R_cc)  # (n,)
                    TM_list.append(TM_cc)  # (n,H)

        if len(C_list) > 0:
            C_all = torch.cat([c.view(-1, 2) for c in C_list], dim=0)  # (N,2)
            R_all = torch.cat([r.view(-1) for r in R_list], dim=0).view(1, -1)  # (1,N)
            TM_cat = torch.cat([tm if tm.dim() == 2 else tm.view(-1, H) for tm in TM_list], dim=0)  # (N,H)
            TM_all = TM_cat.T  # (H,N)

            q_ws_seq = q_seq
            if self.transforms is not None and model_id in self.transforms:
                offset = self.transforms[model_id].to(device)
                q_ws_seq = q_seq + offset.view(1, 1, -1)  # (B,H,2)

            # （B,H,N）
            diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
            dist = diff.norm(dim=-1).clamp_min(1e-6)  # (B,H,N)
            A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
            r_exp = (R_all + (R + 0.02))  # (1,N)
            h_s = dist - r_exp  # (B,H,N)

            # time window & Narrow
            active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
            tau_s = 0.08
            near = (h_s < tau_s) & active  # (B,H,N)

            if near.any():
                # Top-K
                K = 2
                dist_masked = torch.where(near, dist, torch.full_like(dist, 1e9))
                k = min(K, dist_masked.shape[-1])
                topk = torch.topk(dist_masked, k=k, dim=-1, largest=False)
                idx = topk.indices  # (B,H,K)

                sel_A = torch.gather(A_s, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))  # (B,H,K,2)
                sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

                # ---- C-5：优先级偏置 β（偶数先行/奇数让行），可叠加 CostConstraint 的 priority_weight ----
                base_beta = 0.05
                sign = +1.0
                if hasattr(self, "agent_id"):
                    sign = +1.0 if (self.agent_id % 2 == 0) else -1.0
                scale_pw = 1.0

                # 若注册了 CostConstraint，可用其 priority_weight 微调（可选）
                # if (hasattr(self, "soft_constraints") and model_id in self.vertex_cost_constraints):
                #     try:
                #         pw = float(self.soft_constraints[model_id].priority_weight)
                #         scale_pw = max(0.5, min(2.0, pw))
                #     except Exception:
                #         pass
                beta = sign * base_beta * scale_pw
                sel_b = - (eta / dt) * sel_h + beta  # (B,H,K)
                U = u_safe.clone()  # (B,H,2)

                for kk in range(k):
                    Ak = sel_A[:, :, kk, :]  # (B,H,2)
                    bk = sel_b[:, :, kk]  # (B,H)
                    lhs_k = (Ak * U).sum(dim=-1)  # (B,H)
                    viol = lhs_k < bk
                    if viol.any():
                        den_k = (Ak.pow(2).sum(dim=-1) + 1e-9)
                        alpha_k = torch.zeros_like(lhs_k)
                        alpha_k[viol] = (bk[viol] - lhs_k[viol]) / den_k[viol]
                        U = U + alpha_k.unsqueeze(-1) * Ak
                u_safe = U  #

        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)

        # ===== costs: runtime + terminal =====
        q_goal = self.goal_state_pos.to(device)
        dist_to_goal = (q_seq - q_goal.view(1, 1, -1)).norm(dim=2)  # (B,H)
        ctrl = (u_safe * u_safe).sum(dim=2)  # (B,H)

        runtime_cost = torch.zeros(B, H, device=device)
        runtime_cost += 1.0 * ctrl
        runtime_cost += 5 * dist_to_goal

        t_star = max(1, int(0.95 * H))
        t_idx = torch.arange(H, device=device).view(1, H)
        d0 = (state_init.q.to(device) - q_goal).norm().view(1, 1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)
        runtime_cost += 1.0 * (dist_to_goal - d_hat).pow(2)

        s = (t_idx.float() / t_star).clamp(max=1.0).view(1, H, 1)
        v_bar = (d0.view(1, 1, 1) / (t_star * dt + 1e-6)) * (4 * s * (1.0 - s))
        u_norm = u_safe.norm(dim=-1, keepdim=True)
        runtime_cost += 0.5 * (u_norm - v_bar).pow(2).squeeze(-1)

        r_goal = 0.10
        inside_goal_early = (dist_to_goal <= r_goal) & (t_idx < t_star)
        runtime_cost += 0.5 * ((t_star - t_idx).clamp(min=0).float() / H) * inside_goal_early.float()

        q_ws = q_seq
        if self.transforms is not None and model_id in self.transforms:
            offset = self.transforms[model_id].to(device)
            q_ws = q_seq + offset.view(1, 1, -1)

        sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
        sdf_seq = sdf_seq - (mparams.robot_planar_disk_radius + 0.02)
        pen = (-sdf_seq).clamp(min=0.0)
        runtime_cost += 5000.0 * pen.pow(2)

        # free_mask_wall = self._segment_free_mask(
        #     q_ws, env_model=env, robot_radius=mparams.robot_planar_disk_radius, step=0.03
        # )
        # free_mask = free_mask_wall

        free_mask = torch.ones(B, dtype=torch.bool, device=device)

        # terminal
        w_T = getattr(self, "w_T_base", 8)
        terminal_cost = w_T * dist_to_goal[:, -1]
        cost_sum = runtime_cost.sum(dim=1) + terminal_cost

        return cost_sum, q_seq, free_mask

    @torch.no_grad()
    def _rollout_single_batch_new2_ultrafast(self, model_id, state_init, us, use_amp: bool = False):
        """
        Ultra-fast rollout that matches _rollout_single_batch_new2's math & weights:
          - keeps original parameters/weights (distance=5.0, v_bar factor=4.0, TopK=2, 2-step CBF)
          - avoids redundant allocs via static/scratch buffers (per (B,H,device,dtype,model_id))
          - fixes broadcasting; no unnecessary CPU<->GPU sync (.item)
          - SDF path penalty squared in fp32 for stability (behavior preserved)
        Returns: cost_sum: (B,), q_seq: (B,H,2), free_mask: (B,)
        """
        device, dtype = us.device, us.dtype
        B, H, _ = us.shape
        dt = float(self.robot.dt)

        # ===== constants (aligned with original) =====
        R = float(mparams.robot_planar_disk_radius)
        margin = 0.01
        tau = 0.05
        eta = 0.8
        eps = 1e-9
        huge = 1e9
        tau_s = 0.08
        r_goal = 0.10
        w_T = getattr(self, "w_T_base", 8)  # keep type/behavior
        t_star = max(1, int(0.9 * H))
        sign = +1.0 if (getattr(self, "agent_id", 0) % 2 == 0) else -1.0
        beta = sign * 0.05  # same as original base_beta * scale_pw(=1)

        # ===== one-time caches =====
        if not hasattr(self, "_rollout_static"):   self._rollout_static = {}
        if not hasattr(self, "_rollout_tf_cache"): self._rollout_tf_cache = {}
        if not hasattr(self, "_rollout_scratch"):  self._rollout_scratch = {}

        s_key = (B, H, device, model_id)
        st = self._rollout_static.get(s_key)
        if st is None:
            t_idx = torch.arange(H, device=device)
            st = {
                "t_idx_i": t_idx.view(1, H),
                "t_idx_f": t_idx.float().view(1, H),
                "ones_B_bool": torch.ones(B, dtype=torch.bool, device=device),
            }
            self._rollout_static[s_key] = st
        t_idx_i, t_idx_f, ones_B_bool = st["t_idx_i"], st["t_idx_f"], st["ones_B_bool"]

        # transform offset cache: (1,1,2)
        tf_key = (model_id, device)
        offset = self._rollout_tf_cache.get(tf_key, None)
        if (self.transforms is not None) and (model_id in self.transforms) and (offset is None):
            self._rollout_tf_cache[tf_key] = self.transforms[model_id].to(device).contiguous().view(1, 1, -1)
            offset = self._rollout_tf_cache[tf_key]

        # scratch buffers
        sc_key = (B, H, device, dtype)
        sc = self._rollout_scratch.get(sc_key)
        if sc is None:
            sc = {
                # flat / batched
                "BH2": torch.empty((B * H, 2), device=device, dtype=dtype),
                "BH": torch.empty((B * H,), device=device, dtype=dtype),
                "BH_": torch.empty((B, H), device=device, dtype=dtype),
                "BH1": torch.empty((B, H, 1), device=device, dtype=dtype),
                "BH1b": torch.empty((B, H, 1), device=device, dtype=dtype),
                # reused vectors
                "A2": torch.empty((B * H,), device=device, dtype=dtype),
                "lhs": torch.empty((B * H,), device=device, dtype=dtype),
                "alpha": torch.empty((B * H,), device=device, dtype=dtype),
                # vertex constraints
                "min_vals": torch.empty((B, H), device=device, dtype=dtype),
                "min_idx": torch.empty((B, H), device=device, dtype=torch.long),
                "lhs_k": torch.empty((B, H), device=device, dtype=dtype),
                "den_k": torch.empty((B, H), device=device, dtype=dtype),
                "alpha_k": torch.empty((B, H), device=device, dtype=dtype),
                # cost stage
                "dist_to_goal": torch.empty((B, H), device=device, dtype=dtype),
                "ctrl": torch.empty((B, H), device=device, dtype=dtype),
                "runtime_cost": torch.empty((B, H), device=device, dtype=dtype),
                # bool masks
                "bool_BH": torch.empty((B * H,), device=device, dtype=torch.bool),
            }
            self._rollout_scratch[sc_key] = sc

        # helpers
        def add_offset(q):  # q(...,2)
            return q if offset is None else (q + offset.view(*([1] * (q.dim() - 1)), 2))

        def rm_offset(q):
            return q if offset is None else (q - offset.view(*([1] * (q.dim() - 1)), 2))

        # init states / views
        q0 = state_init.q.to(device=device, dtype=dtype).view(1, 1, 2).expand(B, 1, 2)
        us_flat = us.reshape(B * H, 2)

        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if (use_amp and torch.cuda.is_available()) else contextlib.nullcontext())

        with autocast_ctx:
            # ---------- 1) CBF-like projection (2-step, same as original) ----------
            q_init_exp = q0.expand(B, H, 2).reshape(B * H, 2)  # (B*H,2)
            q_ws_flat = add_offset(q_init_exp.view(B * H, 1, 2))  # (B*H,1,2)

            sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat, model_id=model_id)
            sdf_flat = sdf_flat.view(B * H)  # (B*H,)
            A = grad_flat.view(B * H, 2)  # (B*H,2)

            sc["A2"].copy_((A * A).sum(-1)).add_(eps)  # ||A||^2 + eps(=1e-9)
            h = sc["BH"].copy_(sdf_flat).add_(-(R + margin))  # (B*H,)
            u_flat = sc["BH2"].copy_(us_flat)  # clone-like but from scratch
            goodA = sc["bool_BH"]
            goodA.copy_((A * A).sum(-1) > 1e-18)  # ||A||>~1e-9 (squared)
            need = sc["bool_BH"].clone().copy_(h < tau)
            need &= goodA

            b = sc["BH"].copy_(h).mul_(-eta / dt)  # b = -(eta/dt)*h
            sc["lhs"].copy_((A * u_flat).sum(-1))  # A·u
            den = sc["A2"]  # reuse

            sc["alpha"].zero_()
            if need.any():
                num = (b - sc["lhs"])[need]
                sc["alpha"][need].copy_((num / den[need]).clamp(min=0.0))

            u_flat.add_(sc["alpha"].unsqueeze(-1) * A)

            # second correction (original has this)
            lhs2 = (A * u_flat).sum(-1)
            bad = goodA & (lhs2 < b)
            if bad.any():
                alpha2 = torch.zeros_like(lhs2)
                alpha2[bad] = ((b - lhs2)[bad] / den[bad]).clamp(min=0.0)
                u_flat = u_flat + alpha2.unsqueeze(-1) * A

            # push out if still inside
            still_in = (h < 0.0) & goodA
            if still_in.any():
                Ai = A[still_in]
                inv = torch.rsqrt((Ai * Ai).sum(-1, keepdim=True) + eps)
                n = Ai * inv
                step = (0.5 * (-h[still_in])).clamp_max(0.10)
                q_ws_corr = sc["BH2"].view(B * H, 2)
                q_ws_corr.copy_(q_ws_flat.view(B * H, 2))
                q_ws_corr[still_in].add_(step.unsqueeze(-1) * n)
                q_init_exp[still_in].copy_(rm_offset(q_ws_corr[still_in]))

            u_safe = u_flat.view(B, H, 2)
            q_seq = q0 + torch.cumsum(u_safe * dt, dim=1)

            # ---------- 2) Vertex soft-constraints (TopK=2, same as original) ----------
            if hasattr(self, "soft_constraints") and (model_id in self.soft_constraints):
                sph_list = self.soft_constraints[model_id] or []
                if len(sph_list) > 0:
                    C_list, R_list, TM_list = [], [], []
                    for sph in sph_list:
                        Cc = sph.qs.to(device=device, dtype=dtype)[:, :2].contiguous()
                        if Cc.numel() == 0: continue
                        C_list.append(Cc)
                        R_list.append(sph.radii.to(device=device, dtype=dtype).view(-1))
                        TR = sph.traj_ranges.to(device, non_blocking=True).long()
                        t_full = t_idx_i.expand(Cc.shape[0], -1)  # (n,H)
                        TM_list.append((t_full >= TR[:, :1]) & (t_full < TR[:, 1:2]))  # (n,H)

                    if len(C_list) > 0:
                        C_all = torch.cat(C_list, dim=0).contiguous()  # (N,2)
                        R_all = torch.cat(R_list, dim=0).contiguous().view(1, -1)  # (1,N)
                        TM_all = torch.cat([tm.view(-1, H) for tm in TM_list], 0).t().contiguous()  # (H,N)

                        q_ws_seq = add_offset(q_seq)  # (B,H,2)
                        diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
                        dist = diff.norm(dim=-1).clamp_(min=1e-6)  # (B,H,N)
                        A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
                        h_s = dist - (R_all + (R + 0.02))  # (B,H,N)

                        active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
                        near = (h_s < tau_s) & active  # (B,H,N)

                        if near.any():
                            # TopK by dist (as original)
                            dist_masked = torch.where(near, dist, torch.full_like(dist, huge))
                            K = min(2, dist_masked.shape[-1])
                            topk = torch.topk(dist_masked, k=K, dim=-1, largest=False)
                            idx = topk.indices  # (B,H,K)

                            sel_A = torch.gather(A_s, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))  # (B,H,K,2)
                            sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

                            U = u_safe.clone()
                            for kk in range(K):
                                Ak = sel_A[:, :, kk, :]  # (B,H,2)
                                hk = sel_h[:, :, kk]  # (B,H)
                                bk = -(eta / dt) * hk + beta  # (B,H)
                                lhs_k = (Ak * U).sum(-1)  # (B,H)
                                viol = lhs_k < bk
                                if viol.any():
                                    den_k = (Ak * Ak).sum(-1) + eps
                                    alpha_k = torch.zeros_like(lhs_k)
                                    alpha_k[viol] = ((bk - lhs_k)[viol] / den_k[viol]).clamp(min=0.0)
                                    U = U + alpha_k.unsqueeze(-1) * Ak
                            u_safe = U

                        # re-integrate once
                        q_seq = q0 + torch.cumsum(u_safe * dt, dim=1)

            # ---------- 3) Costs (weights exactly match original) ----------
            q_goal = self.goal_state_pos.to(device=device, dtype=dtype).view(1, 1, 2)
            sc["dist_to_goal"].copy_((q_seq - q_goal).norm(dim=2))  # (B,H)
            sc["ctrl"].copy_((u_safe * u_safe).sum(dim=2))  # (B,H)

            rc = sc["runtime_cost"]
            rc.copy_(sc["ctrl"]).add_(sc["dist_to_goal"] * 5.0)  # 1*ctrl + 5*dist  (original)

            # d_hat term
            w = torch.clamp(1.0 - t_idx_f / t_star, min=0.0)  # (1,H)
            d0 = (state_init.q.to(device=device, dtype=dtype).view(1, 1, 2) - q_goal).norm(dim=2)  # (1,1)
            d_hat = sc["BH_"]
            d_hat.copy_(w.expand_as(d_hat)).mul_(d0)  # (B,H)
            rc.add_((sc["dist_to_goal"] - d_hat).pow(2))  # 1.0 *

            # v_bar & speed regularization (factor 1.0)
            s = (t_idx_f / t_star).clamp(max=1.0)  # (1,H)
            v_bar = sc["BH1"]
            if v_bar.shape != (B, H, 1):
                sc["BH1"] = torch.empty((B, H, 1), device=device, dtype=dtype)
                v_bar = sc["BH1"]
            v_bar.copy_(d0.view(1, 1, 1).expand(B, H, 1))
            v_bar.div_(t_star * dt + 1e-6)
            v_bar.mul_(1.0 * s.view(1, H, 1) * (1.0 - s).view(1, H, 1))

            u_norm = sc["BH1b"]
            if u_norm.shape != (B, H, 1):
                sc["BH1b"] = torch.empty((B, H, 1), device=device, dtype=dtype)
                u_norm = sc["BH1b"]
            u_norm.copy_(u_safe.norm(dim=-1, keepdim=True))
            rc.add_(0.5 * (u_norm.squeeze(-1) - v_bar.squeeze(-1)).pow(2))

            # early-arrival penalty (unchanged)
            inside_goal_early = (sc["dist_to_goal"] <= r_goal) & (t_idx_i < t_star)
            rc.add_(0.5 * ((t_star - t_idx_i).clamp(min=0).float() / H) * inside_goal_early.float())

            # SDF path penalty (fp32-square, same weight 5000)
            # q_ws = q_seq (+ offset like original)
            q_ws = add_offset(q_seq)
            sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
            sdf_seq = sdf_seq - (R + 0.02)
            pen = sc["BH_"].copy_(sdf_seq).neg_().clamp_(min=0.0)  # = relu(-(sdf))
            pen32 = pen.to(torch.float32)
            rc.add_((5000.0 * (pen32 * pen32)).to(rc.dtype))

            terminal_cost = w_T * sc["dist_to_goal"][:, -1]
            cost_sum = rc.sum(dim=1) + terminal_cost

            free_mask = ones_B_bool

        return cost_sum, q_seq, free_mask

    def _rollout_us(self, model_id: int, state: torch.tensor, us: torch.tensor):
        """
        Args:
            model_id: Function that takes (state, action) and returns new state
            state: Initial state object (must have reward and pipeline_state attributes)
            us: Tensor of actions shape [T, action_dim]
        Returns:
            rews: Tensor of rewards shape [T]
            pipeline_states: List of pipeline states
        """
        # us = torch.ones_like(us) * torch.tensor([0.1, 0.0], device=self.device)
        env_model = self.env_models[model_id]

        costs = []
        pipeline_states = []
        if torch.is_tensor(us):
            us = [u for u in us]

        for u in us:
            state = self.robot.step(state, torch.from_numpy(u), env_model)
            pos_error = torch.norm(self.goal_state_pos - state.q)
            costs.append(state.collision_cost + state.control_cost + 2.0 * pos_error)
            # costs.append(state.cost)
            pipeline_states.append(state.pipeline_state)

        if torch.is_tensor(costs[0]):
            costs = torch.stack(costs)
        else:
            costs = torch.tensor(costs)

        return costs, pipeline_states

    def _rollout_double_batch(self, model_id, state_init, a_seq, v_max: float = 2.0, a_max: float = 2,
                              w_a: float = 1e-2, w_jerk: float = 1e-3):
        """
        Double-integrator batch rollout .
        a_seq : (B, H, q_dim)  accelaration
        return:
          cost_sum : (B,)
          q_seq    : (B, H, q_dim)
          free_mask: (B,)
          v_seq    : (B, H, q_dim)
        """
        env = self.env_models[model_id]
        B, H, q_dim = a_seq.shape
        dt = self.robot.dt

        q0 = state_init.q.view(1, 1, q_dim).to(a_seq.device).expand(B, 1, q_dim)
        v0 = state_init.q_dot.view(1, 1, q_dim).to(a_seq.device).expand(B, 1, q_dim)

        a_seq = a_seq.clamp(-a_max, a_max)  # (B,H,q)
        v_seq = (v0 + torch.cumsum(a_seq, dim=1) * dt).clamp(-v_max, v_max)
        q_seq = q0 + torch.cumsum(v_seq, dim=1) * dt

        pos_err = (self.goal_state_pos.view(1, 1, q_dim) - q_seq).norm(dim=2)  # (B,H)
        cost_track = 500.0 * pos_err
        cost_acc = w_a * (a_seq ** 2).sum(dim=(1, 2))
        if H >= 2:
            jerk = a_seq[:, 1:, :] - a_seq[:, :-1, :]
            cost_jerk = w_jerk * (jerk ** 2).sum(dim=(1, 2))
        else:
            cost_jerk = torch.zeros(B, device=a_seq.device)
        cost_sum = cost_track.sum(dim=1) + cost_acc + cost_jerk

        if env is not None:
            q_ws = q_seq
            if self.transforms is not None and model_id in self.transforms:
                offset = self.transforms[model_id].to(q_seq.device)
                q_ws = q_seq + offset.view(1, 1, -1)
            sdf = env.compute_sdf(q_ws.reshape(-1, q_dim)).view(B, H)
            sdf = sdf - (mparams.robot_planar_disk_radius + 1e-8)
            free_mask = (sdf.min(dim=1).values >= 0)
        else:
            free_mask = torch.ones(B, dtype=torch.bool, device=a_seq.device)

        return cost_sum, q_seq, free_mask

    def _rollout_single_batch(self, model_id, state_init, us):
        """
        Args
        ----
        Single-integrator batch rollout .
        us : (B, H, 2)  ‑‑
        Returns
        -------
        cost   : (B,)
        q_seq  : (B, H, 2)
        """
        env = self.env_models[model_id]
        B, H, _ = us.shape
        dt = self.robot.dt

        dq = us * dt  # (B,H,2)
        q0 = state_init.q.unsqueeze(0)  # (1,2)
        q_seq = torch.cumsum(dq, dim=1) + q0  # (B,H,2)

        q_goal = self.goal_state_pos.to(self.device)
        pos_err = (self.goal_state_pos - q_seq).norm(dim=2)  # (B,H)
        ctrl = (us * us).sum(dim=2)  # (B,H)
        cost = ctrl + 500 * pos_err  # (B,H)

        # 1) Progress cost: d_t follows linear schedule to t_star
        t_star = max(1, int(0.98 * H))  # plan to arrive near 98% horizon
        t_idx = torch.arange(H, device=self.device).view(1, H)  # (1,H)
        d_t = (q_seq - q_goal.view(1, 1, -1)).norm(dim=2)  # (B,H)
        d0 = (state_init.q.to(self.device) - q_goal).norm().view(1, 1)  # (1,1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)  # (1,H)
        w_prog = 0
        J_prog = w_prog * (d_t - d_hat).pow(2)  # (B,H)
        cost = cost + J_prog

        if env is not None:
            q_ws = q_seq  # default : already global
            if self.transforms is not None and model_id in self.transforms:
                offset = self.transforms[model_id].to(q_seq.device)  # (2,)
                q_ws = q_seq + offset.view(1, 1, -1)  # broadcast to (B,H,2)

            # q_ws = self._project_to_free_space(q_ws, env, mparams.robot_planar_disk_radius, max_iter=3, grad_clip=2)
            sdf = env.compute_sdf(q_ws.reshape(-1, self.robot.q_dim)).view(B, H)  # (B,H)
            sdf -= (mparams.robot_planar_disk_radius + 1e-8)
            pen = (-sdf).clamp(min=0)
            # cost += 5000 * pen.pow(2)
            free_mask = (sdf.min(dim=1).values >= 0)
            # free_mask = self._segment_free_mask(
            #     q_ws, env_model=env, robot_radius=mparams.robot_planar_disk_radius, step=0.03
            # )
        else:
            free_mask = torch.ones(B, dtype=torch.bool, device=us.device)

        return cost.sum(dim=1), q_seq, free_mask

    @torch.no_grad()
    def run_inference(
            self,
            contexts: List[dict] = None,
            hard_conds: Dict[int, dict] = None,
            cross_conds: Dict[Tuple[int, int], Tuple[int, int]] = None,
            model_ids: List[int] = None,
            n_samples: int = 100,
            batch_size: int = 32,
            return_chain: bool = False,
            soft_constraints=[],
            **diffusion_kwargs):
        """
        :param contexts:
        :param hard_conds:
        :param cross_conds: Keys are tuples of model indices and values are tuples of trajectory timesteps.
                            For example, {(0, 1): {0, 64}} means that the value of the state 0 in the trajectory
                            of model 0 needs to be the same as the value of the state 64 in the trajectory of model 1.
        :param n_samples:
        :param return_chain:
        :param diffusion_kwargs:
        :return:
            if return_chain is True，return the diffusion chain
            else return final optimized trajectory
        """
        contexts = deepcopy(contexts)
        hard_conds = deepcopy(hard_conds)
        cross_conds = deepcopy(cross_conds)
        if contexts is None:
            contexts = [None] * len(self.models)
        for m, c_dict in hard_conds.items():
            for k, v in c_dict.items():
                # k is the key of the condition, usually state index in the trajectory
                hard_conds[m][k] = einops.repeat(v, 'd -> b d', b=n_samples)

        results = {}
        chains = {}

        for model_id in self.models.keys():
            traj_n = torch.zeros((n_samples, self.models[model_id]['params']['horizon'], self.robot.q_dim * 2),
                                 device=self.device)

            B, H, _ = traj_n.shape
            # cold start soft constraints (acceleration)
            self.soft_constraints[model_id] = soft_constraints
            if len(self.soft_constraints[model_id]) > 0:
                self._prepare_soft_constraints(model_id, H, traj_n.device)

            # Reverse
            traj_i = traj_n
            model_chain = []
            costs = []

            best_traj = None
            best_cost = float('inf')

            for i in tqdm(range(self.models[model_id]['params']['n_diffusion_step'] - 1, -1, -1),
                          desc=f"Diffusing model {model_id}"):
                traj_i, sample_i, cost_i = self._reverse_diffusion_step(model_id, i, traj_i)
                if return_chain:
                    model_chain.append(traj_i)
                costs.append(cost_i)
                if cost_i < best_cost:
                    best_cost, best_traj = cost_i, sample_i

            if return_chain:
                chains[model_id] = torch.stack(model_chain)

            # evaluate final cost
            final_traj = chains[model_id][-1, ...]
            final_actions = self.robot.get_velocity(
                final_traj).cpu().numpy() if self.device != 'cpu' else self.robot.get_velocity(traj_i).numpy()
            costs_final, q = self._rollout_us(model_id, self.state_inits, final_actions)
            results[model_id] = {
                'trajectory': final_traj,
                'cost': costs_final.mean().item(),
                'cost_history': costs
            }

        if return_chain:
            return chains

        return results
