import torch
import torch.nn.functional as F
import math

def smooth_weight(
        step: int, total_steps: int,
        start: float = 1.0,
        end: float = 0.1,
        schedule: str = "cosine"
):
    step = max(0, min(step, total_steps))
    if schedule == "cosine":
        alpha = 0.5 * (1.0 + torch.cos(torch.tensor(step / total_steps * math.pi)))
        w = end + (start - end) * alpha
    elif schedule == "poly":
        gamma = 2.0
        alpha = (1.0 - step / max(1, total_steps)) ** gamma
        w = end + (start - end) * alpha
    else:
        w = torch.tensor(start)
    return float(w)


def smooth_loss_acc(trajs_pos, dt=0.1):
    vel = torch.diff(trajs_pos, dim=-2) / dt  # (B,H-1,D)
    acc = torch.diff(vel, dim=-2) / dt  # (B,H-2,D)
    return (acc ** 2).sum(dim=-1).mean(dim=-1)  # (B,)


def smooth_loss_jerk(trajs_pos, dt=0.1):
    vel = torch.diff(trajs_pos, dim=-2) / dt
    acc = torch.diff(vel, dim=-2) / dt
    jerk = torch.diff(acc, dim=-2) / dt  # (B,H-3,D)
    return (jerk ** 2).sum(dim=-1).mean(dim=-1)  # (B,)


def smooth_loss_laplacian(trajs_pos):
    lap = trajs_pos[..., 2:, :] - 2 * trajs_pos[..., 1:-1, :] + trajs_pos[..., :-2, :]
    return (lap ** 2).sum(dim=-1).mean(dim=-1)  # (B,)


# smooth as soft constraints
def apply_smooth_guidance_to_score(
        score: torch.Tensor,
        trajs_pos: torch.Tensor,
        step: int,
        total_steps: int,
        dt: float = 0.1,
        w_start: float = 1.0,
        w_end: float = 0.1,
        modes: tuple = ("lap", "acc"),
        weights: tuple = (1.0, 0.25),
        schedule: str = "cosine",
) -> torch.Tensor:
    added_score_batch = False
    if score.ndim == 2:              # (H, D_s) -> (1, H, D_s)
        score = score.unsqueeze(0)
        added_score_batch = True
    assert score.ndim == 3, "score must be (B,H,D_s) or (H,D_s)"

    B, H, D_s = score.shape

    if trajs_pos.ndim == 2:          # (H, pos_dim) -> (1, H, pos_dim)
        trajs_pos = trajs_pos.unsqueeze(0)
    assert trajs_pos.ndim == 3, "trajs_pos must be (B,H,pos_dim) or (H,pos_dim)"

    Bp, Hp, pos_dim = trajs_pos.shape
    assert Hp == H, f"time dim mismatch: trajs_pos H={Hp} vs score H={H}"
    assert D_s >= pos_dim, f"score last dim {D_s} must cover pos_dim {pos_dim}"

    if Bp == 1 and B > 1:
        trajs_pos_eff = trajs_pos.expand(B, H, pos_dim)
    elif Bp == B:
        trajs_pos_eff = trajs_pos
    else:
        raise ValueError(f"Batch mismatch: score B={B}, trajs_pos B={Bp}")

    effective_step = total_steps - step
    lam = float(smooth_weight(effective_step, total_steps,
                              start=w_start, end=w_end, schedule=schedule))

    with torch.enable_grad():
        x = trajs_pos_eff.detach().requires_grad_(True)  # (B,H,pos_dim)
        loss = 0.0
        for m, w in zip(modes, weights):
            w = float(w)
            if m == "lap":
                loss = loss + w * smooth_loss_laplacian(x).mean()
            elif m == "acc":
                loss = loss + w * smooth_loss_acc(x, dt=dt).mean()
            elif m == "jerk":
                loss = loss + w * smooth_loss_jerk(x, dt=dt).mean()
            else:
                raise ValueError(f"unknown smooth mode: {m}")
        (grad_x,) = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)  # (B,H,pos_dim)

    out = score.clone()
    out[..., :pos_dim] = out[..., :pos_dim] - lam * grad_x

    if added_score_batch:
        out = out.squeeze(0)
    return out


# smooth as hard constraints
def _d2t_d2_matvec(u):  #
    B, H, D = u.shape
    k = u.new_tensor([1., -4., 6., -4., 1.]).view(1, 1, 5)  # (1,1,5)
    x = u.permute(0, 2, 1).contiguous()  # (B,D,H)
    y = F.conv1d(x, k.expand(D, 1, 5), padding=2, groups=D)  # (B,D,H)
    return y.permute(0, 2, 1).contiguous()  # (B,H,D)


def _A_matvec(u, mu):
    return u + mu * _d2t_d2_matvec(u)


def project_controls_smooth(b, mu=1e-3, iters=8, tol=1e-6):
    """
    Solve (I + mu * D^T D) x = b  with fixed-iteration masked CG smoothing.
    Torch-compile safe (no data-dependent break).
    """
    x = b.clone()
    r = b - _A_matvec(x, mu)
    p = r.clone()
    rs_old = (r * r).sum(dim=(1, 2), keepdim=True)  # (B,1,1)
    active_mask = torch.ones_like(rs_old, dtype=torch.bool)  # (B,1,1)

    for _ in range(iters):
        Ap = _A_matvec(p, mu)
        denom = (p * Ap).sum(dim=(1, 2), keepdim=True) + 1e-12
        alpha = rs_old / denom
        x = x + alpha * p * active_mask
        r = r - alpha * Ap * active_mask

        rs_new = (r * r).sum(dim=(1, 2), keepdim=True)
        converged = torch.sqrt(rs_new) < tol
        active_mask = active_mask & (~converged)

        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new

    return x
