from __future__ import annotations
import torch
from typing import Any, Tuple, Dict, List, Union, Callable
import dataclasses
from collections.abc import Mapping, Sequence

TensorLike = Union[torch.Tensor, float, int, bool, None]
Pytree = Union[TensorLike, Tuple[Any, ...], List[Any], Dict[str, Any]]


# ---------- helpers ----------
def _is_dataclass_instance(x: Any) -> bool:
    return dataclasses.is_dataclass(x) and not isinstance(x, type)


def _is_sequence(x: Any) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes)) and not torch.is_tensor(x)


def _same_mapping_keys(a: Mapping, b: Mapping) -> bool:
    return set(a.keys()) == set(b.keys())


def _iter_mapping_items(a: Mapping):
    return a.items()


# ---------- core ----------
def _tree_copy_into_(dst: Any, src: Any) -> None:
    """
    递归把 src 的内容拷到 dst。容器结构必须匹配。
    - Tensor 叶子：严格校验 shape/dtype/device，调用 copy_。
    - Dataclass：按字段递归。
    - Mapping：键集合一致；对每个 key：
        * 若子项仍是容器/张量/数据类 -> 递归
        * 否则（普通 Python 叶子，如对象/None/数） -> 直接 dst[k] = src[k]
    - Sequence（list/tuple/...）：长度一致；
        * list 可对叶子原地赋值（dst[i] = src[i]）
        * tuple 不可原地赋值：若遇到 tuple 叶子，要求其元素同构并递归，否则建议改成 list
    其他叶子：忽略（视为常量，不拷贝）
    """
    # Tensor
    if torch.is_tensor(dst) and torch.is_tensor(src):
        if (dst.shape != src.shape) or (dst.dtype != src.dtype) or (dst.device != src.device):
            raise ValueError(
                "Shape/dtype/device mismatch:\n"
                f"  dst={dst.shape, dst.dtype, dst.device}\n"
                f"  src={src.shape, src.dtype, src.device}"
            )
        dst.copy_(src)
        return

    # Dataclass
    if _is_dataclass_instance(dst) and _is_dataclass_instance(src) and (type(dst) is type(src)):
        for f in dataclasses.fields(dst):
            _tree_copy_into_(getattr(dst, f.name), getattr(src, f.name))
        return

    # Mapping
    if isinstance(dst, Mapping) and isinstance(src, Mapping):
        if not _same_mapping_keys(dst, src):
            missing = set(dst.keys()) ^ set(src.keys())
            raise KeyError(f"Mapping keys mismatch: {missing}")
        for k, dst_v in _iter_mapping_items(dst):
            src_v = src[k]
            if isinstance(dst_v, Mapping) and isinstance(src_v, Mapping):
                _tree_copy_into_(dst_v, src_v)
            elif _is_sequence(dst_v) and _is_sequence(src_v):
                _tree_copy_into_(dst_v, src_v)
            elif torch.is_tensor(dst_v) and torch.is_tensor(src_v):
                _tree_copy_into_(dst_v, src_v)
            elif _is_dataclass_instance(dst_v) and _is_dataclass_instance(src_v) and (type(dst_v) is type(src_v)):
                _tree_copy_into_(dst_v, src_v)
            else:
                try:
                    dst[k] = src_v
                except Exception:
                    raise TypeError(f"Cannot assign into Mapping value at key {k}: {type(dst)}")
        return

    # Sequence
    if _is_sequence(dst) and _is_sequence(src):
        if len(dst) != len(src):
            raise ValueError(f"Sequence length mismatch: {len(dst)} vs {len(src)}")
        if isinstance(dst, list):
            for i, (d_i, s_i) in enumerate(zip(dst, src)):
                if isinstance(d_i, Mapping) and isinstance(s_i, Mapping):
                    _tree_copy_into_(d_i, s_i)
                elif _is_sequence(d_i) and _is_sequence(s_i):
                    _tree_copy_into_(d_i, s_i)
                elif torch.is_tensor(d_i) and torch.is_tensor(s_i):
                    _tree_copy_into_(d_i, s_i)
                elif _is_dataclass_instance(d_i) and _is_dataclass_instance(s_i) and (type(d_i) is type(s_i)):
                    _tree_copy_into_(d_i, s_i)
                else:
                    dst[i] = s_i
            return
        else:
            for d_i, s_i in zip(dst, src):
                if (isinstance(d_i, Mapping) and isinstance(s_i, Mapping)) or \
                        (_is_sequence(d_i) and _is_sequence(s_i)) or \
                        (torch.is_tensor(d_i) and torch.is_tensor(s_i)) or \
                        (_is_dataclass_instance(d_i) and _is_dataclass_instance(s_i) and (type(d_i) is type(s_i))):
                    _tree_copy_into_(d_i, s_i)
                else:
                    if isinstance(dst, tuple):
                        raise TypeError("Cannot assign into tuple leaf; please use list for mutable containers.")
            return

    return


def _tree_map(fn: Callable[[Any], Any], tree: Any) -> Any:
    # Dataclass
    if _is_dataclass_instance(tree):
        mapped = {f.name: _tree_map(fn, getattr(tree, f.name)) for f in dataclasses.fields(tree)}
        return type(tree)(**mapped)

    # Mapping
    if isinstance(tree, Mapping):
        return {k: _tree_map(fn, v) for k, v in _iter_mapping_items(tree)}

    # Sequence
    if _is_sequence(tree):
        items = [_tree_map(fn, x) for x in tree]
        try:
            return type(tree)(items)
        except Exception:
            return list(items)

    # Leaf
    return fn(tree)


def _tree_zip(a: Any, b: Any) -> Any:
    # Dataclass
    if _is_dataclass_instance(a) and _is_dataclass_instance(b) and (type(a) is type(b)):
        zipped = {f.name: _tree_zip(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)}
        return type(a)(**zipped)

    # Mapping
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if not _same_mapping_keys(a, b):
            missing = set(a.keys()) ^ set(b.keys())
            raise KeyError(f"Mapping keys mismatch: {missing}")
        return {k: _tree_zip(a[k], b[k]) for k in a.keys()}

    # Sequence
    if _is_sequence(a) and _is_sequence(b):
        if len(a) != len(b):
            raise ValueError(f"Sequence length mismatch: {len(a)} vs {len(b)}")
        items = [_tree_zip(x, y) for x, y in zip(a, b)]
        try:
            return type(a)(items)
        except Exception:
            return list(items)

    # Leaves
    return (a, b)


def _clone_static_like(x: Any) -> Any:
    if torch.is_tensor(x):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    if _is_dataclass_instance(x):
        return type(x)(**{f.name: _clone_static_like(getattr(x, f.name)) for f in dataclasses.fields(x)})

    if isinstance(x, Mapping):
        return {k: _clone_static_like(v) for k, v in _iter_mapping_items(x)}

    if _is_sequence(x):
        items = [_clone_static_like(v) for v in x]
        try:
            return type(x)(items)
        except Exception:
            return list(items)

    return x


class RolloutAccelerator:
    """
    Wrap a rollout function with torch.compile and optional CUDA Graph capture.
    Works with arbitrary pytree (dict/list/tuple) of Tensors as inputs/outputs.

    Usage:
        acc = RolloutAccelerator(rollout_fn, example_args=(...),
                                 use_amp=True, amp_dtype=torch.bfloat16)
        acc.prepare()   # warmup + (if CUDA) graph capture
        out = acc.run(*actual_args)   # fast replay
    """

    def __init__(
            self,
            fn: Callable[..., Pytree],
            example_args: Tuple[Pytree, ...],
            *,
            compile_mode: str = "max-autotune",
            fullgraph: bool = True,
            use_amp: bool = True,
            amp_dtype: torch.dtype = torch.bfloat16,
            warmup_steps: int = 3,
            cuda_graph: bool = True,
    ) -> None:
        self.fn = fn
        self.example_args = example_args
        self.compile_mode = compile_mode
        self.fullgraph = fullgraph
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.warmup_steps = max(1, warmup_steps)
        self.enable_cuda_graph = cuda_graph and torch.cuda.is_available()

        # compile
        self.compiled = torch.compile(fn, mode=compile_mode, fullgraph=fullgraph)

        # buffers & graph handles
        self.static_args: Tuple[Pytree, ...] = ()
        self.static_out: Pytree | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self._prepared = False
        self._device = self._infer_device(example_args)

    def _infer_device(self, args: Tuple[Pytree, ...]) -> torch.device:
        dev = None

        def check(x):
            nonlocal dev
            if torch.is_tensor(x):
                dev = x.device if dev is None else dev

        for a in args:
            _tree_map(check, a)
        return dev or torch.device("cpu")

    def _amp_context(self):
        if self._device.type == "cuda" and self.use_amp:
            return torch.autocast("cuda", dtype=self.amp_dtype)

        class _Null:
            def __enter__(self): return None

            def __exit__(self, *exc): return False

        return _Null()

    @torch.inference_mode()
    def prepare(self) -> None:
        """
        Allocate static buffers, warm up compiled graph, and capture CUDA Graph if enabled.
        """
        if self._prepared:
            return

        # 1) make persistent static input buffers
        self.static_args = tuple(_tree_map(_clone_static_like, a) for a in self.example_args)

        # 2) warmup runs (on static buffers) to let Inductor specialize
        def _call_on(static_args: Tuple[Pytree, ...]) -> Pytree:
            with self._amp_context():
                return self.compiled(*static_args)

        # fill static args with example data
        for dst, src in zip(self.static_args, self.example_args):
            _tree_copy_into_(dst, src)

        # a few warmups (no capture)
        for _ in range(self.warmup_steps):
            _ = _call_on(self.static_args)

        # 3) allocate static output buffers by running once
        out = _call_on(self.static_args)
        self.static_out = _tree_map(_clone_static_like, out)

        # 4) CUDA Graph capture (GPU only)
        if self.enable_cuda_graph and self._device.type == "cuda":
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            # ensure no new allocations during capture: reuse static buffers
            with torch.cuda.graph(self.graph):
                out2 = _call_on(self.static_args)
                # copy into static_out in-place to keep stable storage
                _tree_copy_into_(self.static_out, out2)
            torch.cuda.synchronize()

        self._prepared = True

    @torch.inference_mode()
    def run(self, *args: Pytree) -> Pytree:
        """
        Copy user inputs into static buffers, replay (or call compiled), return views to static outputs.
        """
        if not self._prepared:
            raise RuntimeError("Call prepare() before run().")

        # copy inputs into static buffers
        for dst, src in zip(self.static_args, args):
            _tree_copy_into_(dst, src)

        if self.graph is not None:
            self.graph.replay()
            out = self.static_out
        else:
            # CPU or CUDA w/o graphs
            with self._amp_context():
                out = self.compiled(*self.static_args)
                # keep return semantics consistent: copy into static_out
                _tree_copy_into_(self.static_out, out)
                out = self.static_out

        return out  # static buffers: do not .detach()/.cpu() here to avoid sync
