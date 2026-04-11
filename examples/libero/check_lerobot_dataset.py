#!/usr/bin/env python3
"""检查本地或 Hub 上的 LeRobot 格式 LIBERO 数据集是否可读、字段是否齐全。

用于排查「images 文件夹为空」等问题：LeRobot 常把图像以视频/Parquet 存储，
本脚本通过 ``LeRobotDataset`` 直接读样本，确认像素数据是否真实存在。

用法（需在环境中安装 ``lerobot``）::

    uv run python examples/libero/check_lerobot_dataset.py \\
        --repo-id physical-intelligence/libero

    # 指定本地缓存目录（若数据集仅在本地）::
    # 设置环境变量 HF_HOME 或让 huggingface 已下载该 repo

    uv run python examples/libero/check_lerobot_dataset.py \\
        --repo-id your-org/your_combined_libero \\
        --num-probe 500 \\
        --action-horizon 10

    # 固定查看 episode=1 并导出 MP4（仅加载该 episode，较快）::
    uv run python examples/libero/check_lerobot_dataset.py \\
        --repo-id physical-intelligence/libero \\
        --root /path/to/lerobot_dataset_root \\
        --inspect-episode 1 \\
        --export-mp4 /tmp/ep1.mp4
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
from collections import Counter

import numpy as np
import tyro

logger = logging.getLogger("check_lerobot")


@dataclasses.dataclass
class Args:
    # Hugging Face 风格 repo id，例如 physical-intelligence/libero
    repo_id: str = "pi0_fewshot"

    # 本地数据集根目录（含 meta/data/videos）。不设则使用 HF_LEROBOT_HOME 下缓存。
    root: str = "/seu_share2/home/linli/213221101/.cache/huggingface/lerobot/pi0_fewshot"

    # 与训练时一致的 action chunk 长度（影响 delta_timestamps）
    action_horizon: int = 10

    # 随机抽检的帧数（0 表示只检查第 0 帧）
    num_probe: int = 32

    # 若设为 True，额外扫描全表统计 task_index / episode_index（大数据集较慢）
    scan_all_task_stats: bool = False

    # 固定检查并打印该 episode_index 的任务与帧信息（默认 1）
    inspect_episode: int = 0

    # 将该 episode 的主视角（及可选手腕）导出为 MP4（须为 .mp4 文件路径，或已存在的目录则写入 inspect_episode_{N}.mp4）
    export_mp4: str | None = None

    # 导出时若存在手腕视角，是否与主视角横向拼接（高度对齐）
    export_stack_wrist: bool = True

    # 仅加载指定 episode 列表以做抽检（逗号分隔，如 "0,1,2"）。不设则加载全量（大库很慢）
    episodes_filter: str | None = None


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _summarize_array(name: str, arr: np.ndarray | None) -> str:
    if arr is None:
        return f"{name}: None"
    if arr.size == 0:
        return f"{name}: empty shape={arr.shape}"
    return (
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={float(np.min(arr)):.4f} max={float(np.max(arr)):.4f} mean={float(np.mean(arr)):.4f}"
    )


def _infer_action_key(meta) -> str:
    features = getattr(meta, "features", {}) or {}
    if "actions" in features:
        return "actions"
    if "action" in features:
        return "action"
    for key in features:
        if "action" in key.lower():
            return key
    # Fall back to the historical default.
    return "actions"


def _infer_visual_candidates(meta) -> list[str]:
    features = getattr(meta, "features", {}) or {}
    keys = list(features.keys())

    # Prefer explicit common names first.
    preferred = ["image", "observation.image", "rgb", "front_image", "wrist_image"]
    ordered = [k for k in preferred if k in features]

    # Then append all image/video-like feature keys.
    for key, ft in features.items():
        dtype = str(ft.get("dtype", "")).lower() if isinstance(ft, dict) else ""
        key_l = key.lower()
        if key in ordered:
            continue
        if dtype in {"image", "video"} or "image" in key_l or "rgb" in key_l or "camera" in key_l:
            ordered.append(key)

    # Final fallback: keep deterministic order from metadata.
    if not ordered:
        ordered = keys
    return ordered


def _infer_state_key(meta) -> str | None:
    features = getattr(meta, "features", {}) or {}
    if "state" in features:
        return "state"
    if "observation.state" in features:
        return "observation.state"
    for key in features:
        key_l = key.lower()
        if "state" in key_l or "proprio" in key_l:
            return key
    return None


def _load_metadata(lerobot_dataset, repo_id: str, root: str | None):
    if root:
        try:
            return lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=pathlib.Path(root))
        except TypeError:
            return lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    return lerobot_dataset.LeRobotDatasetMetadata(repo_id)


def _resolve_task_text(meta, task_index: int) -> str:
    """将 task_index 映射为可读任务文本。

    LeRobot 版本差异：
    - 旧版：``dict[int, str]``
    - v3+ 常见：`pandas.DataFrame`，**index 为自然语言 task**，列 ``task_index`` 为整数
      （与 ``PromptFromLeRobotTask`` 一致，见 ``openpi.transforms``）。
    """
    tasks = getattr(meta, "tasks", None)
    if tasks is None:
        return f"<无 tasks 元数据, task_index={task_index}>"

    # Legacy: dict[int, str]
    if isinstance(tasks, dict):
        if task_index in tasks:
            return str(tasks[task_index])
        if str(task_index) in tasks:
            return str(tasks[str(task_index)])
        for k, v in tasks.items():
            try:
                if int(k) == int(task_index):
                    return str(v)
            except (TypeError, ValueError):
                continue

    if isinstance(tasks, (list, tuple)) and 0 <= int(task_index) < len(tasks):
        return str(tasks[int(task_index)])

    # LeRobot v3+: DataFrame，index = task 字符串，列 task_index
    if hasattr(tasks, "iterrows"):
        try:
            for task_str, row in tasks.iterrows():
                if not hasattr(row, "index") or "task_index" not in row.index:
                    continue
                ri = row["task_index"]
                if int(ri) == int(task_index):
                    return str(task_str)
        except Exception as e:
            logger.debug("tasks.iterrows 解析失败: %s", e)

    # 列筛选（不依赖 index 是否为 task 文本）
    cols = getattr(tasks, "columns", None)
    if cols is not None and "task_index" in cols:
        try:
            sub = tasks[tasks["task_index"] == int(task_index)]
            if len(sub) >= 1:
                for col in ("task", "text", "instruction", "language_instruction"):
                    if col in sub.columns:
                        val = sub[col].iloc[0]
                        if val is not None and str(val).strip():
                            return str(val)
                # 与官方一致：行索引即任务描述
                return str(sub.index[0])
        except Exception as e:
            logger.debug("tasks 列查询失败: %s", e)

    getter = getattr(meta, "get_task_by_index", None)
    if callable(getter):
        try:
            return str(getter(task_index))
        except Exception:
            pass

    return f"<无法解析 task, task_index={task_index}>"


def _task_text_from_sample_row(row: dict) -> str | None:
    """部分数据集在样本里直接带 task 字符串。"""
    for k in ("task", "language_instruction", "instruction", "prompt"):
        if k not in row or row[k] is None:
            continue
        v = row[k]
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8", errors="replace")
            except Exception:
                return str(v)
        s = str(v).strip()
        if s:
            return s
    return None


def _to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.dtype == np.uint8:
        return x
    if np.issubdtype(x.dtype, np.floating):
        if float(np.nanmax(x)) <= 1.0 + 1e-5:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return x


def _to_video_frame_uint8_rgb(arr) -> np.ndarray:
    """将单帧转为 uint8 HWC RGB，供 OpenCV/ImageIO 写入（要求通道数为 1–4）。

    LeRobot / PyTorch 常为 **CHW (C,H,W)**；若按 HWC 解释会导致「通道数=宽度」而报错。
    """
    x = _to_numpy(arr)
    if x is None:
        raise ValueError("空图像")
    x = np.asarray(x)

    # (B, C, H, W) / (B, H, W, C)
    while x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
        return _to_uint8_hwc(x)

    if x.ndim != 3:
        raise ValueError(f"期望 2D/3D 图像，得到 shape={x.shape}")

    h0, w0, c0 = x.shape[0], x.shape[1], x.shape[2]
    # HWC：最后一维为通道
    if c0 in (1, 2, 3, 4):
        out = _to_uint8_hwc(x)
    # CHW：首维为小通道数且最后一维像空间尺寸
    elif h0 in (1, 2, 3, 4) and c0 >= 8 and w0 >= 8:
        x = np.transpose(x, (1, 2, 0))
        out = _to_uint8_hwc(x)
    else:
        # 兜底：最后一维很大、首维像通道
        if c0 > 4 and h0 in (1, 2, 3, 4):
            x = np.transpose(x, (1, 2, 0))
            out = _to_uint8_hwc(x)
        else:
            raise ValueError(f"无法推断图像布局 shape={x.shape}")

    # 写入视频常用 3 通道 BGR/RGB
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    elif out.shape[2] == 2:
        out = np.concatenate([out, out[:, :, :1]], axis=2)
    elif out.shape[2] == 4:
        out = out[..., :3]
    return out


def _stack_main_wrist(main: np.ndarray, wrist: np.ndarray | None, *, stack: bool) -> np.ndarray:
    if wrist is None or not stack:
        return _to_video_frame_uint8_rgb(main)
    m = _to_video_frame_uint8_rgb(main)
    w = _to_video_frame_uint8_rgb(wrist)
    if m.shape[0] != w.shape[0]:
        try:
            import cv2

            nh = int(m.shape[0])
            nw = max(1, int(round(w.shape[1] * (nh / float(w.shape[0])))))
            w = cv2.resize(w, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            # 无 cv2 时简单中心裁剪/填充不实现，仅拼原尺寸（可能高度不一致）
            pass
    if w.shape[0] != m.shape[0]:
        logger.warning("手腕图与主图高度仍不一致，导出仅拼接主视角")
        return m
    return np.concatenate([m, w], axis=1)


def _normalize_mp4_output_path(path: pathlib.Path, episode_index: int) -> pathlib.Path:
    """保证输出为「具体 .mp4 文件路径」，避免把目录传给 OpenCV/ImageIO 触发图像序列/读文件夹逻辑。

    常见误用：``--export-mp4 ./out`` 且 ``out`` 为已存在目录 → 必须写成 ``out/episode_0.mp4``。
    """
    path = path.expanduser()
    if path.exists() and path.is_dir():
        out = path / f"inspect_episode_{episode_index}.mp4"
        logger.info("export 路径为目录，将写入文件: %s", out)
        return out
    if not path.suffix:
        if path.exists() and path.is_dir():
            return path / f"inspect_episode_{episode_index}.mp4"
        return path.with_suffix(".mp4")
    return path


def _write_mp4(frames: list[np.ndarray], path: pathlib.Path, fps: float, *, episode_index: int = 0) -> None:
    path = _normalize_mp4_output_path(path, episode_index)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    out_str = str(path)
    if not out_str or not out_str.strip():
        raise ValueError("MP4 输出路径为空")

    if not frames:
        raise ValueError("无帧可写入 MP4")

    rgb = [_to_video_frame_uint8_rgb(f) for f in frames]
    h, wdim = rgb[0].shape[:2]

    try:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_str, fourcc, float(fps), (wdim, h))
        if not writer.isOpened():
            raise RuntimeError("VideoWriter 未能打开")
        for fr in rgb:
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        logger.info("已写入 MP4 (OpenCV): %s", path)
        return
    except Exception as e:
        logger.warning("OpenCV 写 MP4 失败 (%s)，尝试 imageio", e)

    try:
        import imageio.v2 as imageio

        # get_writer 按扩展名写入视频，避免 mimsave 把路径误判为「文件夹」插件
        writer = imageio.get_writer(out_str, fps=float(fps), codec="libx264", quality=8)
        try:
            for fr in rgb:
                writer.append_data(fr)
        finally:
            writer.close()
        logger.info("已写入 MP4 (imageio/ffmpeg get_writer): %s", path)
    except Exception as e2:
        try:
            import imageio.v2 as imageio

            imageio.mimsave(out_str, rgb, fps=float(fps), codec="libx264", quality=8)
            logger.info("已写入 MP4 (imageio mimsave): %s", path)
        except Exception as e3:
            try:
                import imageio.v2 as imageio

                imageio.mimsave(out_str, rgb, fps=float(fps))
                logger.info("已写入 MP4 (imageio 默认编码): %s", path)
            except Exception as e4:
                logger.error(
                    "无法写入 MP4。请确认：1) --export-mp4 为具体 .mp4 文件路径或已存在目录；"
                    "2) 安装 imageio-ffmpeg：pip install imageio imageio-ffmpeg。\n"
                    "OpenCV/ImageIO 报错: %s | %s | %s",
                    e2,
                    e3,
                    e4,
                )
                raise


def _print_fixed_episode_info(
    meta,
    ds_ep,
    episode_index: int,
    action_key: str,
    visual_candidates: list[str],
    state_key: str | None,
) -> None:
    """在仅包含目标 episode 子集的数据集上打印任务与元信息。"""
    n = len(ds_ep)
    logger.info("========== episode_index=%d（固定检查）==========", episode_index)
    te = getattr(meta, "total_episodes", None)
    if te is not None:
        logger.info("  meta.total_episodes: %s", te)
    logger.info("  本子集帧数: %d", n)
    if n == 0:
        logger.warning("  该 episode 无帧，跳过详情")
        return

    row0 = ds_ep[0]
    task_idx = row0.get("task_index", None)
    ep_in_row = row0.get("episode_index", None)
    if task_idx is not None:
        try:
            ti = int(task_idx) if not hasattr(task_idx, "item") else int(task_idx.item())
        except Exception:
            ti = int(task_idx)
        logger.info("  task_index: %s", ti)
        task_text = _resolve_task_text(meta, ti)
        if task_text.startswith("<"):
            alt = _task_text_from_sample_row(row0)
            if alt is not None:
                task_text = alt
        logger.info("  task 文本: %s", task_text)
    else:
        logger.info("  task_index: <样本中无该字段>")
        alt = _task_text_from_sample_row(row0)
        if alt is not None:
            logger.info("  task 文本（来自样本字段）: %s", alt)

    if ep_in_row is not None:
        logger.info("  样本内 episode_index: %s", ep_in_row)

    hf = getattr(ds_ep, "hf_dataset", None)
    if hf is not None and "task_index" in hf.column_names and n > 0:
        tcol = hf["task_index"]
        # 子集 hf 可能与全局索引对齐；取该子集第一帧
        try:
            v0 = tcol[0]
            logger.info("  hf_dataset task_index[0]: %s", int(v0) if hasattr(v0, "__int__") else v0)
        except Exception:
            pass

    # 打印首帧可用键与主视觉摘要
    avail = [k for k in visual_candidates if k in row0]
    main_k = avail[0] if avail else None
    logger.info("  首帧 row keys（节选）: %s", sorted(row0.keys())[:20])
    if main_k:
        img0 = _to_numpy(row0.get(main_k))
        if isinstance(img0, np.ndarray):
            logger.info("  首帧 %s: %s", main_k, _summarize_array(main_k, img0))
    if action_key in row0:
        a0 = _to_numpy(row0.get(action_key))
        if isinstance(a0, np.ndarray):
            logger.info("  首帧 %s: %s", action_key, _summarize_array(action_key, a0))
    if state_key and state_key in row0:
        s0 = _to_numpy(row0.get(state_key))
        if isinstance(s0, np.ndarray):
            logger.info("  首帧 %s: %s", state_key, _summarize_array(state_key, s0))
    logger.info("========== episode %d 检查结束 ==========", episode_index)


def _export_episode_mp4(
    ds_ep,
    meta,
    out_path: pathlib.Path,
    fps: float,
    visual_candidates: list[str],
    *,
    stack_wrist: bool,
    episode_index: int,
) -> None:
    avail = [k for k in visual_candidates if k in ds_ep[0]]
    if not avail:
        raise ValueError("无法导出 MP4：首帧无候选视觉键")
    main_k = avail[0]
    wrist_k = None
    for k in avail:
        if "wrist" in k.lower():
            wrist_k = k
            break

    frames: list[np.ndarray] = []
    for i in range(len(ds_ep)):
        row = ds_ep[i]
        main = _to_numpy(row.get(main_k))
        wri = _to_numpy(row.get(wrist_k)) if wrist_k else None
        if not isinstance(main, np.ndarray) or main.size == 0:
            logger.warning("跳过第 %d 帧：%s 无效", i, main_k)
            continue
        stacked = _stack_main_wrist(main, wri, stack=stack_wrist and wrist_k is not None)
        frames.append(stacked)

    _write_mp4(frames, out_path, fps, episode_index=episode_index)


def main(args: Args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from lerobot.datasets import lerobot_dataset as lerobot_dataset
    except ImportError as e:
        logger.error("请先安装 lerobot: uv pip install lerobot （或见项目依赖）\n%s", e)
        return 1

    repo_id = args.repo_id
    logger.info("Loading metadata: %s (root=%s)", repo_id, args.root or "<默认缓存>")

    try:
        meta = _load_metadata(lerobot_dataset, repo_id, args.root)
    except Exception as e:
        logger.error("无法加载元数据（网络/路径/repo_id）: %s", e)
        return 1

    fps = float(meta.fps)
    logger.info("fps=%s robot_type=%s", fps, getattr(meta, "robot_type", "?"))

    action_key = _infer_action_key(meta)
    visual_candidates = _infer_visual_candidates(meta)
    state_key = _infer_state_key(meta)
    logger.info("使用动作列: %s", action_key)
    logger.info("使用状态列: %s", state_key if state_key is not None else "<未检测到，跳过状态校验>")
    logger.info("候选视觉列: %s", visual_candidates[:6] if len(visual_candidates) > 6 else visual_candidates)
    delta_ts = {action_key: [t / fps for t in range(args.action_horizon)]}

    def _open_ds(episodes: list[int] | None):
        kwargs: dict = {"delta_timestamps": delta_ts}
        if args.root:
            kwargs["root"] = pathlib.Path(args.root)
        if episodes is not None:
            kwargs["episodes"] = episodes
        return lerobot_dataset.LeRobotDataset(repo_id, **kwargs)

    # 固定检查并打印 inspect_episode 的任务等信息（仅加载该 episode，速度快）
    try:
        ds_inspect = _open_ds([args.inspect_episode])
    except Exception as e:
        logger.error("无法为 inspect_episode=%d 构造 LeRobotDataset: %s", args.inspect_episode, e)
        return 1

    _print_fixed_episode_info(
        meta,
        ds_inspect,
        args.inspect_episode,
        action_key,
        visual_candidates,
        state_key,
    )

    if args.export_mp4:
        out = pathlib.Path(args.export_mp4)
        try:
            _export_episode_mp4(
                ds_inspect,
                meta,
                out,
                fps,
                visual_candidates,
                stack_wrist=args.export_stack_wrist,
                episode_index=args.inspect_episode,
            )
        except Exception as e:
            logger.error("导出 MP4 失败: %s", e)
            return 1

    episodes_filter: list[int] | None = None
    if args.episodes_filter:
        episodes_filter = [int(x.strip()) for x in args.episodes_filter.split(",") if x.strip()]

    try:
        ds = _open_ds(episodes_filter)
    except Exception as e:
        logger.error("无法构造 LeRobotDataset（抽检用）: %s", e)
        return 1

    if episodes_filter is None:
        logger.warning("未设置 --episodes-filter：抽检将使用全量数据集（体积大时初始化可能很慢）")

    n = len(ds)
    logger.info("数据集总帧数 len(dataset)=%d", n)
    if n == 0:
        logger.error("数据集长度为 0")
        return 1

    # 任务表
    tasks = meta.tasks
    if isinstance(tasks, dict):
        logger.info("meta.tasks: %d entries (dict)", len(tasks))
    elif hasattr(tasks, "__len__"):
        logger.info("meta.tasks: table-like, len=%s", len(tasks))
    else:
        logger.info("meta.tasks: %s", type(tasks))

    # 抽检索引
    rng = np.random.default_rng(42)
    if args.num_probe <= 0:
        indices = [0]
    elif args.num_probe >= n:
        indices = list(range(n))
    else:
        indices = sorted(rng.choice(n, size=args.num_probe, replace=False).tolist())

    bad = 0
    for i in indices:
        try:
            row = ds[i]
        except Exception as e:
            logger.error("ds[%d] 失败: %s", i, e)
            bad += 1
            continue

        required_keys = (action_key,) if state_key is None else (state_key, action_key)
        missing_key = False
        for key in required_keys:
            if key not in row:
                logger.error("样本缺少键 %r (index=%d)", key, i)
                bad += 1
                missing_key = True
                break
        if missing_key:
            continue

        available_visual_keys = [k for k in visual_candidates if k in row]
        if not available_visual_keys:
            logger.error("样本缺少可用视觉键 (index=%d), row keys=%s", i, list(row.keys()))
            bad += 1
            continue

        main_image_key = available_visual_keys[0]
        wrist_key = None
        for k in available_visual_keys:
            if "wrist" in k.lower():
                wrist_key = k
                break

        img = _to_numpy(row.get(main_image_key))
        wri = _to_numpy(row.get(wrist_key)) if wrist_key is not None else None
        if isinstance(img, np.ndarray) and img.size > 0:
            if float(np.max(img)) == float(np.min(img)) and img.ndim >= 1:
                logger.warning("index=%d %s 可能为常数（全相同像素）", i, main_image_key)
        else:
            logger.error("index=%d %s 无效或为空", i, main_image_key)
            bad += 1

        if wrist_key is not None and wri is not None and isinstance(wri, np.ndarray) and wri.size == 0:
            logger.error("index=%d %s 为空", i, wrist_key)
            bad += 1

        if i == indices[0]:
            logger.info("示例样本 [index=%d]:", i)
            for k in ("task_index", "episode_index", "index"):
                if k in row:
                    logger.info("  %s = %s", k, row[k])
            if isinstance(img, np.ndarray):
                logger.info("  %s", _summarize_array(main_image_key, img))
            if wrist_key is not None and isinstance(wri, np.ndarray):
                logger.info("  %s", _summarize_array(wrist_key, wri))
            act = _to_numpy(row.get(action_key))
            if isinstance(act, np.ndarray):
                logger.info("  %s", _summarize_array(action_key, act))

    if bad > 0:
        logger.error("抽检发现问题: %d 条", bad)

    # 全表 task_index 分布（可选）
    hf = getattr(ds, "hf_dataset", None)
    if args.scan_all_task_stats and hf is None:
        logger.warning("当前 LeRobotDataset 无 hf_dataset 属性，跳过全表统计。")
    elif args.scan_all_task_stats and hf is not None:
        if "task_index" in hf.column_names:
            tidx = np.array(hf["task_index"])
            logger.info("task_index: min=%d max=%d unique=%d", int(tidx.min()), int(tidx.max()), len(np.unique(tidx)))
            ctr = Counter(int(x) for x in tidx.tolist())
            most = ctr.most_common(5)
            logger.info("出现最多的 task_index (top5): %s", most)
        if "episode_index" in hf.column_names:
            ep = np.array(hf["episode_index"])
            logger.info("episode_index: min=%d max=%d unique_episodes=%d", int(ep.min()), int(ep.max()), len(np.unique(ep)))

    logger.info("检查完成。若磁盘上 images/ 子目录为空，通常仍属正常（图像在视频/Parquet 中）。")
    return 1 if bad > 0 else 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
