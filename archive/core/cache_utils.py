"""Shared cache utilities.

Eliminates repeated logic for cache batch-size resolution, conditioning
seed derivation, and cache shuffle/rebatch across cache_trajectory.py,
cache_random.py, train.py, and train_cyclic.py.
"""

import gc
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch


# One persistent threadpool for pin_memory calls.
# pin_memory() is a kernel syscall (mlock) — it releases the GIL, so threads
# give real parallelism here. 4 workers covers the typical 4-tensor-per-entry
# pattern without over-subscribing the memory bus.
_PIN_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pin")


def _pin(t: torch.Tensor | None) -> torch.Tensor | None:
    """Pin a single tensor in a thread-pool worker. None-safe."""
    if t is None:
        return None
    return t.pin_memory()


def pin_tensors_parallel(*tensors):
    """Pin multiple tensors concurrently. Returns a list in the same order."""
    futures = [_PIN_POOL.submit(_pin, t) for t in tensors]
    return [f.result() for f in futures]


def resolve_gen_batch_size(cache_batch_size, fallback_batch_size: int) -> int:
    """Resolve the effective batch size for cache generation.

    If cache_batch_size is None, 0, or unset, fall back to the training batch_size.
    """
    if cache_batch_size is None or cache_batch_size == 0:
        return fallback_batch_size
    return cache_batch_size


def warn_batch_mismatch(gen_batch_size: int, train_batch_size: int):
    """Print a warning if cache generation is inefficiently sized."""
    if gen_batch_size < train_batch_size:
        ratio = train_batch_size / gen_batch_size
        print(f"    Warning: cache_batch_size={gen_batch_size} < batch_size={train_batch_size}. "
              f"Cache generation will do ~{ratio:.1f}x more forward passes than needed. "
              f"Use cache_batch_size >= batch_size for efficiency.")
    elif gen_batch_size % train_batch_size != 0:
        waste = gen_batch_size % train_batch_size
        print(f"    Warning: cache_batch_size={gen_batch_size} is not a multiple of "
              f"batch_size={train_batch_size}. {waste} sample(s) per cache entry "
              f"will be discarded during shuffling. Use a multiple for efficiency.")


def shuffle_and_rebatch_cache(cache_raw, batch_size: int, shuffle_seed: int, effective_batch_size: int = None):
    """Flatten -> shuffle -> re-batch cache into training batches.

    Cache entry formats:
    v3: (x_t, target, ctx, y, ctx_u, y_u, alpha, sigma, t_val) - 9 elements
    v2: (x_t, target, ctx, y, alpha, sigma, t_val) - 7 elements
    v1: (x_t, target, alpha, sigma, t_val, cond_seed) - 6 elements
    
    Args:
        batch_size: Original batch size
        shuffle_seed: Random seed for shuffling
        effective_batch_size: If provided, use this instead of batch_size.
                             If None, defaults to batch_size.
                             Use batch_size // 2 when CFG is enabled to process
                             cond-uncond pairs instead of 2x batch.
    
    Returns a list of tuples, each containing `effective_batch_size` samples, ready for training.
    """
    if effective_batch_size is None:
        effective_batch_size = batch_size
    flat = []
    for entry in cache_raw:
        L = len(entry)
        x_t_b, target_b = entry[0], entry[1]
        
        # Determine number of samples in this entry
        N = x_t_b.shape[0]
        
        for k in range(N):
            sample = [
                x_t_b[k:k+1].contiguous(),
                target_b[k:k+1].contiguous(),
            ]
            
            if L >= 9:
                # v3/v4: (x_t, eps_cond, ctx, y, ctx_u, y_u, alpha, sigma, t_val[, target_uncond])
                ctx_b, y_b, ctx_u_b, y_u_b = entry[2], entry[3], entry[4], entry[5]
                alpha, sigma, t_val = entry[6], entry[7], entry[8]

                def _get_k(t, idx):
                    if t is None: return None
                    if not isinstance(t, torch.Tensor): return t
                    if t.shape[0] > 1: return t[idx:idx+1].contiguous()
                    return t.contiguous()

                sample.extend([
                    _get_k(ctx_b, k),
                    _get_k(y_b, k),
                    _get_k(ctx_u_b, k),
                    _get_k(y_u_b, k),
                ])

                def _get_val(v, idx):
                    if isinstance(v, (list, tuple, torch.Tensor)) and len(v) > 1:
                        return v[idx]
                    if isinstance(v, (list, tuple, torch.Tensor)):
                        return v[0]
                    return v

                sample.extend([
                    _get_val(alpha, k),
                    _get_val(sigma, k),
                    _get_val(t_val, k),
                ])

                # target_uncond at index 9
                if L >= 10:
                    sample.append(_get_k(entry[9], k))
            elif L >= 7:
                # v2: (x_t, target, ctx, y, alpha, sigma, t_val)
                ctx_b, y_b = entry[2], entry[3]
                alpha, sigma, t_val = entry[4], entry[5], entry[6]
                
                def _get_k(t, idx):
                    if t is None: return None
                    if not isinstance(t, torch.Tensor): return t
                    if t.shape[0] > 1: return t[idx:idx+1].contiguous()
                    return t.contiguous()

                sample.extend([
                    _get_k(ctx_b, k),
                    _get_k(y_b, k),
                    None, # ctx_u
                    None, # y_u
                ])
                
                def _get_val(v, idx):
                    if isinstance(v, (list, tuple, torch.Tensor)) and len(v) > 1:
                        return v[idx]
                    if isinstance(v, (list, tuple, torch.Tensor)):
                        return v[0]
                    return v
                
                sample.extend([
                    _get_val(alpha, k),
                    _get_val(sigma, k),
                    _get_val(t_val, k),
                ])
            else:
                # v1: (x_t, target, alpha, sigma, t_val, ...)
                alpha, sigma, t_val = entry[2], entry[3], entry[4]
                sample.extend([None, None, None, None]) # ctx, y, ctx_u, y_u
                
                def _get_val(v, idx):
                    if isinstance(v, (list, tuple, torch.Tensor)) and len(v) > 1:
                        return v[idx]
                    if isinstance(v, (list, tuple, torch.Tensor)):
                        return v[0]
                    return v
                
                sample.extend([
                    _get_val(alpha, k),
                    _get_val(sigma, k),
                    _get_val(t_val, k),
                ])
                # Special: store cond_seed if present for v1 reconstruction
                if L > 5:
                    sample.append(_get_val(entry[5], k))
                if L > 6:
                    sample.append(_get_val(entry[6], k))

            flat.append(tuple(sample))
            
    del cache_raw
    gc.collect()

    rng = random.Random(shuffle_seed ^ 0xDEADBEEF)
    rng.shuffle(flat)  # shuffle while it's still a list -- list indexing is O(1),
                        # deque indexing is O(n), so shuffling a deque would be
                        # just as slow as the pop(0) pattern we're fixing below.

    # Popping from the front of a list (flat.pop(0)) is O(n) per call, making
    # the whole rebatch loop O(n^2) -- with n_samples up to 200,000 this could
    # get slow. A deque gives O(1) popleft() instead.
    flat = deque(flat)

    # Pre-build all batched chunks, then pin all tensors in parallel.
    # pin_memory() is a kernel mlock syscall that releases the GIL, so 4
    # concurrent workers give real speedup with no correctness risk.
    raw_batches = []
    while flat:
        chunk = []
        while flat and len(chunk) < effective_batch_size:
            chunk.append(flat.popleft())
        
        if len(chunk) < effective_batch_size:
            del chunk
            continue

        # Concatenate on CPU first (fast, stays on CPU)
        x_t_cat    = torch.cat([c[0] for c in chunk])
        tgt_c_cat  = torch.cat([c[1] for c in chunk])
        ctx_cat    = torch.cat([c[2] for c in chunk]) if chunk[0][2] is not None else None
        y_cat      = torch.cat([c[3] for c in chunk]) if chunk[0][3] is not None else None
        ctx_u_cat  = torch.cat([c[4] for c in chunk]) if chunk[0][4] is not None else None
        y_u_cat    = torch.cat([c[5] for c in chunk]) if chunk[0][5] is not None else None
        alphas     = [c[6] for c in chunk]
        sigmas     = [c[7] for c in chunk]
        t_vals     = [c[8] for c in chunk]
        tgt_u_cat  = (torch.cat([c[9] for c in chunk]) if len(chunk[0]) > 9 and chunk[0][9] is not None
                      else None)

        raw_batches.append((x_t_cat, tgt_c_cat, ctx_cat, y_cat, ctx_u_cat,
                            y_u_cat, alphas, sigmas, t_vals, tgt_u_cat))
        del chunk

    del flat
    gc.collect()

    # Pin all tensors across all batches in parallel.
    # Collect all (batch_idx, field_idx, tensor) tuples, submit, gather.
    batched = []
    for (x_t_cat, tgt_c_cat, ctx_cat, y_cat, ctx_u_cat,
         y_u_cat, alphas, sigmas, t_vals, tgt_u_cat) in raw_batches:

        pinned = pin_tensors_parallel(
            x_t_cat, tgt_c_cat, ctx_cat, y_cat, ctx_u_cat, y_u_cat, tgt_u_cat
        )
        x_t_p, tgt_c_p, ctx_p, y_p, ctx_u_p, y_u_p, tgt_u_p = pinned

        entry = (x_t_p, tgt_c_p, ctx_p, y_p, ctx_u_p, y_u_p, alphas, sigmas, t_vals)
        if tgt_u_p is not None:
            entry = entry + (tgt_u_p,)
        batched.append(entry)

    del raw_batches
    gc.collect()
    return batched


