# SIMP Solver Optimization Plan

## Objective
Achieve 3-4x speedup on SIMPSolverGPU with low-effort, high-payoff optimizations.

## Phase 1: Quick Wins (3-4x speedup, 1-2 hours work)

### 1. Float32 Support (2x speedup)
**Current**: All operations use `torch.float64`
**Change**: Default to `torch.float32`, allow override
**Files**: `simp_solver_gpu.py`
**Lines**: 34, 75, 225, 348
**Changes**:
- Add `dtype` parameter to `__init__` (default `torch.float32`)
- Update `_get_Ke()` cast from `.double()` to `.to(dtype)`
- Update `_build_filter()` to use dtype
- Update `run()` to use `dtype` instead of hardcoded `float64`
- Update `_pcg_matrix_free()` Jacobi preconditioner dtype

### 2. Adaptive PCG Tolerance (30-50% fewer iterations)
**Current**: `tol=1e-7, max_iter=2000` fixed
**Change**: Adaptive based on SIMP iteration
**Files**: `simp_solver_gpu.py`
**Lines**: 172, 196, 259
**Changes**:
- Modify `_pcg()` to accept `x0` (warm-start) parameter
- Modify `_solve()` to accept and pass `tol` and `max_iter`
- In `run()`, compute adaptive tolerance:
  - Iter 0-20: `tol=1e-4, max_iter=200`
  - Iter 20-40: `tol=1e-5, max_iter=300`
  - Iter 40+: `tol=1e-6, max_iter=300`

### 3. Warm-Start PCG (50% fewer iterations after iter 2)
**Current**: `x = torch.zeros_like(b)` every iteration
**Change**: Pass previous solution as initial guess
**Files**: `simp_solver_gpu.py`
**Lines**: 81-94, 259, 267-283
**Changes**:
- Store `self._u_prev = None` in `__init__`
- In `_sensitivity()`, pass `self._u_prev` to `_solve()`
- Return solution `u` from `_sensitivity()` (already done)
- Store `self._u_prev = u` for next iteration

### 4. Cached COO Sparsity Pattern (Eliminates recomputation)
**Current**: `iK, jK` recomputed every SIMP iteration in `_assemble_K()`
**Change**: Cache in `__init__`, only recompute `sK`
**Files**: `simp_solver_gpu.py`
**Lines**: 161-169
**Changes**:
- In `__init__`, precompute `self._iK`, `self._jK` from `edof_mat`
- In `_assemble_K()`, use cached indices, only compute `sK`

## Phase 2: Medium Effort (Additional 2x speedup)

### 5. Better Preconditioner (SSOR or ILU)
**Current**: Jacobi (diagonal) preconditioner
**Change**: Implement SSOR or use scipy ILU on CPU
**Estimated effort**: Half day

### 6. Direct Solver for Small Grids
**Current**: PCG for all grid sizes
**Change**: Use `torch.linalg.solve` (dense Cholesky) for <5k DOF
**Threshold**: 16×8×8 = ~4k DOF

## Implementation Order
1. Float32 (safest, easiest, immediate 2x)
2. Tolerance tuning (30-50% reduction)
3. Warm-start (50% reduction after iter 2)
4. Cached sparsity (minor but free)

## Expected Results
- Float32 alone: 2x speedup
- + Adaptive tolerance: 2.5-3x
- + Warm-start: 3-4x
- + Cached sparsity: 3.5-4.5x

## Testing Plan
- Benchmark before/after on 32×8×8 and 64×16×16 grids
- Verify compliance values remain similar (<1% change)
- Check for NaN/instability with float32
