import torch
import torch.nn.functional as F
import numpy as np
from genpipeline.vae_design_model import DesignVAE
from genpipeline.topology.simp_solver_gpu import SIMPSolverGPU
from genpipeline.cuda_kernels import simp_sensitivity

def test_comparison():
    vae = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda().eval()
    z = torch.randn(1, 32, device='cuda', requires_grad=True)
    
    nx, ny, nz = 32, 8, 8
    solver = SIMPSolverGPU(nx=nx, ny=ny, nz=nz)
    
    print("Forward pass...")
    logits = vae.decode_logits(z)
    voxels_32 = F.interpolate(logits, size=(nx, ny, nz), mode='trilinear', align_corners=False).squeeze()
    xPhys = torch.sigmoid(voxels_32)
    
    print("Physics solve...")
    f, fixed = solver._get_bcs(1000.0)
    K = solver._assemble_K(xPhys)
    u = solver._solve(K, f, fixed)
    
    print("Pure PyTorch sensitivity...")
    u_e = u[solver._edof_mat]
    ce = torch.sum((u_e @ solver.Ke) * u_e, dim=1)
    dc_pt = -solver.penal * (torch.clamp(xPhys.flatten(), min=1e-3)**(solver.penal-1)) * ce
    print(f"dc_pt norm: {dc_pt.norm().item()}")
    
    print("Custom sensitivity kernel...")
    try:
        dc_custom = simp_sensitivity(
            xPhys.flatten(), u, solver.Ke, solver._edof_mat, 
            solver.penal, nx, ny, nz
        )
        print(f"dc_custom norm: {dc_custom.norm().item()}")
        
        diff = torch.norm(dc_pt - dc_custom.cpu())
        print(f"Difference: {diff.item()}")
    except Exception as e:
        print(f"Custom kernel failed: {e}")

if __name__ == "__main__":
    test_comparison()
