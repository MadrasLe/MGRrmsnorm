import torch
import rmsnorm_cuda_ops
import math
from torch.autograd import gradcheck

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, epsilon=1e-5):
        output, inv_rms = rmsnorm_cuda_ops.rmsnorm_forward(input, weight, epsilon)
        ctx.save_for_backward(input, weight, inv_rms)
        ctx.epsilon = epsilon
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, inv_rms = ctx.saved_tensors
        grad_input, grad_weight = rmsnorm_cuda_ops.rmsnorm_backward(
            grad_output.contiguous(), input, weight, inv_rms
        )
        return grad_input, grad_weight, None

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, epsilon=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, input):
        return RMSNormFunction.apply(input, self.weight, self.epsilon)

def test_rmsnorm():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("--- Running Manual Check ---")
    # Dimensions
    batch = 32
    seq_len = 128
    hidden = 4096 
    
    # Setup inputs with grad
    input = torch.randn(batch * seq_len, hidden, device=device, requires_grad=True)
    weight = torch.randn(hidden, device=device, requires_grad=True)
    
    # ---------------------------------------------------------
    # PyTorch Reference (Manual Implementation for correctness)
    # ---------------------------------------------------------
    input_ref = input.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    rms_ref = torch.rsqrt(input_ref.pow(2).mean(-1, keepdim=True) + 1e-5)
    output_ref = input_ref * rms_ref * weight_ref
    
    # Backward Ref
    loss_ref = output_ref.sum()
    loss_ref.backward()

    # ---------------------------------------------------------
    # Custom Kernel
    # ---------------------------------------------------------
    model = RMSNorm(hidden).to(device)
    # Force weights to match
    model.weight = torch.nn.Parameter(weight.detach().clone()) 
    
    # Forward
    output_custom = model(input)
    
    # Backward
    loss_custom = output_custom.sum()
    loss_custom.backward()

    # ---------------------------------------------------------
    # Verify Forward
    # ---------------------------------------------------------
    diff = (output_custom - output_ref.detach()).abs().max()
    print(f"Forward Max diff: {diff.item()}")
    assert diff < 1e-4, "Forward verification failed!"

    # ---------------------------------------------------------
    # Verify Backward
    # ---------------------------------------------------------
    # Grad Input
    diff_grad_input = (input.grad - input_ref.grad).abs().max()
    print(f"Grad Input Max diff: {diff_grad_input.item()}")
    
    # Grad Weight
    diff_grad_weight = (model.weight.grad - weight_ref.grad).abs().max()
    print(f"Grad Weight Max diff: {diff_grad_weight.item()}")

    # Tolerances for float32 backward can be a bit higher due to accumulation order
    if diff_grad_input < 1e-3 and diff_grad_weight < 1e-3:
        print("Manual Backward Verification PASSED")
    else:
        print("Manual Backward Verification FAILED")

    # ---------------------------------------------------------
    # Autograd Gradcheck (Rigorous)
    # ---------------------------------------------------------
    print("\n--- Running torch.autograd.gradcheck (float32) ---")
    
    # Use smaller size for gradcheck to be fast and avoid huge Jacobian
    # Note: hidden size must be divisible by 4 for our kernel
    batch_gc = 2
    seq_gc = 4
    hidden_gc = 128 
    
    test_input_f32 = torch.randn(batch_gc * seq_gc, hidden_gc, device=device, dtype=torch.float32, requires_grad=True)
    test_weight_f32 = torch.randn(hidden_gc, device=device, dtype=torch.float32, requires_grad=True)

    # eps: Perturbation size for finite differences. Must be > machine epsilon of float32.
    # atol: Absolute tolerance. float32 accumulation noise can be significant.
    # rtol: Relative tolerance.
    try:
        ok = gradcheck(
            RMSNormFunction.apply, 
            (test_input_f32, test_weight_f32, 1e-5), 
            eps=1e-3, 
            atol=1e-2,
            rtol=1e-3,
            nondet_tol=0.0
        )
        print(f"Gradcheck result: {ok}")
    except Exception as e:
        print(f"Gradcheck failed with error: {e}")
        # Re-raise if we want to fail the CI, but for now let's just print
        raise e

if __name__ == "__main__":
    test_rmsnorm()
