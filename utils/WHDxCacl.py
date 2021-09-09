from collections import namedtuple
import torch
import torch.autograd as tag
import cupy as cp
from cupy.cuda import function
from pynvrtc.compiler import Program

WHDx_forward_T_totThreads = 128
code = '''
// sum across dim col using a, x, sh_W[0..31]
__forceinline__
__device__ float ReduxDimCol(const float* sh_W, unsigned int a, unsigned int x)
{
    float sum = 0;
    unsigned int aXORx = a ^ x;
    for (int bix = 0; bix < 32; bix++, aXORx = aXORx >> 1)
    {
        sum += (aXORx & 0x1U) == 0 ? 0 : sh_W[bix];
    }
    return sum;
}
__global__ void WHDx_forward_cuda_T(float* WHDx_T, const unsigned int* uA_T, const unsigned int* uX_B_T, const float* W, int N, int M, int D)
{
    const int packDimBlocks = 4;
    const int totThreads = WHDx_forward_T_totThreads;
    if (blockDim.x != totThreads)
        asm("trap;");
    if (D % (4*32) != 0)
        asm("trap;");
    __shared__ float sh_W[totThreads * packDimBlocks];
    int totDimCols = D / (packDimBlocks*32);
    int m = blockDim.x * blockIdx.x + threadIdx.x;
    int uA_ix = m;
    int uX_ix = 0;
    for (int dimCol = 0; dimCol < totDimCols; dimCol++)
    {
        __syncthreads(); 
        if (threadIdx.x < 32 * packDimBlocks)
            sh_W[threadIdx.x] = abs(W[dimCol*32*packDimBlocks+threadIdx.x]);
        __syncthreads(); 
        if (m < M)
        {
            unsigned int a1 = uA_T[uA_ix]; uA_ix+=M;
            unsigned int a2 = uA_T[uA_ix]; uA_ix+=M;
            unsigned int a3 = uA_T[uA_ix]; uA_ix+=M;
            unsigned int a4 = uA_T[uA_ix]; uA_ix+=M;
            for (int n = 0, outIx = m; n < N; n++, outIx+=M)
            {
                unsigned int x1 = uX_B_T[uX_ix++];
                unsigned int x2 = uX_B_T[uX_ix++];
                unsigned int x3 = uX_B_T[uX_ix++];
                unsigned int x4 = uX_B_T[uX_ix++];
                float sum = 0;
                sum += ReduxDimCol(sh_W + 0 * 32, a1, x1);
                sum += ReduxDimCol(sh_W + 1 * 32, a2, x2);
                sum += ReduxDimCol(sh_W + 2 * 32, a3, x3);
                sum += ReduxDimCol(sh_W + 3 * 32, a4, x4);
                WHDx_T[outIx] += sum;
            }
        }
    }
}
'''
WHDx_backward_cuda_T_totThreads = 64
code += '''
__forceinline__
__device__ float Sgn(const float x)
{
    return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0);
}

__global__ void WHDx_backward_cuda_T(float* grad_W, const unsigned int* uA_T, const unsigned int* uX_T, const float* W, const float* grad_output, int N, int M, int D)
{
    if (D % 32 != 0) // 61
        asm("trap;");
    int laneId = threadIdx.x & 31;
    int warpId = threadIdx.x / 32;
    __shared__ float sh_grad_output[WHDx_backward_cuda_T_totThreads];
    __shared__ unsigned int sh_Xt_dim[WHDx_backward_cuda_T_totThreads]; // each warp has its len 32 sub_array to cache 32 N's
    float* sh_grad_output_warp = sh_grad_output + warpId * 32;
    int d = blockDim.x * blockIdx.x + threadIdx.x;
    int dBlock = d / 32;
    unsigned int laneMask = 1U << laneId;
    float sum = 0;
    float sign_w = Sgn(W[d]);
    unsigned int* sh_byWarp_Xt_dim = sh_Xt_dim + 32 * warpId;
    const unsigned int* At_dim = uA_T + dBlock * M;
    const unsigned int* Xt_dim = uX_T + dBlock * N;
    for (int blockN = 0; blockN < N; blockN += 32)
    {
        sh_byWarp_Xt_dim[laneId] = Xt_dim[laneId + blockN];
        for (int rowM = 0, go_ix = blockN + laneId; rowM < M; rowM++, go_ix += N)
        {
            unsigned int a = At_dim[rowM];

            sh_grad_output[threadIdx.x] = grad_output[go_ix];
            for (int elemInBlockN = 0; elemInBlockN < 32; elemInBlockN++)
            {
                float go = sh_grad_output_warp[elemInBlockN];
                unsigned int x = sh_byWarp_Xt_dim[elemInBlockN];
                unsigned int aMx = a ^ x;
                sum += (aMx & laneMask) == 0 ? 0 : go * sign_w;
            }
        }
    }
    grad_W[d] = sum;
}

''' 
code=code.replace("WHDx_forward_T_totThreads", str(WHDx_forward_T_totThreads))
code=code.replace("WHDx_backward_cuda_T_totThreads", str(WHDx_backward_cuda_T_totThreads))
cuda_func_names =  ["WHDx_forward_cuda_T", "WHDx_backward_cuda_T"]
mod_cuda = cp.RawModule(code=code, options=('-std=c++11',), name_expressions=cuda_func_names)
WHDx_forward_cuda_T = mod_cuda.get_function(cuda_func_names[0]) 
WHDx_backward_cuda_T = mod_cuda.get_function(cuda_func_names[1]) 



class WHDxFunction(tag.Function):
    @staticmethod
    # A:(M,D) X:(N,D) W:(D,)
    def forward(ctx, A_bits_T, X_bits_T, X_bits_T_byDimBlock4, W): 
        with torch.no_grad():
            ctx.save_for_backward(A_bits_T, X_bits_T, W)
            if (not X_bits_T.is_contiguous()):
                raise Exception("X_bits_T must be contiguous")
            if (not X_bits_T_byDimBlock4.is_contiguous()):
                raise Exception("X_bits_T_byDimBlock4 must be contiguous")
            if (not A_bits_T.is_contiguous()):
                raise Exception("A_bits_T must be contiguous")
            D = A_bits_T.shape[0]*32
            if (D % (32*4) != 0):
                raise Exception("D must be a multiple of 32*4")
            M = A_bits_T.shape[1]
            N = (X_bits_T.shape[0] * X_bits_T.shape[1]) // (D // 32)
            totThreads = WHDx_forward_T_totThreads
            totBlocks = M // totThreads 
            if (M % totThreads != 0):
                totBlocks += 1

            ## slow, mem-inneficient version wich uses original X and A (not the bit-packed version):
            # WHDx_T = ((A - X) * W).abs()

            WHDx_T = torch.zeros((N,M), dtype=W.dtype, device=W.device, requires_grad=False) 
            Stream = namedtuple('Stream', ['ptr'])
            s = Stream(ptr=torch.cuda.current_stream().cuda_stream)
            WHDx_forward_cuda_T(grid=(totBlocks,1,1), block=(totThreads,1,1), 
                                args=[WHDx_T.data_ptr(), A_bits_T.data_ptr(), X_bits_T_byDimBlock4.data_ptr(), W.data_ptr(), N, M, D], 
                                stream=s)
            torch.cuda.synchronize()

            WHDx = WHDx_T.T
        return WHDx

    @staticmethod
    def backward(ctx, grad_output): 
        with torch.no_grad():
            grad_A = grad_X = grad_X_byDimBlock4 = grad_W = None
            if ctx.needs_input_grad[0]:
                    raise Exception("Can't differenciate wrt A")
            if ctx.needs_input_grad[1]:
                    raise Exception("Can't differenciate wrt X")
            if ctx.needs_input_grad[2]:
                    raise Exception("Can't differenciate wrt X_byDimBlock4")
            if ctx.needs_input_grad[3]:
                A_bits_T, X_bits_T, W = ctx.saved_tensors
                if (not X_bits_T.is_contiguous()):
                    raise Exception("X_bits_T must be contiguous")
                if (not A_bits_T.is_contiguous()):
                    raise Exception("A_bits_T must be contiguous")
                if (not grad_output.is_contiguous()):
                    grad_output = grad_output.contiguous()
                D = A_bits_T.shape[0] * 32
                M = A_bits_T.shape[1]
                N = (X_bits_T.shape[0] * X_bits_T.shape[1]) // (D // 32)
                if (N % 32 != 0):
                    raise Exception("N % 32 != 0")
                #totThreads = 1024 if D >= 45056 else 128
                totThreads = 64
                if (totThreads != WHDx_backward_cuda_T_totThreads):
                    raise Exception("totThreads != WHDx_backward_cuda_T_totThreads")
                if (totThreads % 32 != 0):
                    raise Exception("totThreads % 32 != 0")
                totBlocks = D // totThreads
                if (D % totThreads != 0):
                    raise Exception("D must be a multiple of totThreads")

                ## slow, mem-inneficient version. uses original X and A, not the bit-packed version
                #A_r = A.repeat_interleave(repeats=N, dim=0) # A_r:(M*N,D)
                #X_ri = X.repeat(M, 1) # X_ri:(N*M,D)
                #grad_W = (grad_output.view(M * N) * ((A_r - X_ri) * ((A_r - X_ri) * W).sign()) .T).sum(-1)

                grad_W = torch.empty((D), dtype=W.dtype, device=W.device, requires_grad=False) 
                Stream = namedtuple('Stream', ['ptr'])
                s = Stream(ptr=torch.cuda.current_stream().cuda_stream)
                WHDx_backward_cuda_T(grid=(totBlocks,1,1), block=(totThreads,1,1), 
                                    args=[grad_W.data_ptr(), A_bits_T.data_ptr(), X_bits_T.data_ptr(), W.data_ptr(), grad_output.data_ptr(), N, M, D], 
                                    stream=s)
                torch.cuda.synchronize()

            return grad_A, grad_X, grad_X_byDimBlock4, grad_W

    @staticmethod
    def GetDimBlock4(t_X_T_bits):
        # TODO: there are better ways to do this...
        t_tmp = torch.reshape(torch.empty_like(t_X_T_bits), (t_X_T_bits.shape[0] // 4, -1))
        for dimIx in range(0, t_X_T_bits.shape[0], 4):
            t_tmp[dimIx // 4] = t_X_T_bits[dimIx:dimIx + 4,:].T.flatten()
        return torch.reshape(t_tmp, (-1,4))


