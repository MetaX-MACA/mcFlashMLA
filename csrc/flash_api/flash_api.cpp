// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)and
// deepseek-ai/FlashMLA(https://github.com/deepseek-ai/FlashMLA)

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <mctlass/fast_math.h>

#include "flash_mla.h"
#include "static_switch.h"
#include "run_mla.h"
#include "host_utils.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

inline int int64_stride_to_int(int64_t orig_stride) {
    if (orig_stride > std::numeric_limits<int>::max()) {
        TORCH_CHECK(false, "[Sparse TopK Attention] Stride exceeds int32 limit: ", orig_stride);
    }
    return static_cast<int>(orig_stride);
}

// Note: should match the kernel dispatch tile size
inline std::pair<int, int> get_tile_size(int arch, int seqlen_q, bool is_sparse_attn) {
    int block_m = 0;
    int block_n = 0;
    if (is_sparse_attn) {
        // xcore1500 use the same kernel with xcore1000 in sparse decode now
        block_m = 64, block_n = 16;
    } else {
        if (arch >= 1500) {
            // only support blockM=64 in xcore1500 dense decode now
            block_m = 64, block_n = 32;
        } else {
            if (seqlen_q >= 64) {
                block_m = 64, block_n = 16;
            } else if (seqlen_q >= 32) {
                block_m = 32, block_n = 16;
            } else {
                block_m = 16, block_n = 16;
            }
        }
    }
    return {block_m, block_n};
}

struct DecodingAttnImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int k_block_size;
};

DecodingAttnImplMeta get_attn_impl_meta(
    int arch,
    int sm_count,
    int num_q_tokens_per_head_k,
    int h_k,
    int block_m,
    int block_n,
    std::optional<int> h_q_,
    bool is_fp8_kvcache,
    bool is_sparse_attn
) {
    if (is_sparse_attn) {
        if (is_fp8_kvcache) {
            TORCH_CHECK(false, "Sparse fp8 MLA is not supported.");
        } else {
            // Sparse BF16 MLA
            TORCH_CHECK(h_q_.has_value());
            int h_q = h_q_.value();
            TORCH_CHECK(h_q % h_k == 0, "h_k must be divisible by h_q.");
            int s_q = num_q_tokens_per_head_k * h_k / h_q;
            // BF16/FP16 + Sparse MLA
            return {
                std::max((sm_count/2) / h_k / (mctlass::ceil_div(h_q/h_k, 2*64) * s_q), 1),
                5,
                block_n // block_n
            };
        }
    } else {
        TORCH_CHECK(!is_fp8_kvcache, "FP8 KV Cache is not supported.");
        // Dense BF16/FP8 MLA
        return {
            std::max(sm_count / h_k / mctlass::ceil_div(num_q_tokens_per_head_k, block_m), 1),
            5,
            block_n,
        };
    }
}

std::vector<at::Tensor>
fwd_kvcache_mla(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    c10::optional<const at::Tensor> &vcache_,    // num_blocks x page_block_size x num_heads_k x head_size_v
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    bool is_fp8_kvcache,                         // fp8 kvcache=False
    c10::optional<const at::Tensor> &indices,    // None, or batch_size x seqlen_q x topk
    c10::optional<const at::Tensor> &indices_all_valid_per_q,    // batch_size x seqlen_q x 1, per-query flag indicating whether all top-k indices for each query token are valid.
    int const cp_world_size,  // context parallelism (cp) world size
    int const cp_rank,         // cp rank
    c10::optional<const at::Tensor> &cp_tot_seqused_k_ // b. total seqused_k in cp world
) {
    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;

    at::Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kBFloat16 || q_dtype == torch::kFloat16);
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(!is_fp8_kvcache, "flash mla with kvcache api not support fp8 now");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    bool is_sparse_attn = indices.has_value();
    int topk = is_sparse_attn ? indices->size(-1) : -1;
    TORCH_CHECK(!is_sparse_attn || indices->dtype() == torch::kInt32, "indices must have dtype int32");
    TORCH_CHECK(!is_sparse_attn || indices->stride(-1) == 1, "indices must have contiguous last dimension");
    TORCH_CHECK(!is_sparse_attn || indices_all_valid_per_q->dtype() == torch::kBool, "indices_all_valid_per_q must have dtype bool");
    TORCH_CHECK(!is_sparse_attn || indices_all_valid_per_q->stride(-1) == 1, "indices_all_valid_per_q must have contiguous last dimension");


    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q_ori = sizes[1];
    const int num_heads_ori = sizes[2];
    const int head_size = sizes[3];
    const int num_heads_k = kcache.size(2);
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");

    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q_ori == 1) { is_causal = false; }
    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    if (is_sparse_attn){
        TORCH_CHECK(num_heads_ori >= 64 || seqlen_q_ori == 1, "sparse decoding head q must greter than 64 when seqlen q > 1");
    }
    q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
            .reshape({batch_size, seqlen_q, num_heads, head_size});

    int head_size_k = head_size;
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    if (vcache_.has_value()) { CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_v); }
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);


    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    CHECK_DEVICE(seqlens_k);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(seqlens_k, batch_size);

    if (cp_tot_seqused_k_.has_value()) {
        auto cp_tot_seqused_k = cp_tot_seqused_k_.value();
        TORCH_CHECK(cp_tot_seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        CHECK_DEVICE(cp_tot_seqused_k); CHECK_CONTIGUOUS(cp_tot_seqused_k);
        CHECK_SHAPE(cp_tot_seqused_k, batch_size);
    }



    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    at::Tensor out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts);
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    mcFlashAttn::Flash_fwd_mla_params params = {};
    params.rotary_dim = 0;
    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    // params.seqlen_k = seqlens_k.max().cpu().item<int>();
    params.cu_seqlens_k = seqlens_k.data_ptr<int>();
    params.is_seqlens_k_cumulative = false; // seqlens_k always has value
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.is_sparse_attn = is_sparse_attn;
    params.topk = topk;
    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.v_ptr = vcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kcache.stride(0);
    params.v_batch_stride = vcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kcache.stride(-3);
    params.v_row_stride = vcache.stride(-3);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kcache.stride(-2);
    params.v_head_stride = vcache.stride(-2);
    params.o_head_stride = out.stride(-2);

    // indices ptr
    params.indices_ptr = is_sparse_attn ? indices->data_ptr<int32_t>() : nullptr;
    params.indices_batch_stride = is_sparse_attn ? indices->stride(0) : 0;
    params.indices_row_stride = is_sparse_attn ? indices->stride(1) : 0;

    params.indices_all_valid_per_q_ptr = is_sparse_attn ? indices_all_valid_per_q->data_ptr<bool>() : nullptr;
    params.indices_all_valid_per_q_batch_stride = is_sparse_attn ? indices_all_valid_per_q->stride(0) : 0;
    params.indices_all_valid_per_q_row_stride = is_sparse_attn ? indices_all_valid_per_q->stride(1) : 0;

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;
    params.arch = arch;

    params.cp_world_size = cp_world_size;
    params.cp_rank = cp_rank;
    params.cp_tot_seqused_k = cp_tot_seqused_k_.has_value() ? cp_tot_seqused_k_->data_ptr<int>() : nullptr;
    TORCH_CHECK(cp_world_size > 0, "cp_world_size must be positive, required by downstream unified code path. Use 1 if CP is not enabled.");
    TORCH_CHECK(cp_world_size != 1 || cp_rank == 0, "When context parallelism is disabled, cp_rank must be zero");
    TORCH_CHECK(cp_world_size == 1 || cp_tot_seqused_k_.has_value(), "cp_tot_seqused_k_ must be provided when context parallelism is enabled.");

    TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");
    // printf("num_splits%d",num_splits);
    CHECK_DEVICE(num_splits);
    CHECK_CONTIGUOUS(num_splits);

    TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    CHECK_DEVICE(tile_scheduler_metadata);
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_sm_parts = tile_scheduler_metadata.size(0);
    params.num_splits_ptr = num_splits.data_ptr<int>();
    at::Tensor softmax_lse_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor out_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q, head_size_v}, opts.dtype(at::kFloat));
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_ptr = out_accum.data_ptr();


    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(head_size == 576);
    params.is_bf16 = q_dtype == torch::kBFloat16;
    run_mla_fwd(params, stream);
    out = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(2, 3)
            .reshape({batch_size, num_heads_ori, seqlen_q_ori});
    return {out, softmax_lse};
}


std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v,
    const at::Tensor &indices_all_valid_per_q
) {
    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;

    CHECK_DEVICE(q);
    CHECK_DEVICE(kv);
    CHECK_DEVICE(indices);
    CHECK_DEVICE(indices_all_valid_per_q);

    TORCH_CHECK(q.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.dtype() == torch::kInt32);
    TORCH_CHECK(indices_all_valid_per_q.dtype() == torch::kBool);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int topk = indices.size(2);
    TORCH_CHECK(h_q % 64 == 0 && h_q >= 64);

    CHECK_SHAPE(q, s_q, h_q, d_qk);
    CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    CHECK_SHAPE(indices, s_q, h_kv, topk);
    CHECK_SHAPE(indices_all_valid_per_q, s_q, 1);

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(kv.stride(-1) == 1);
    TORCH_CHECK(indices.stride(-1) == 1);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    CHECK_CONTIGUOUS(out);

    at::Tensor buf_attn_score, max_logits, lse, p_sum;
    max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    CHECK_CONTIGUOUS(max_logits);
    CHECK_CONTIGUOUS(lse);

    SparsePrefillParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * 1.44269504f,
        arch,
        (mctlass::bfloat16_t*)q.data_ptr(),
        (mctlass::bfloat16_t*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        (bool*)indices_all_valid_per_q.data_ptr(),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(out.stride(0)),int64_stride_to_int(out.stride(1)),

        (mctlass::bfloat16_t*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        at::cuda::getCurrentCUDAStream().stream()
    };

    run_mla_fwd(params);

    return {out, max_logits, lse};
}

std::vector<at::Tensor>
get_mla_decoding_metadata(
    at::Tensor &seqlens_k,
    const int num_q_tokens_per_head_k,
    const int h_k,
    const std::optional<int> h_q,
    const bool is_fp8_kvcache,
    const std::optional<int> topk
) {
    auto dprops = flash::mcGetCurrentDeviceProperties();
    int arch = dprops.major * 100 + dprops.minor;
    // This should match the logic in the MLA kernel.
    const int seqlen_q  = num_q_tokens_per_head_k * h_k;
    bool is_sparse_attn = topk.has_value();
    const auto [block_size_m, block_size_n] = get_tile_size(arch, seqlen_q, is_sparse_attn);

    CHECK_DEVICE(seqlens_k);
    TORCH_CHECK(seqlens_k.is_contiguous());
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);
    if (is_sparse_attn)
        TORCH_CHECK(h_q.has_value(), "num_heads_q must be provided when topk is provided");

    CHECK_DEVICE(seqlens_k);
    TORCH_CHECK(seqlens_k.is_contiguous());
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);

    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data_ptr<int>();
    auto options = seqlens_k.options();

    int sm_count = dprops.multiProcessorCount;
    const char* val = std::getenv("FMLA_SM");
    if(val != nullptr){
        sm_count = std::stoi(val);
    }

    DecodingAttnImplMeta attn_impl_meta = get_attn_impl_meta(arch, sm_count, num_q_tokens_per_head_k, h_k, block_size_m, block_size_n, h_q, is_fp8_kvcache, is_sparse_attn);
    if(std::getenv("FMLA_LOG")){
        printf("block_size_m %d, num_q_tokens_per_head_k %d, h_k %d, seqlen_q %d, sm_count %d, sm_parts %d \n",
        block_size_m, num_q_tokens_per_head_k, h_k, seqlen_q, sm_count, attn_impl_meta.num_sm_parts);
    }

    auto tile_scheduler_metadata = torch::empty({attn_impl_meta.num_sm_parts, TileSchedulerMetaDataSize}, options);
    auto num_splits = torch::empty({batch_size + 1}, options);
    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();

    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    GetDecodingMetadataParams params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = attn_impl_meta.k_block_size;
    params.fixed_overhead_num_blocks = attn_impl_meta.fixed_overhead_num_blocks;
    params.num_sm_parts = attn_impl_meta.num_sm_parts;
    params.topk = is_sparse_attn ? topk.value() : -1;
    run_get_mla_metadata_kernel(params, stream);

    return {tile_scheduler_metadata, num_splits};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("get_mla_metadata", &get_mla_decoding_metadata);
    m.def("fwd_kvcache_mla", &fwd_kvcache_mla);
    m.def("sparse_prefill_fwd", &sparse_prefill_fwd);
}
