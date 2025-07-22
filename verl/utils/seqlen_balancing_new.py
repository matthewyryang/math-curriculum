def rearrange_micro_batches_with_dataproto(data: DataProto, max_token_len, dp_group=None):
    """Split the batch and non-tensor batch in DataProto into a list micro batches and micro non-tensor batches. In each micro batch the max_token_len is smaller than max_token_len
    and the number of valid tokens in each micro batch is well balanced.
    """
    batch = data.batch
    assert isinstance(batch, TensorDict), f"batch must be a TensorDict, got {type(batch)}"
    non_tensor_batch = data.non_tensor_batch
    assert isinstance(non_tensor_batch, dict), f"non_tensor_batch must be a dict, got {type(non_tensor_batch)}"

    # this is per local micro_bsz
    max_seq_len = batch['attention_mask'].shape[-1]
    assert max_token_len >= max_seq_len, \
        f'max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}'

    seq_len_effective: torch.Tensor = batch['attention_mask'].sum(dim=1)
    total_seqlen = seq_len_effective.sum().item()
    num_micro_batches = ceildiv(total_seqlen, max_token_len)
    if dist.is_initialized():
        num_micro_batches = torch.tensor([num_micro_batches], device='cuda')
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()

    seq_len_effective = seq_len_effective.tolist()
    assert num_micro_batches <= len(seq_len_effective)

    micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

    micro_batches = []
    micro_non_tensor_batches = []

    for partition in micro_bsz_idx:
        curr_micro_batch = []
        curr_micro_non_tensor_batch = {k: np.array([]) for k in non_tensor_batch.keys()}
        
        for idx in partition:
            curr_micro_batch.append(batch[idx:idx + 1])
            import numpy as np
            curr_micro_non_tensor_batch = {k: np.append(curr_micro_non_tensor_batch[k], non_tensor_batch[k][idx:idx + 1])
                                             for k in non_tensor_batch.keys()}
        curr_micro_non_tensor_batch = {k: np.concatenate([d[k] for d in curr_micro_non_tensor_batch]) for k in non_tensor_batch.keys()}
        curr_micro_batch = torch.cat(curr_micro_batch)

        micro_batches.append(curr_micro_batch)
        micro_non_tensor_batches.append(curr_micro_non_tensor_batch)

    return micro_batches, micro_non_tensor_batches, micro_bsz_idx
