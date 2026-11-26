from gguf import GGUFWriter
names = [n for n in dir(GGUFWriter) if not n.startswith("_")]
for want in ("add_name", "add_context_length", "add_embedding_length", "add_block_count", "add_head_count",
             "add_head_count_kv", "add_feed_forward_length", "add_rope_dimension_count", "add_rope_freq_base",
             "add_layer_norm_rms_eps", "add_file_type", "add_tokenizer_model", "add_tokenizer_pre",
             "add_token_list", "add_token_scores", "add_token_types", "add_bos_token_id", "add_eos_token_id",
             "add_unk_token_id", "add_add_bos_token", "add_add_eos_token", "add_tensor",
             "write_header_to_file", "write_kv_data_to_file", "write_tensors_to_file", "close"):
    print(want, "OK" if want in names else "MISSING")
