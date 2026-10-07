{
  "_runtime": 32916.779120943,
  "_step": 47,
  "_timestamp": 1790393595.022334,
  "_wandb.runtime": 32916,
  "eval/loss": 2.0688629150390625,
  "eval/runtime": 416.4735,
  "eval/samples_per_second": 0.339,
  "eval/steps_per_second": 0.17,
  "total_flos": 115576051934103550,
  "train/epoch": 3,
  "train/global_step": 423,
  "train/grad_norm": 0.40228986740112305,
  "train/learning_rate": 5.6745442994576854e-8,
  "train/loss": 1.9448617935180663,
  "train_loss": 2.1557215573376234,
  "train_runtime": 32918.4838,
  "train_samples_per_second": 0.102,
  "train_steps_per_second": 0.013
}


{
  "bf16": {
    "value": false
  },
  "fp16": {
    "value": true
  },
  "fsdp": {
    "value": []
  },
  "seed": {
    "value": 42
  },
  "tf32": {
    "value": null
  },
  "debug": {
    "value": []
  },
  "dtype": {
    "value": "float16"
  },
  "optim": {
    "value": "adamw_8bit"
  },
  "_wandb": {
    "value": {
      "m": [
        {
          "1": "train/global_step",
          "6": [
            3
          ],
          "7": []
        },
        {
          "2": "*",
          "5": 1,
          "6": [
            1
          ],
          "7": []
        }
      ],
      "t": {
        "1": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          63,
          71,
          75,
          84,
          95,
          98
        ],
        "2": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          63,
          71,
          75,
          84,
          95,
          98
        ],
        "3": [
          7,
          13,
          19,
          62,
          66
        ],
        "4": "3.12.13",
        "5": "0.26.1",
        "6": "5.5.0",
        "8": [
          2,
          13
        ],
        "9": {
          "1": "transformers_trainer"
        },
        "12": "0.26.1",
        "13": "linux-x86_64"
      },
      "cli_version": "0.26.1",
      "python_version": "3.12.13"
    }
  },
  "do_eval": {
    "value": true
  },
  "packing": {
    "value": true
  },
  "project": {
    "value": "huggingface"
  },
  "use_cpu": {
    "value": false
  },
  "do_train": {
    "value": false
  },
  "head_dim": {
    "value": 128
  },
  "id2label": {
    "value": {
      "0": "LABEL_0",
      "1": "LABEL_1"
    }
  },
  "label2id": {
    "value": {
      "LABEL_0": 0,
      "LABEL_1": 1
    }
  },
  "mlp_bias": {
    "value": false
  },
  "run_name": {
    "value": "medquad-qlora-kaggle"
  },
  "data_seed": {
    "value": 3407
  },
  "deepspeed": {
    "value": null
  },
  "eos_token": {
    "value": "<EOS_TOKEN>"
  },
  "hub_token": {
    "value": "<HUB_TOKEN>"
  },
  "log_level": {
    "value": "passive"
  },
  "loss_type": {
    "value": "nll"
  },
  "max_steps": {
    "value": -1
  },
  "pad_token": {
    "value": "<PAD_TOKEN>"
  },
  "report_to": {
    "value": [
      "wandb"
    ]
  },
  "use_cache": {
    "value": false
  },
  "adam_beta1": {
    "value": 0.9
  },
  "adam_beta2": {
    "value": 0.999
  },
  "do_predict": {
    "value": false
  },
  "eval_delay": {
    "value": 0
  },
  "eval_steps": {
    "value": 100
  },
  "hidden_act": {
    "value": "silu"
  },
  "local_rank": {
    "value": -1
  },
  "max_length": {
    "value": 2048
  },
  "model_type": {
    "value": "llama"
  },
  "optim_args": {
    "value": null
  },
  "output_dir": {
    "value": "/kaggle/working/medquad-lora"
  },
  "save_steps": {
    "value": 200
  },
  "vocab_size": {
    "value": 128256
  },
  "ddp_backend": {
    "value": null
  },
  "ddp_timeout": {
    "value": 1800
  },
  "fsdp_config": {
    "value": {
      "xla": false,
      "xla_fsdp_v2": false,
      "min_num_params": 0,
      "xla_fsdp_grad_ckpt": false
    }
  },
  "hidden_size": {
    "value": 3072
  },
  "label_names": {
    "value": null
  },
  "logging_dir": {
    "value": null
  },
  "peft_config": {
    "value": {
      "default": {
        "r": 16,
        "bias": "none",
        "revision": null,
        "use_dora": false,
        "lora_bias": false,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "eva_config": null,
        "lora_alpha": 32,
        "use_bdlora": null,
        "use_qalora": false,
        "use_rslora": false,
        "arrow_config": null,
        "auto_mapping": {
          "unsloth_fixed": true,
          "parent_library": "transformers.models.llama.modeling_llama",
          "base_model_class": "LlamaForCausalLM"
        },
        "corda_config": null,
        "lora_dropout": 0.05,
        "peft_version": "0.19.1",
        "megatron_core": "megatron.core",
        "fan_in_fan_out": false,
        "inference_mode": false,
        "layers_pattern": null,
        "lora_ga_config": null,
        "runtime_config": {
          "ephemeral_gpu_offload": false
        },
        "target_modules": [
          "k_proj",
          "o_proj",
          "down_proj",
          "q_proj",
          "gate_proj",
          "up_proj",
          "v_proj"
        ],
        "exclude_modules": null,
        "megatron_config": null,
        "modules_to_save": null,
        "init_lora_weights": true,
        "layer_replication": null,
        "qalora_group_size": 16,
        "target_parameters": null,
        "ensure_weight_tying": false,
        "layers_to_transform": null,
        "alora_invocation_tokens": null,
        "base_model_name_or_path": "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit",
        "trainable_token_indices": null
      }
    }
  },
  "push_to_hub": {
    "value": false
  },
  "return_dict": {
    "value": true
  },
  "adam_epsilon": {
    "value": 1e-8
  },
  "bos_token_id": {
    "value": 128000
  },
  "disable_tqdm": {
    "value": false
  },
  "eos_token_id": {
    "value": 128009
  },
  "eval_packing": {
    "value": null
  },
  "hub_model_id": {
    "value": null
  },
  "hub_revision": {
    "value": null
  },
  "hub_strategy": {
    "value": "every_save"
  },
  "pad_token_id": {
    "value": 128004
  },
  "padding_free": {
    "value": true
  },
  "problem_type": {
    "value": null
  },
  "rms_norm_eps": {
    "value": 0.00001
  },
  "warmup_ratio": {
    "value": null
  },
  "warmup_steps": {
    "value": 50
  },
  "weight_decay": {
    "value": 0.001
  },
  "_name_or_path": {
    "value": "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
  },
  "architectures": {
    "value": [
      "LlamaForCausalLM"
    ]
  },
  "eval_on_start": {
    "value": false
  },
  "eval_strategy": {
    "value": "steps"
  },
  "learning_rate": {
    "value": 0.0002
  },
  "logging_steps": {
    "value": 10
  },
  "max_grad_norm": {
    "value": 1
  },
  "save_strategy": {
    "value": "steps"
  },
  "torch_compile": {
    "value": false
  },
  "unsloth_fixed": {
    "value": true
  },
  "attention_bias": {
    "value": false
  },
  "bf16_full_eval": {
    "value": false
  },
  "dataset_kwargs": {
    "value": null
  },
  "fp16_full_eval": {
    "value": true
  },
  "max_seq_length": {
    "value": 2048
  },
  "pretraining_tp": {
    "value": 1
  },
  "hub_always_push": {
    "value": false
  },
  "rope_parameters": {
    "value": {
      "factor": 32,
      "rope_type": "llama3",
      "rope_theta": 500000,
      "low_freq_factor": 1,
      "high_freq_factor": 4,
      "original_max_position_embeddings": 8192
    }
  },
  "save_only_model": {
    "value": false
  },
  "unsloth_version": {
    "value": "2026.9.11"
  },
  "dataset_num_proc": {
    "value": 2
  },
  "full_determinism": {
    "value": false
  },
  "hub_private_repo": {
    "value": null
  },
  "ignore_data_skip": {
    "value": false
  },
  "log_on_each_node": {
    "value": true
  },
  "logging_strategy": {
    "value": "steps"
  },
  "num_train_epochs": {
    "value": 3
  },
  "packing_strategy": {
    "value": "bfd"
  },
  "save_total_limit": {
    "value": 2
  },
  "trackio_space_id": {
    "value": "trackio"
  },
  "use_liger_kernel": {
    "value": false
  },
  "attention_dropout": {
    "value": 0
  },
  "ddp_bucket_cap_mb": {
    "value": null
  },
  "greater_is_better": {
    "value": false
  },
  "initializer_range": {
    "value": 0.02
  },
  "intermediate_size": {
    "value": 8192
  },
  "log_level_replica": {
    "value": "warning"
  },
  "lr_scheduler_type": {
    "value": "cosine"
  },
  "model_init_kwargs": {
    "value": null
  },
  "num_hidden_layers": {
    "value": 28
  },
  "output_attentions": {
    "value": false
  },
  "push_to_hub_token": {
    "value": null
  },
  "save_on_each_node": {
    "value": false
  },
  "accelerator_config": {
    "value": {
      "even_batches": true,
      "non_blocking": false,
      "split_batches": false,
      "dispatch_batches": null,
      "use_seedable_sampler": true,
      "gradient_accumulation_kwargs": null
    }
  },
  "batch_eval_metrics": {
    "value": false
  },
  "chat_template_path": {
    "value": null
  },
  "dataset_text_field": {
    "value": "text"
  },
  "is_encoder_decoder": {
    "value": false
  },
  "length_column_name": {
    "value": "length"
  },
  "logging_first_step": {
    "value": false
  },
  "pad_to_multiple_of": {
    "value": null
  },
  "parallelism_config": {
    "value": null
  },
  "torch_compile_mode": {
    "value": null
  },
  "unsloth_num_chunks": {
    "value": -1
  },
  "assistant_only_loss": {
    "value": false
  },
  "include_for_metrics": {
    "value": []
  },
  "liger_kernel_config": {
    "value": null
  },
  "lr_scheduler_kwargs": {
    "value": null
  },
  "neftune_noise_alpha": {
    "value": null
  },
  "num_attention_heads": {
    "value": 24
  },
  "num_key_value_heads": {
    "value": 8
  },
  "quantization_config": {
    "value": {
      "load_in_4bit": true,
      "load_in_8bit": false,
      "quant_method": "bitsandbytes",
      "llm_int8_threshold": 6,
      "bnb_4bit_quant_type": "nf4",
      "llm_int8_skip_modules": [
        "lm_head",
        "multi_modal_projector",
        "merger",
        "modality_projection",
        "model.layers.1.mlp"
      ],
      "bnb_4bit_compute_dtype": "float16",
      "llm_int8_has_fp16_weight": false,
      "bnb_4bit_use_double_quant": true,
      "llm_int8_enable_fp32_cpu_offload": false
    }
  },
  "skip_memory_metrics": {
    "value": true
  },
  "tie_word_embeddings": {
    "value": true
  },
  "auto_find_batch_size": {
    "value": false
  },
  "completion_only_loss": {
    "value": null
  },
  "dataloader_drop_last": {
    "value": false
  },
  "model/num_parameters": {
    "value": 3237063680
  },
  "optim_target_modules": {
    "value": null
  },
  "output_hidden_states": {
    "value": false
  },
  "prediction_loss_only": {
    "value": false
  },
  "transformers_version": {
    "value": "5.5.0"
  },
  "vllm_sampling_params": {
    "value": null
  },
  "activation_offloading": {
    "value": false
  },
  "dataloader_pin_memory": {
    "value": true
  },
  "ddp_broadcast_buffers": {
    "value": null
  },
  "enable_jit_checkpoint": {
    "value": false
  },
  "metric_for_best_model": {
    "value": "loss"
  },
  "remove_unused_columns": {
    "value": false
  },
  "torch_compile_backend": {
    "value": null
  },
  "dataloader_num_workers": {
    "value": 2
  },
  "eval_do_concat_batches": {
    "value": true
  },
  "eval_use_gather_object": {
    "value": false
  },
  "gradient_checkpointing": {
    "value": true
  },
  "label_smoothing_factor": {
    "value": 0
  },
  "load_best_model_at_end": {
    "value": true
  },
  "logging_nan_inf_filter": {
    "value": false
  },
  "resume_from_checkpoint": {
    "value": null
  },
  "chunk_size_feed_forward": {
    "value": 0
  },
  "eval_accumulation_steps": {
    "value": 2
  },
  "max_position_embeddings": {
    "value": 131072
  },
  "torch_empty_cache_steps": {
    "value": 250
  },
  "train_sampling_strategy": {
    "value": "random"
  },
  "unsloth_grpo_mini_batch": {
    "value": null
  },
  "dataloader_prefetch_factor": {
    "value": null
  },
  "ddp_find_unused_parameters": {
    "value": null
  },
  "per_device_eval_batch_size": {
    "value": 2
  },
  "gradient_accumulation_steps": {
    "value": 4
  },
  "per_device_train_batch_size": {
    "value": 2
  },
  "average_tokens_across_devices": {
    "value": true
  },
  "dataloader_persistent_workers": {
    "value": false
  },
  "gradient_checkpointing_kwargs": {
    "value": {
      "use_reentrant": true
    }
  },
  "include_num_input_tokens_seen": {
    "value": "no"
  },
  "unsloth_logit_chunk_multiplier": {
    "value": null
  },
  "restore_callback_states_from_checkpoint": {
    "value": false
  }
}