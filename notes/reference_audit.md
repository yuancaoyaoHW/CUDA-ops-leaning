# Reference Material Consistency Audit

Audit date: 2026-05-31

Scope:
- `resources/README.md`
- `resources/reading-schedule.md`
- `resources/download_all.sh`
- `resources/interview-resources.md`
- `study-plan/progress.yaml` references section only
- `resources/papers/`

No downloads were attempted.

## Referenced PDF Filenames

### `resources/README.md`

`resources/README.md` does not use literal PDF filenames in its paper table. It references papers by number/title. Mapping those numbered paper entries to the repository filename convention yields:

- `01_online_softmax_milakov_2018.pdf`
- `02_zero_rajbhandari_2020.pdf`
- `03_flashattention_dao_2022.pdf`
- `04_flashattention2_dao_2023.pdf`
- `05_flashdecoding_dao_2023.pdf` (implied by README item 5; no matching file or download target found)
- `06_pagedattention_vllm_kwon_2023.pdf`
- `07_orca_yu_2022.pdf`
- `08_speculative_decoding_leviathan_2023.pdf`
- `09_staged_speculative_decoding_spector_2023.pdf`
- `10_sglang_zheng_2023.pdf`
- `11_mooncake_kvcache_disaggregated_2024.pdf`
- `12_ring_attention_liu_2023.pdf`
- `13_tensorrt_llm_2024.pdf`
- `14_gptq_frantar_2023.pdf`
- `15_awq_lin_2023.pdf`
- `16_fp8_formats_micikevicius_2022.pdf`
- `17_deepseek_v3_2024.pdf`
- `18_attention_is_all_you_need_2017.pdf`
- `19_roformer_rope_su_2021.pdf`
- `20_gqa_ainslie_2023.pdf`
- `21_lora_hu_2021.pdf`
- `22_qlora_dettmers_2023.pdf`

### `resources/reading-schedule.md`

- `01_online_softmax_milakov_2018.pdf`
- `02_zero_rajbhandari_2020.pdf`
- `03_flashattention_dao_2022.pdf`
- `04_flashattention2_dao_2023.pdf`
- `06_pagedattention_vllm_kwon_2023.pdf`
- `07_orca_yu_2022.pdf`
- `08_speculative_decoding_leviathan_2023.pdf`
- `09_staged_speculative_decoding_spector_2023.pdf`
- `10_sglang_zheng_2023.pdf`
- `11_mooncake_kvcache_disaggregated_2024.pdf`
- `12_ring_attention_liu_2023.pdf`
- `13_tensorrt_llm_2024.pdf`
- `14_gptq_frantar_2023.pdf`
- `15_awq_lin_2023.pdf`
- `16_fp8_formats_micikevicius_2022.pdf`
- `17_deepseek_v3_2024.pdf`
- `18_attention_is_all_you_need_2017.pdf`
- `19_roformer_rope_su_2021.pdf`
- `20_gqa_ainslie_2023.pdf`
- `21_lora_hu_2021.pdf`
- `22_qlora_dettmers_2023.pdf`
- `23_mqa_shazeer_2019.pdf`
- `24_deepseek_v2_mla_2024.pdf`
- `25_instructgpt_rlhf_ouyang_2022.pdf`
- `26_dpo_rafailov_2023.pdf`
- `28_rag_lewis_2020.pdf`

### `resources/interview-resources.md`

Literal filenames:

- `08_speculative_decoding_leviathan_2023.pdf`
- `10_sglang_zheng_2023.pdf`
- `11_mooncake_kvcache_disaggregated_2024.pdf`
- `17_deepseek_v3_2024.pdf`
- `24_deepseek_v2_mla_2024.pdf`

Wildcard filename references:

- `03_*.pdf` (matches `03_flashattention_dao_2022.pdf`)
- `04_*.pdf` (matches `04_flashattention2_dao_2023.pdf`)
- `13_*.pdf` (intended match: `13_tensorrt_llm_2024.pdf`)
- `14_*.pdf` (matches `14_gptq_frantar_2023.pdf`)
- `15_*.pdf` (matches `15_awq_lin_2023.pdf`)

### `study-plan/progress.yaml`

- `01_online_softmax_milakov_2018.pdf`
- `02_zero_rajbhandari_2020.pdf`
- `03_flashattention_dao_2022.pdf`
- `04_flashattention2_dao_2023.pdf`
- `06_pagedattention_vllm_kwon_2023.pdf`
- `07_orca_yu_2022.pdf`
- `08_speculative_decoding_leviathan_2023.pdf`
- `09_staged_speculative_decoding_spector_2023.pdf`
- `10_sglang_zheng_2023.pdf`
- `11_mooncake_kvcache_disaggregated_2024.pdf`
- `12_ring_attention_liu_2023.pdf`
- `14_gptq_frantar_2023.pdf`
- `15_awq_lin_2023.pdf`
- `16_fp8_formats_micikevicius_2022.pdf`
- `17_deepseek_v3_2024.pdf`
- `18_attention_is_all_you_need_2017.pdf`
- `19_roformer_rope_su_2021.pdf`
- `20_gqa_ainslie_2023.pdf`
- `21_lora_hu_2021.pdf`
- `22_qlora_dettmers_2023.pdf`

### `resources/download_all.sh`

Paper targets:

- `01_online_softmax_milakov_2018.pdf`
- `02_zero_rajbhandari_2020.pdf`
- `03_flashattention_dao_2022.pdf`
- `04_flashattention2_dao_2023.pdf`
- `06_pagedattention_vllm_kwon_2023.pdf`
- `07_orca_yu_2022.pdf`
- `08_speculative_decoding_leviathan_2023.pdf`
- `09_staged_speculative_decoding_spector_2023.pdf`
- `10_sglang_zheng_2023.pdf`
- `11_mooncake_kvcache_disaggregated_2024.pdf`
- `12_ring_attention_liu_2023.pdf`
- `13_tensorrt_llm_2024.pdf`
- `14_gptq_frantar_2023.pdf`
- `15_awq_lin_2023.pdf`
- `16_fp8_formats_micikevicius_2022.pdf`
- `17_deepseek_v3_2024.pdf`
- `18_attention_is_all_you_need_2017.pdf`
- `19_roformer_rope_su_2021.pdf`
- `20_gqa_ainslie_2023.pdf`
- `21_lora_hu_2021.pdf`
- `22_qlora_dettmers_2023.pdf`
- `23_mqa_shazeer_2019.pdf`
- `24_deepseek_v2_mla_2024.pdf`
- `25_instructgpt_rlhf_ouyang_2022.pdf`
- `26_dpo_rafailov_2023.pdf`
- `27_ppo_schulman_2017.pdf`
- `28_rag_lewis_2020.pdf`
- `29_bert_devlin_2018.pdf`

Manual/document PDF targets outside `resources/papers/`:

- `CUDA_C_Programming_Guide.pdf`
- `NsightCompute.pdf`
- `NsightSystems.pdf`

## Actual PDFs Present Under `resources/papers/`

- `01_online_softmax_milakov_2018.pdf`
- `02_zero_rajbhandari_2020.pdf`
- `03_flashattention_dao_2022.pdf`
- `04_flashattention2_dao_2023.pdf`
- `06_pagedattention_vllm_kwon_2023.pdf`
- `07_orca_yu_2022.pdf`
- `08_speculative_decoding_leviathan_2023.pdf`
- `09_staged_speculative_decoding_spector_2023.pdf`
- `10_sglang_zheng_2023.pdf`
- `11_mooncake_kvcache_disaggregated_2024.pdf`
- `12_ring_attention_liu_2023.pdf`
- `14_gptq_frantar_2023.pdf`
- `15_awq_lin_2023.pdf`
- `16_fp8_formats_micikevicius_2022.pdf`
- `17_deepseek_v3_2024.pdf`
- `18_attention_is_all_you_need_2017.pdf`
- `19_roformer_rope_su_2021.pdf`
- `20_gqa_ainslie_2023.pdf`
- `21_lora_hu_2021.pdf`
- `22_qlora_dettmers_2023.pdf`

## Referenced-but-missing PDFs

Missing from `resources/papers/`:

- `05_flashdecoding_dao_2023.pdf` (README-implied only; README calls this a blog post, not a paper PDF)
- `13_tensorrt_llm_2024.pdf`
- `23_mqa_shazeer_2019.pdf`
- `24_deepseek_v2_mla_2024.pdf`
- `25_instructgpt_rlhf_ouyang_2022.pdf`
- `26_dpo_rafailov_2023.pdf`
- `27_ppo_schulman_2017.pdf`
- `28_rag_lewis_2020.pdf`
- `29_bert_devlin_2018.pdf`

## Present-but-not-referenced PDFs

None. Every PDF currently present under `resources/papers/` is referenced by at least one audited source.

## Mismatches Between README/Reading-Schedule/Progress/Download-All

- `resources/README.md` lists papers 1-22, including item 5 FlashDecoding and item 13 TensorRT-LLM, but uses titles instead of filenames. `resources/download_all.sh` has no `05_*.pdf` target and treats FlashDecoding as a blog/manual resource, while it does include `13_tensorrt_llm_2024.pdf`.
- `study-plan/progress.yaml` references only papers 1-22 except item 13. It omits `13_tensorrt_llm_2024.pdf`, `23_mqa_shazeer_2019.pdf`, `24_deepseek_v2_mla_2024.pdf`, `25_instructgpt_rlhf_ouyang_2022.pdf`, `26_dpo_rafailov_2023.pdf`, `27_ppo_schulman_2017.pdf`, `28_rag_lewis_2020.pdf`, and `29_bert_devlin_2018.pdf`.
- `resources/reading-schedule.md` references `13_tensorrt_llm_2024.pdf`, `23_mqa_shazeer_2019.pdf`, `24_deepseek_v2_mla_2024.pdf`, `25_instructgpt_rlhf_ouyang_2022.pdf`, `26_dpo_rafailov_2023.pdf`, and `28_rag_lewis_2020.pdf`, but none of those files are currently present under `resources/papers/`.
- `resources/download_all.sh` includes `27_ppo_schulman_2017.pdf` and `29_bert_devlin_2018.pdf`, but these are not referenced by `README.md`, `reading-schedule.md`, or `progress.yaml`.
- `resources/interview-resources.md` references `24_deepseek_v2_mla_2024.pdf` and wildcard `13_*.pdf`; both point to files that `download_all.sh` can download but that are absent locally.
- `resources/interview-resources.md` wildcard patterns are less durable than explicit filenames. `03_*.pdf`, `04_*.pdf`, `13_*.pdf`, `14_*.pdf`, and `15_*.pdf` depend on there being exactly one compatible paper per prefix.

## Suggested Fixes

- Decide whether FlashDecoding should remain a blog-only resource. If so, remove it from the README paper table or mark it explicitly as a blog/no local PDF instead of implying `05_flashdecoding_dao_2023.pdf`.
- Add the supplemental papers 23-29 and TensorRT-LLM 13 consistently across README, reading schedule, and progress references, or remove them from `download_all.sh` if they are out of scope.
- Prefer explicit filenames in `resources/interview-resources.md` instead of wildcard references so source coverage can be checked mechanically.
- If local source completeness is required before Tasks 2-12, obtain the missing PDFs through the existing script or manual process in a separate setup step. This audit intentionally did not download anything.
- Keep `resources/papers/` and the `study-plan/progress.yaml` references section in sync whenever a new paper becomes part of the study plan.

## Subagent audit checklist

Use this checklist when auditing Tasks 2-12:

- Source coverage: every factual claim names a source file, paper, official doc, or local note used for that claim.
- Path existence: every local path mentioned by the task exists, or the report explicitly marks it missing.
- PDF coverage: every cited PDF exists under `resources/papers/`, or the report lists it under referenced-but-missing.
- Cross-reference consistency: README, reading schedule, progress references, and download script agree on required paper filenames for the task area.
- No unsupported claims: summaries avoid adding performance numbers, company practices, or implementation details that are not present in the cited source.
- No forbidden setup actions: do not download model weights, install packages, clone large repos, or run system package installation.
- Verification: run the narrowest relevant command and record exact command names plus pass/fail result.
