# CUDA Performance 事故演练 Playbook

---

## 事故 1：GPU Utilization High but Throughput Low（GPU 利用率高但吞吐低）

### Symptom（现象）
- `nvidia-smi` 显示 GPU utilization 90%+
- 但实际 tokens/s throughput 远低于预期
- SM occupancy 看起来正常
- 用户感知到"GPU 很忙但出活慢"

### Possible Root Causes（可能原因）
1. **Memory-bound kernel**: kernel 是 memory-bound 的，GPU 在等待内存访问，compute unit 空转但 utilization 显示高（因为有 kernel 在跑）
2. **Kernel launch overhead**: 大量小 kernel 频繁 launch，GPU 利用率高但有效计算少
3. **CPU-GPU 同步瓶颈**: 频繁的 `cuda.synchronize()` 导致 GPU pipeline bubble
4. **低效 memory access pattern**: non-coalesced memory access，带宽利用率低
5. **Warp divergence**: 条件分支导致 warp 内线程执行不同路径
6. **Register spilling**: 寄存器溢出到 local memory，增加延迟
7. **Bank conflict**: shared memory bank conflict 导致序列化访问

### Metrics to Check
```
# Nsight Compute 指标
- sm__throughput.avg.pct_of_peak_sustained_elapsed  # SM 吞吐
- dram__throughput.avg.pct_of_peak_sustained_elapsed  # DRAM 吞吐
- gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
- sm__warps_active.avg.pct_of_peak_sustained_active  # Occupancy
- l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  # Global load
- smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct  # Coalescing

# nvidia-smi 指标
- gpu_utilization (这个指标有误导性！)
- memory_bandwidth_utilization
- sm_clock / memory_clock

# 应用指标
- tokens_per_second
- requests_per_second
- kernel_execution_time_breakdown
```

### Logs to Check
```bash
# 检查 kernel launch 频率
nsys stats --report cuda_kern_exec_trace profile.nsys-rep | head -50

# 检查 CPU-GPU sync
nsys stats --report cuda_api_trace profile.nsys-rep | grep -i "sync\|memcpy"

# 检查 kernel 大小分布
nsys stats --report cuda_kern_exec_trace profile.nsys-rep | awk '{print $NF}' | sort -n
```

### Profiling Method
```bash
# 1. Nsight Systems — 全局 timeline 分析
nsys profile --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  -o gpu_util_debug \
  python inference_server.py

# 2. Nsight Compute — kernel 级别分析
ncu --set full \
  --target-processes all \
  --launch-skip 100 --launch-count 10 \
  -o kernel_analysis \
  python inference_server.py

# 3. Roofline 分析
ncu --set roofline \
  -o roofline_analysis \
  python inference_server.py

# 4. 检查 memory throughput vs compute throughput
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python inference_server.py
```

### Immediate Mitigation
1. **确认瓶颈类型**: 用 NCU roofline 确认是 memory-bound 还是 compute-bound
2. **减少 kernel launch**: 使用 CUDA Graph 减少 launch overhead
3. **增加 batch size**: 如果是 memory-bound，增加 batch 可以提高 arithmetic intensity
4. **关闭 debug 模式**: 确保没有额外的 sync/logging

### Rollback
- 回滚到上一个性能正常的代码版本
- 恢复之前的 CUDA Graph 配置
- 恢复之前的 batch size 设置

### Long-term Fix
1. **Kernel fusion**: 合并小 kernel 减少 launch overhead 和中间内存访问
2. **Memory access 优化**: 确保 coalesced access，使用 shared memory 缓存
3. **CUDA Graph**: 将推理 pipeline 编译为 CUDA Graph
4. **Operator 重排**: 调整算子执行顺序减少内存访问
5. **Mixed precision**: 使用 FP16/BF16/FP8 减少内存带宽需求
6. **Custom kernel**: 对热点 kernel 手写优化版本

### Postmortem
- **关键发现**: GPU utilization ≠ GPU efficiency
- **教训**: 需要区分 "GPU 在忙" 和 "GPU 在做有效工作"
- **指标改进**: 用 SM throughput + DRAM throughput 替代 nvidia-smi utilization

### Interview Answer
> "GPU utilization 高但 throughput 低是一个经典的性能陷阱。nvidia-smi 的 utilization 只表示'有 kernel 在跑'，不代表 GPU 在高效工作。
> 
> 我的排查步骤：
> 1. 用 Nsight Compute 做 roofline 分析，确认 kernel 是 memory-bound 还是 compute-bound
> 2. 如果是 memory-bound：检查 DRAM throughput 是否接近峰值带宽。如果远低于峰值，说明 memory access pattern 有问题（non-coalesced, bank conflict）
> 3. 用 Nsight Systems 看 timeline：如果有大量小 kernel 和 gap，说明 launch overhead 是瓶颈
> 4. 检查 CPU-GPU sync：频繁的 synchronize 会导致 GPU pipeline bubble
> 
> 具体案例：一个 decode attention kernel 的 GPU utilization 95% 但 throughput 只有理论峰值的 30%。NCU 显示 DRAM throughput 只有 40% of peak，原因是 KV cache 的 memory layout 导致 non-coalesced access。优化 memory layout 后 throughput 提升 2.1×。"

---

## 事故 2：CUDA Kernel Regression（CUDA Kernel 性能回归）

### Symptom（现象）
- 某个 kernel 的执行时间突然增加（如 attention kernel 从 0.5ms 变为 1.2ms）
- 整体推理延迟增加
- 代码更新或依赖升级后出现
- 特定输入 shape 下性能下降

### Possible Root Causes
1. **依赖升级**: FlashAttention/FlashInfer/cuDNN 版本更新引入回归
2. **编译器变更**: CUDA toolkit 升级导致 kernel 编译结果不同
3. **Autotuning 失效**: Triton kernel 的 autotuning 配置在新环境下不适用
4. **Shape 变化**: 输入 tensor shape 变化导致 kernel 选择了次优实现
5. **Driver 更新**: NVIDIA driver 更新影响 kernel 调度
6. **硬件差异**: 部署到不同 GPU 型号（如 A100 → H100）未重新 tune

### Metrics to Check
```
# Kernel 级别
- kernel_execution_time_ms (per kernel name)
- kernel_occupancy
- kernel_memory_throughput
- kernel_compute_throughput

# 对比指标
- before_vs_after_kernel_time
- before_vs_after_memory_throughput
- before_vs_after_register_usage
```

### Logs to Check
```bash
# 检查依赖版本变化
pip list | grep -i "flash\|triton\|torch\|cuda"
nvcc --version
nvidia-smi | grep "Driver Version"

# 检查 kernel 选择
grep "kernel_select\|dispatch" /var/log/inference/debug.log

# Triton autotuning cache
ls ~/.triton/cache/
```

### Profiling Method
```bash
# 1. A/B 对比 profiling
# 旧版本
ncu --set full -o kernel_before python benchmark_kernel.py
# 新版本
ncu --set full -o kernel_after python benchmark_kernel.py

# 2. 对比分析
ncu --page details --set full kernel_before.ncu-rep kernel_after.ncu-rep

# 3. 检查 register 使用变化
ncu --metrics launch__registers_per_thread kernel_after.ncu-rep

# 4. 检查 occupancy 变化
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active kernel_after.ncu-rep

# 5. Triton kernel 重新 autotuning
python -c "import triton; triton.runtime.cache.default_cache_dir()"
rm -rf ~/.triton/cache/  # 清除旧 cache
python benchmark_kernel.py  # 重新 autotune
```

### Immediate Mitigation
1. **回滚依赖版本**: `pip install flash-attn==2.x.x` 回滚到已知好的版本
2. **清除 Triton cache**: 删除 `~/.triton/cache/` 强制重新编译
3. **固定 kernel 选择**: 如果有多个 kernel 实现，手动指定已知快的版本
4. **降级 driver**: 如果是 driver 问题，回滚 NVIDIA driver

### Rollback
- `pip install flash-attn==<previous_version>`
- `pip install triton==<previous_version>`
- 回滚 CUDA toolkit 版本
- 回滚 NVIDIA driver

### Long-term Fix
1. **Kernel benchmark CI**: 每次依赖更新自动跑 kernel benchmark
2. **版本锁定**: 在 requirements.txt 中锁定精确版本
3. **多版本 kernel**: 保留多个 kernel 实现，运行时选择最快的
4. **Autotuning 持久化**: 将 autotuning 结果保存并版本化
5. **硬件感知部署**: 不同 GPU 型号使用不同的 kernel 配置

### Interview Answer
> "Kernel 性能回归的排查我会用 A/B 对比方法：
> 1. 首先用 NCU 分别 profile 回归前后的 kernel，对比 execution time, occupancy, memory throughput
> 2. 检查 register 使用量变化——如果新版本 register 增加导致 occupancy 下降
> 3. 检查 memory access pattern——编译器优化可能改变了 access pattern
> 
> 预防措施：建立 kernel benchmark CI，每次依赖更新自动对比性能。我们设置了 5% 的回归阈值，超过就阻止合入。同时将 Triton autotuning 结果持久化到 CI artifact 中。"

---

## 事故 3：Quantization Accuracy Regression（量化精度回归）

### Symptom（现象）
- 模型输出质量下降（如 MMLU 分数下降 2%+）
- 用户反馈"回答变差了"
- 量化模型与 FP16 baseline 的 perplexity gap 增大
- 特定类型的问题（如数学、代码）准确率明显下降

### Possible Root Causes
1. **量化配置错误**: calibration dataset 不合适或 calibration 步骤不足
2. **新模型不适配**: 新模型的权重分布与量化方案不匹配
3. **量化粒度问题**: per-tensor 量化对某些层不够精确
4. **Outlier 处理失败**: 权重/激活中的 outlier 未被正确处理
5. **Dequant kernel bug**: 反量化 kernel 的数值精度问题
6. **混合精度配置**: 某些敏感层（如 attention QKV）被错误量化
7. **Calibration data drift**: calibration 数据与实际推理数据分布不同

### Metrics to Check
```
# 精度指标
- model_perplexity (FP16 vs quantized)
- mmlu_score
- humaneval_pass_rate
- math_accuracy
- per_layer_quantization_error

# 数值指标
- max_abs_error_per_layer
- kl_divergence_per_layer (FP16 vs quantized output distribution)
- outlier_percentage_per_layer
- activation_range_per_layer
```

### Logs to Check
```bash
# 量化配置
cat quantization_config.json

# Calibration 日志
grep "calibration\|scale\|zero_point" /var/log/quantize/calibration.log

# 精度测试结果
cat evaluation_results.json | jq '.mmlu, .humaneval, .perplexity'

# 检查 outlier
python check_weight_distribution.py --model quantized_model/ --layer all
```

### Profiling Method
```bash
# 1. 逐层精度分析
python per_layer_accuracy.py \
  --fp16_model original/ \
  --quant_model quantized/ \
  --dataset calibration_data.json

# 2. 找到精度损失最大的层
python find_sensitive_layers.py \
  --model quantized/ \
  --metric kl_divergence

# 3. 对比不同量化方案
python compare_quantization.py \
  --methods "awq,gptq,smoothquant" \
  --model original/ \
  --eval_dataset mmlu

# 4. Activation 分布分析
python analyze_activations.py \
  --model original/ \
  --dataset calibration_data.json \
  --output activation_stats.json
```

### Immediate Mitigation
1. **回滚到 FP16**: 如果精度不可接受，临时切回 FP16 模型
2. **混合精度**: 将敏感层（attention, lm_head）保持 FP16
3. **提高量化精度**: 从 INT4 切换到 INT8
4. **更换 calibration data**: 使用更接近实际分布的数据重新 calibrate

### Rollback
- 切回 FP16 模型部署
- 回滚到上一个量化版本
- 恢复之前的量化配置

### Long-term Fix
1. **敏感层检测**: 自动识别对量化敏感的层，保持高精度
2. **混合量化**: 不同层使用不同量化精度（如 attention FP16, MLP INT8）
3. **Calibration 优化**: 使用更大/更多样的 calibration dataset
4. **SmoothQuant**: 对 activation outlier 使用 smoothing
5. **精度监控**: 部署后持续监控模型输出质量
6. **A/B 测试**: 量化模型上线前做 A/B 测试验证精度

### Interview Answer
> "量化精度回归的排查：
> 1. 首先做逐层 KL divergence 分析，找到精度损失最大的层
> 2. 检查这些层的权重/激活分布——是否有 outlier 导致量化误差放大
> 3. 检查 calibration data 是否覆盖了实际推理的数据分布
> 
> 解决方案：对敏感层使用混合精度（如 attention 的 QKV projection 保持 FP16），对有 outlier 的层使用 SmoothQuant 平滑激活分布。我们建立了量化后的自动评测 pipeline：perplexity < 0.1 gap, MMLU < 1% drop 才允许部署。"
