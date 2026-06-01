# GPU Cluster 事故演练 Playbook

---

## 事故 1：NCCL Timeout（NCCL 通信超时）

### Symptom（现象）
- 分布式推理/训练任务 hang 住或 crash
- 错误日志出现 `NCCL timeout` 或 `NCCL watchdog timeout`
- 某些 rank 正常，其他 rank 无响应
- GPU 利用率突然降为 0（等待通信）
- 任务在 all-reduce/all-gather 操作时卡住

### Possible Root Causes（可能原因）
1. **网络故障**: InfiniBand/RoCE 链路故障或丢包
2. **GPU 故障**: 某个 GPU 硬件错误（ECC error, XID error）
3. **进程 crash**: 某个 rank 的进程 OOM 或 segfault
4. **网络拥塞**: 其他任务占用网络带宽
5. **NCCL 版本不兼容**: 不同节点的 NCCL 版本不一致
6. **NVLink 故障**: 节点内 NVLink 连接异常
7. **PCIe 错误**: PCIe bus error 导致 GPU 通信失败
8. **防火墙/安全组**: 网络策略阻止了 NCCL 端口

### Metrics to Check
```
# NCCL 指标
- nccl_allreduce_latency_ms
- nccl_bandwidth_gbps
- nccl_timeout_count
- nccl_retry_count

# 网络指标
- ib_port_rcv_errors (InfiniBand)
- ib_port_xmit_discards
- ib_port_data_rate_gbps
- network_packet_loss_rate
- rdma_timeout_count

# GPU 指标
- gpu_ecc_errors (per GPU)
- gpu_xid_errors
- nvlink_error_count
- pcie_replay_count

# 系统指标
- per_rank_alive_status
- per_rank_gpu_utilization
```

### Logs to Check
```bash
# NCCL 错误日志
export NCCL_DEBUG=WARN
grep -i "nccl\|timeout\|watchdog" /var/log/training/rank_*.log

# GPU 硬件错误
nvidia-smi -q -d ECC | grep -i "error\|retired"
dmesg | grep -i "xid\|nvrm\|gpu\|pcie"

# InfiniBand 状态
ibstat | grep -i "state\|rate"
ibdiagnet --ls 2>&1 | grep -i "error\|warn"

# 检查所有 rank 状态
for i in $(seq 0 7); do
  ssh node$i "nvidia-smi --query-gpu=gpu_bus_id,ecc.errors.uncorrected.volatile.total --format=csv"
done

# 检查 NCCL 环境变量
env | grep NCCL
```

### Profiling Method
```bash
# 1. NCCL debug 模式
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
python -m torch.distributed.launch --nproc_per_node=8 train.py 2>&1 | tee nccl_debug.log

# 2. 网络带宽测试
# InfiniBand
ib_write_bw -d mlx5_0 --report_gbits
# 对端
ib_write_bw -d mlx5_0 <server_ip> --report_gbits

# 3. NCCL all-reduce benchmark
nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 8

# 4. NVLink 带宽测试
nvidia-smi nvlink --status
cuda-samples/bin/p2pBandwidthLatencyTest

# 5. GPU 健康检查
nvidia-smi -q -d HEALTH
dcgmi diag -r 3  # DCGM 诊断
```

### Immediate Mitigation
1. **重启失败的 rank**: 如果是单个进程 crash，重启该 rank
2. **排除故障 GPU**: 如果是 GPU 硬件错误，将该 GPU 从集群中移除
3. **增加 timeout**: 临时增加 `NCCL_TIMEOUT`（如 1800s）
4. **切换网络路径**: 如果是特定 IB 端口故障，切换到备用端口
5. **降低 TP/PP 度**: 减少通信需求

### Rollback
- 回滚到单机推理（如果是分布式推理）
- 回滚 NCCL 版本
- 恢复之前的网络配置

### Long-term Fix
1. **健康检查**: 定期运行 NCCL all-reduce benchmark 检测通信性能
2. **故障自动恢复**: 实现 rank failure detection + automatic restart
3. **冗余网络**: 配置多路 InfiniBand，单路故障自动切换
4. **GPU 监控**: 实时监控 ECC error，预防性替换
5. **NCCL 参数调优**: 根据网络拓扑优化 NCCL 参数
6. **Elastic training/inference**: 支持动态增减 rank

### Postmortem
- **Timeline**: 故障发生 → 检测 → 定位 → 恢复
- **Root Cause**: 具体是网络/GPU/软件哪个层面
- **Blast Radius**: 影响了多少任务/用户
- **MTTR**: 平均恢复时间
- **Prevention**: 如何提前检测

### Interview Answer
> "NCCL timeout 的排查我按以下优先级：
> 1. **快速定位故障 rank**: 检查哪个 rank 最后一次通信成功，通常是该 rank 或其下一个 rank 出问题
> 2. **检查 GPU 健康**: `nvidia-smi -q -d ECC` 和 `dmesg | grep xid`，排除硬件故障
> 3. **检查网络**: `ibstat` 看 IB 端口状态，`ib_write_bw` 测试带宽
> 4. **检查进程**: 确认所有 rank 进程存活，检查 OOM
> 
> 一个案例：8 卡 TP 推理中 rank 3 timeout。排查发现 GPU 3 有 uncorrected ECC error，导致该 GPU 计算结果错误，all-reduce 时其他 rank 等待。解决：替换 GPU，建立 ECC error 监控告警（threshold: 1 uncorrected error → 立即告警）。"

---

## 事故 2：Kubernetes GPU Node NotReady（K8s GPU 节点不可用）

### Symptom（现象）
- `kubectl get nodes` 显示 GPU 节点状态为 `NotReady`
- Pod 被 evict 或无法调度到 GPU 节点
- `nvidia-smi` 在节点上执行失败或超时
- GPU device plugin pod crash
- 节点上的推理服务不可用

### Possible Root Causes（可能原因）
1. **GPU Driver crash**: NVIDIA driver 崩溃（XID error 导致）
2. **Device plugin 故障**: nvidia-device-plugin pod crash 或 hang
3. **Kubelet 问题**: kubelet 无法与 API server 通信
4. **GPU 硬件故障**: GPU 物理损坏或过热
5. **内存不足**: 节点 OOM（系统内存，非 GPU 内存）
6. **磁盘满**: 节点磁盘空间不足导致 kubelet 异常
7. **网络分区**: 节点与 K8s control plane 网络断开
8. **GPU Operator 问题**: GPU Operator 组件异常

### Metrics to Check
```
# Kubernetes 指标
- node_status_condition{condition="Ready"}
- kube_node_status_allocatable{resource="nvidia.com/gpu"}
- kube_pod_status_phase{pod=~"nvidia-device-plugin.*"}
- kubelet_running_pods

# 节点指标
- node_memory_MemAvailable_bytes
- node_filesystem_avail_bytes
- node_cpu_seconds_total

# GPU 指标
- nvidia_gpu_duty_cycle
- nvidia_gpu_temperature_celsius
- nvidia_gpu_memory_used_bytes
- dcgm_gpu_health_status
```

### Logs to Check
```bash
# Kubernetes 事件
kubectl describe node <gpu-node> | grep -A 20 "Conditions"
kubectl get events --field-selector involvedObject.name=<gpu-node> --sort-by='.lastTimestamp'

# Kubelet 日志
journalctl -u kubelet --since "1 hour ago" | grep -i "error\|fail\|gpu\|nvidia"

# Device plugin 日志
kubectl logs -n kube-system nvidia-device-plugin-<pod> --previous

# GPU driver 日志
ssh <gpu-node> "dmesg | grep -i 'nvrm\|nvidia\|xid\|gpu'"
ssh <gpu-node> "nvidia-smi 2>&1"

# 系统资源
ssh <gpu-node> "free -h; df -h; uptime"
```

### Profiling Method
```bash
# 1. GPU 健康诊断
ssh <gpu-node> "nvidia-smi -q -d HEALTH"
ssh <gpu-node> "dcgmi diag -r 3"  # Level 3 诊断

# 2. Driver 状态检查
ssh <gpu-node> "cat /proc/driver/nvidia/version"
ssh <gpu-node> "lsmod | grep nvidia"
ssh <gpu-node> "nvidia-smi --query-gpu=gpu_bus_id,driver_version,pstate,temperature.gpu --format=csv"

# 3. Device plugin 状态
kubectl get pods -n kube-system -l app=nvidia-device-plugin -o wide
kubectl describe pod -n kube-system nvidia-device-plugin-<pod>

# 4. 节点资源检查
kubectl top node <gpu-node>
ssh <gpu-node> "ps aux --sort=-%mem | head -20"
```

### Immediate Mitigation
1. **重启 device plugin**: `kubectl delete pod -n kube-system nvidia-device-plugin-<pod>`
2. **重启 kubelet**: `ssh <node> "systemctl restart kubelet"`
3. **重载 GPU driver**: `ssh <node> "nvidia-smi -r"` 或 `rmmod nvidia && modprobe nvidia`
4. **Cordon 节点**: `kubectl cordon <node>` 防止新 pod 调度
5. **Drain 节点**: `kubectl drain <node> --ignore-daemonsets` 迁移 workload
6. **重启节点**: 如果以上都不行，重启整个节点

### Rollback
- 将 workload 迁移到其他健康 GPU 节点
- 回滚 GPU driver 版本
- 回滚 GPU Operator 版本
- 恢复之前的 K8s 配置

### Long-term Fix
1. **GPU 健康检查 DaemonSet**: 定期运行 `dcgmi diag` 检测 GPU 健康
2. **自动 cordon**: GPU 异常时自动 cordon 节点
3. **冗余节点**: 保持 N+1 GPU 节点冗余
4. **Driver 版本管理**: 使用 GPU Operator 统一管理 driver 版本
5. **监控告警**: GPU 温度/ECC error/XID error 告警
6. **定期维护**: 定期重启 GPU 节点清理 driver 状态

### Interview Answer
> "K8s GPU 节点 NotReady 的排查：
> 1. **快速分类**: 是节点级别问题（kubelet/网络）还是 GPU 级别问题（driver/hardware）
> 2. **检查 kubelet**: `journalctl -u kubelet` 看是否有 OOM 或通信错误
> 3. **检查 GPU**: SSH 到节点执行 `nvidia-smi`，如果失败说明 driver crash
> 4. **检查 device plugin**: 看 pod 状态和日志
> 
> 恢复步骤：如果是 driver crash，先 `nvidia-smi -r` 尝试 reset；如果失败，`rmmod nvidia && modprobe nvidia`；如果还是失败，重启节点。同时 drain 节点迁移 workload。
> 
> 预防：我们部署了 GPU health check DaemonSet，每 5 分钟运行 `dcgmi diag -r 1`，发现异常自动 cordon 节点并告警。MTTR 从 30 分钟降到 5 分钟。"

---

## GPU 集群运维最佳实践

### 监控体系

```
Layer 1 — 硬件层:
├── GPU 温度、功耗、频率
├── ECC error count (corrected + uncorrected)
├── NVLink error count
├── PCIe replay count
├── InfiniBand port errors
└── 磁盘 SMART 状态

Layer 2 — Driver/Runtime 层:
├── nvidia-smi 可用性
├── CUDA runtime 状态
├── NCCL 通信延迟
├── Device plugin 状态
└── GPU Operator 组件状态

Layer 3 — Kubernetes 层:
├── Node Ready 状态
├── GPU 资源 allocatable
├── Pod 调度成功率
├── GPU 利用率
└── GPU 内存使用率

Layer 4 — 应用层:
├── 推理延迟 (TTFT, TPOT)
├── Throughput (tokens/s)
├── 错误率
├── SLA 达成率
└── 用户体验指标
```

### 告警规则

| 告警 | 条件 | 严重级别 | 响应时间 |
|------|------|----------|----------|
| GPU ECC Uncorrected | > 0 | Critical | 5 min |
| GPU Temperature | > 85°C | Warning | 15 min |
| GPU Temperature | > 90°C | Critical | 5 min |
| Node NotReady | duration > 2min | Critical | 5 min |
| NCCL Timeout | any | Critical | 5 min |
| Device Plugin Crash | restart > 3 | Warning | 15 min |
| GPU Utilization | < 10% for 30min | Warning | 30 min |
| IB Port Error | rate > 100/min | Warning | 15 min |

### 故障恢复 SOP

```
1. 检测 (Detection)
   └── 自动监控告警 → PagerDuty/飞书

2. 分类 (Triage)
   ├── 硬件故障 → 替换硬件
   ├── Driver 故障 → 重载/重启
   ├── 网络故障 → 切换路径
   └── 软件故障 → 重启/回滚

3. 缓解 (Mitigation)
   ├── Cordon 故障节点
   ├── Drain workload
   └── 扩容健康节点

4. 恢复 (Recovery)
   ├── 修复故障节点
   ├── 验证 GPU 健康
   ├── Uncordon 节点
   └── 验证 workload 正常

5. 复盘 (Postmortem)
   ├── Timeline
   ├── Root cause
   ├── Impact
   └── Prevention
```
