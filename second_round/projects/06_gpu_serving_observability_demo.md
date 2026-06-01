# GPU Serving Observability Demo

> GPU 推理服务可观测性演示，覆盖 Prometheus + Grafana 监控、GPU 利用率追踪、请求延迟监控、自动扩缩容模拟和告警规则。

[![Prometheus](https://img.shields.io/badge/Prometheus-latest-orange.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-latest-yellow.svg)](https://grafana.com/)

## Motivation

生产级 LLM serving 需要完整的可观测性体系：实时监控 GPU 利用率、请求延迟分布、吞吐量变化，并基于这些指标做自动扩缩容决策。本项目构建一个轻量级的 observability demo，展示 GPU serving 监控的最佳实践。

## Key Results

> ⚠️ 以下为目标值，需完成实现后填入实际数据

- 目标：完整的 Grafana dashboard 覆盖 GPU/请求/系统三层指标
- 目标：基于 GPU utilization + queue depth 的 autoscaling 模拟
- 目标：P1/P2 告警规则覆盖关键故障场景
- 目标：可在本地 docker-compose 一键启动

## Directory Structure

```
gpu-serving-observability/
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml         # Prometheus 配置
│   │   ├── alert_rules.yml        # 告警规则
│   │   └── recording_rules.yml    # 预计算规则
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── gpu_overview.json      # GPU 总览 dashboard
│   │   │   ├── request_latency.json   # 请求延迟 dashboard
│   │   │   └── autoscaling.json       # 扩缩容 dashboard
│   │   └── provisioning/
│   │       ├── datasources.yml
│   │       └── dashboards.yml
│   └── exporters/
│       ├── gpu_metrics_exporter.py    # GPU 指标导出（模拟）
│       └── serving_metrics_exporter.py # Serving 指标导出
├── simulation/
│   ├── traffic_generator.py       # 流量模拟器
│   ├── autoscaler_sim.py          # Autoscaling 模拟
│   ├── failure_injection.py       # 故障注入
│   └── scenarios/
│       ├── normal_traffic.yaml    # 正常流量场景
│       ├── burst_traffic.yaml     # 突发流量场景
│       └── gpu_oom.yaml           # GPU OOM 场景
├── alerting/
│   ├── alert_definitions.md       # 告警定义文档
│   └── runbooks/
│       ├── high_latency.md        # 高延迟处理手册
│       ├── gpu_oom.md             # GPU OOM 处理手册
│       └── scaling_failure.md     # 扩容失败处理手册
├── docs/
│   ├── architecture.md            # 监控架构文档
│   ├── metrics_catalog.md         # 指标目录
│   └── scaling_policy.md          # 扩缩容策略文档
├── docker-compose.yml             # 一键启动
└── README.md
```

---

## 监控指标设计

### GPU 层指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `gpu_utilization_percent` | Gauge | SM 活跃率 |
| `gpu_memory_used_bytes` | Gauge | 显存使用量 |
| `gpu_memory_total_bytes` | Gauge | 显存总量 |
| `gpu_temperature_celsius` | Gauge | GPU 温度 |
| `gpu_power_watts` | Gauge | GPU 功耗 |

### 请求层指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `llm_request_duration_seconds` | Histogram | 请求延迟分布 |
| `llm_ttft_seconds` | Histogram | TTFT 分布 |
| `llm_tpot_seconds` | Histogram | TPOT 分布 |
| `llm_tokens_generated_total` | Counter | 生成 token 总数 |
| `llm_requests_total` | Counter | 请求总数 |
| `llm_requests_in_queue` | Gauge | 队列中请求数 |
| `llm_active_requests` | Gauge | 正在处理的请求数 |

### 系统层指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `serving_replicas_desired` | Gauge | 期望副本数 |
| `serving_replicas_ready` | Gauge | 就绪副本数 |
| `serving_scaling_events_total` | Counter | 扩缩容事件数 |
| `serving_cold_start_seconds` | Histogram | 冷启动时间 |

---

## Autoscaling 策略模拟

### 策略设计

```python
class AutoscalingPolicy:
    """基于多指标的扩缩容策略"""
    
    def decide(self, metrics):
        # Scale up 条件（满足任一）
        if metrics.gpu_util > 85% for 2min:
            return ScaleUp(reason="high_gpu_util")
        if metrics.queue_depth > 10 for 1min:
            return ScaleUp(reason="queue_buildup")
        if metrics.p95_latency > SLA_THRESHOLD for 3min:
            return ScaleUp(reason="latency_breach")
        
        # Scale down 条件（全部满足）
        if (metrics.gpu_util < 30% for 10min and
            metrics.queue_depth == 0 and
            metrics.p95_latency < SLA_THRESHOLD * 0.5):
            return ScaleDown(reason="underutilized")
        
        return NoAction()
```

### 模拟场景

| 场景 | 流量模式 | 预期行为 |
|------|---------|---------|
| 正常流量 | 稳定 10 QPS | 维持 1 replica |
| 突发流量 | 10→100 QPS (30s ramp) | 2min 内扩到 3 replicas |
| 渐进增长 | 10→50 QPS (10min ramp) | 逐步扩容 |
| 流量回落 | 50→5 QPS | 10min cooldown 后缩容 |
| GPU OOM | 单请求 seq_len=32K | 触发 OOM 告警，拒绝请求 |

---

## 告警规则

### P1 告警（立即响应）

```yaml
- alert: GPUMemoryExhausted
  expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "GPU memory usage > 95%"
    runbook: "docs/runbooks/gpu_oom.md"

- alert: HighRequestLatency
  expr: histogram_quantile(0.99, llm_request_duration_seconds) > 10
  for: 3m
  labels:
    severity: critical
  annotations:
    summary: "p99 latency > 10s"
```

### P2 告警（24h 内响应）

```yaml
- alert: HighGPUUtilization
  expr: gpu_utilization_percent > 90
  for: 10m
  labels:
    severity: warning

- alert: QueueBuildUp
  expr: llm_requests_in_queue > 20
  for: 5m
  labels:
    severity: warning
```

---

## Setup

```bash
# 一键启动监控栈
docker-compose up -d

# 访问 Grafana
open http://localhost:3000  # admin/admin

# 启动流量模拟
python simulation/traffic_generator.py --scenario scenarios/burst_traffic.yaml

# 启动 autoscaling 模拟
python simulation/autoscaler_sim.py --policy gpu_util_based
```

## Correctness Test Design

```python
def test_autoscaler_scales_up_on_high_util():
    """GPU 利用率持续高于阈值时应触发扩容"""
    sim = AutoscalerSimulation(policy=GPUUtilPolicy(threshold=85))
    sim.inject_metrics(gpu_util=90, duration_min=3)
    actions = sim.get_actions()
    assert any(a.type == "scale_up" for a in actions)

def test_alert_fires_on_oom():
    """GPU memory > 95% 应触发 P1 告警"""
    alerts = evaluate_rules(gpu_memory_used=15.2e9, gpu_memory_total=16e9)
    assert "GPUMemoryExhausted" in [a.name for a in alerts]
```

## Expected Metrics

| 场景 | 扩容响应时间 | 最终副本数 | SLA 违反率 |
|------|------------|-----------|-----------|
| 突发流量 | 目标 < 2min | 3 | 目标 < 5% |
| 渐进增长 | 目标 < 5min | 2 | 目标 0% |
| GPU OOM | N/A（告警） | 1 | 告警触发 < 1min |

## Resume Bullet

*（完成后使用）*
- "Designed GPU serving observability system with Prometheus/Grafana monitoring, covering GPU utilization, request latency (TTFT/TPOT p50/p95/p99), and queue depth metrics"
- "Implemented GPU-aware autoscaling simulation achieving < 2min scale-up response time under burst traffic with < 5% SLA violation rate"

## Interview Talking Points

### 系统设计

- "我的监控体系分三层：GPU 硬件层（utilization, memory, power）、请求层（TTFT, TPOT, queue depth）、系统层（replicas, scaling events）"
- "Autoscaling 策略基于多指标联合判断：GPU util > 85% OR queue > 10 OR p95 > SLA，避免单指标误判"
- "告警分 P1/P2，P1 立即响应（OOM, latency breach），P2 24h 内处理（高 util, queue buildup）"

### 深度追问

1. "GPU utilization 高但 latency 正常，要扩容吗？" → 不一定，看 headroom 和趋势
2. "冷启动时间长怎么办？" → 预热 replica pool + 模型预加载
3. "多租户场景怎么隔离？" → per-tenant queue + priority scheduling + resource quota
4. "怎么做 graceful drain？" → 停止接收新请求 + 等待 in-flight 完成 + health check 摘除
