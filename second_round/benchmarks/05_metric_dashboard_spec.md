# Metric Dashboard Spec

## 目的

定义 GPU Serving 监控面板的设计规范，覆盖 GPU 硬件层、请求层、系统层三个维度，可直接导入 Grafana 使用。

---

## 1. Dashboard 总览

| Dashboard | 用途 | 刷新频率 | 时间范围 |
|-----------|------|---------|---------|
| GPU Overview | GPU 硬件状态总览 | 10s | 1h / 6h / 24h |
| Request Latency | 请求延迟分布与趋势 | 10s | 1h / 6h |
| Throughput & QPS | 吞吐量与请求速率 | 10s | 1h / 6h |
| Autoscaling | 扩缩容状态与决策 | 30s | 6h / 24h |
| Alerts | 告警历史与状态 | 60s | 24h / 7d |

---

## 2. GPU Overview Dashboard

### Panel 布局

```
┌─────────────────────────────────────────────────────────┐
│  [Stat] GPU Util %   [Stat] Memory Used   [Stat] Temp  │
├─────────────────────────────────┬───────────────────────┤
│  [TimeSeries] GPU Utilization   │ [Gauge] Memory Usage  │
│  过去 1h 趋势                    │ 当前 / 总量           │
├─────────────────────────────────┼───────────────────────┤
│  [TimeSeries] Memory Usage      │ [Stat] Power (W)      │
│  过去 1h 趋势                    │ 当前功耗              │
├─────────────────────────────────┴───────────────────────┤
│  [Table] Per-GPU Status (多卡时)                         │
└─────────────────────────────────────────────────────────┘
```

### Panel 定义

#### GPU Utilization (TimeSeries)

```yaml
panel:
  title: "GPU Utilization (%)"
  type: timeseries
  datasource: prometheus
  queries:
    - expr: "gpu_utilization_percent"
      legendFormat: "GPU {{gpu_id}}"
  thresholds:
    - value: 80
      color: yellow
    - value: 95
      color: red
  yaxis:
    min: 0
    max: 100
    unit: percent
```

#### Memory Usage (Gauge)

```yaml
panel:
  title: "GPU Memory"
  type: gauge
  queries:
    - expr: "gpu_memory_used_bytes / gpu_memory_total_bytes * 100"
  thresholds:
    - value: 70
      color: green
    - value: 85
      color: yellow
    - value: 95
      color: red
  unit: percent
```

---

## 3. Request Latency Dashboard

### Panel 布局

```
┌─────────────────────────────────────────────────────────┐
│ [Stat] TTFT p50  [Stat] TPOT p50  [Stat] E2E p99      │
├─────────────────────────────────┬───────────────────────┤
│  [TimeSeries] TTFT Distribution │ [Heatmap] Latency    │
│  p50 / p95 / p99               │ 请求延迟热力图        │
├─────────────────────────────────┼───────────────────────┤
│  [TimeSeries] TPOT Distribution │ [Histogram] E2E      │
│  p50 / p95 / p99               │ 延迟分布直方图        │
├─────────────────────────────────┴───────────────────────┤
│  [TimeSeries] SLA Compliance Rate                       │
│  满足 SLA 的请求比例                                     │
└─────────────────────────────────────────────────────────┘
```

### Panel 定义

#### TTFT Percentiles (TimeSeries)

```yaml
panel:
  title: "Time To First Token (TTFT)"
  type: timeseries
  queries:
    - expr: "histogram_quantile(0.50, rate(llm_ttft_seconds_bucket[5m]))"
      legendFormat: "p50"
    - expr: "histogram_quantile(0.95, rate(llm_ttft_seconds_bucket[5m]))"
      legendFormat: "p95"
    - expr: "histogram_quantile(0.99, rate(llm_ttft_seconds_bucket[5m]))"
      legendFormat: "p99"
  unit: seconds
  thresholds:
    - value: 0.5   # 500ms SLA
      color: red
      line: true
```

#### TPOT Percentiles (TimeSeries)

```yaml
panel:
  title: "Time Per Output Token (TPOT)"
  type: timeseries
  queries:
    - expr: "histogram_quantile(0.50, rate(llm_tpot_seconds_bucket[5m]))"
      legendFormat: "p50"
    - expr: "histogram_quantile(0.95, rate(llm_tpot_seconds_bucket[5m]))"
      legendFormat: "p95"
    - expr: "histogram_quantile(0.99, rate(llm_tpot_seconds_bucket[5m]))"
      legendFormat: "p99"
  unit: seconds
```

#### SLA Compliance (TimeSeries)

```yaml
panel:
  title: "SLA Compliance Rate"
  type: timeseries
  queries:
    - expr: |
        sum(rate(llm_request_duration_seconds_bucket{le="5.0"}[5m]))
        /
        sum(rate(llm_request_duration_seconds_count[5m]))
        * 100
      legendFormat: "% requests within 5s SLA"
  unit: percent
  thresholds:
    - value: 95
      color: green
    - value: 90
      color: yellow
      below: true
```

---

## 4. Throughput & QPS Dashboard

### Panel 布局

```
┌─────────────────────────────────────────────────────────┐
│ [Stat] Current TPS  [Stat] Current QPS  [Stat] Queue   │
├─────────────────────────────────┬───────────────────────┤
│  [TimeSeries] Throughput (TPS)  │ [TimeSeries] QPS     │
│  tokens/s 趋势                  │ requests/s 趋势      │
├─────────────────────────────────┼───────────────────────┤
│  [TimeSeries] Queue Depth       │ [TimeSeries] Active  │
│  等待队列长度                    │ 正在处理的请求数      │
├─────────────────────────────────┴───────────────────────┤
│  [TimeSeries] Error Rate                                │
│  错误率趋势                                              │
└─────────────────────────────────────────────────────────┘
```

### Panel 定义

#### Throughput (TimeSeries)

```yaml
panel:
  title: "Token Throughput"
  type: timeseries
  queries:
    - expr: "rate(llm_tokens_generated_total[1m])"
      legendFormat: "tokens/s"
  unit: "tokens/s"
```

#### Queue Depth (TimeSeries)

```yaml
panel:
  title: "Request Queue Depth"
  type: timeseries
  queries:
    - expr: "llm_requests_in_queue"
      legendFormat: "queued"
    - expr: "llm_active_requests"
      legendFormat: "active"
  thresholds:
    - value: 10
      color: yellow
    - value: 30
      color: red
```

---

## 5. Autoscaling Dashboard

### Panel 布局

```
┌─────────────────────────────────────────────────────────┐
│ [Stat] Desired  [Stat] Ready  [Stat] Last Scale Event  │
├─────────────────────────────────┬───────────────────────┤
│  [TimeSeries] Replica Count     │ [TimeSeries] Scaling  │
│  desired vs ready               │ 触发指标趋势          │
├─────────────────────────────────┼───────────────────────┤
│  [TimeSeries] Scale Triggers    │ [Log] Scale Events    │
│  GPU util / queue / latency     │ 扩缩容事件日志        │
└─────────────────────────────────┴───────────────────────┘
```

### Panel 定义

#### Replica Count (TimeSeries)

```yaml
panel:
  title: "Serving Replicas"
  type: timeseries
  queries:
    - expr: "serving_replicas_desired"
      legendFormat: "desired"
    - expr: "serving_replicas_ready"
      legendFormat: "ready"
  overrides:
    - matcher: { id: "byName", options: "desired" }
      properties:
        - id: custom.lineStyle
          value: { fill: "dash" }
```

#### Scaling Triggers (TimeSeries)

```yaml
panel:
  title: "Autoscaling Trigger Metrics"
  type: timeseries
  queries:
    - expr: "gpu_utilization_percent"
      legendFormat: "GPU Util %"
    - expr: "llm_requests_in_queue"
      legendFormat: "Queue Depth"
    - expr: "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) * 1000"
      legendFormat: "p95 Latency (ms)"
  # 多 Y 轴
```

---

## 6. 告警规则完整定义

### Prometheus Alert Rules

```yaml
# monitoring/prometheus/alert_rules.yml
groups:
  - name: gpu_serving_alerts
    rules:
      # P1: GPU Memory 即将耗尽
      - alert: GPUMemoryExhausted
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
        for: 1m
        labels:
          severity: critical
          team: inference
        annotations:
          summary: "GPU memory usage > 95% on {{ $labels.instance }}"
          description: "Current usage: {{ $value | humanizePercentage }}"
          runbook_url: "docs/runbooks/gpu_oom.md"

      # P1: 请求延迟超 SLA
      - alert: HighRequestLatency
        expr: histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m])) > 10
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "p99 request latency > 10s"

      # P2: GPU 利用率持续高
      - alert: HighGPUUtilization
        expr: gpu_utilization_percent > 90
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU utilization > 90% for 10min, consider scaling up"

      # P2: 请求队列堆积
      - alert: QueueBuildUp
        expr: llm_requests_in_queue > 20
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Request queue depth > 20"

      # P2: 错误率升高
      - alert: HighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Error rate > 5%"

      # P3: 扩容失败
      - alert: ScalingFailure
        expr: serving_replicas_desired - serving_replicas_ready > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Desired replicas not met for 5min"
```

---

## 7. Recording Rules（预计算）

```yaml
# monitoring/prometheus/recording_rules.yml
groups:
  - name: llm_serving_recordings
    rules:
      # 预计算 throughput
      - record: llm:throughput:rate5m
        expr: rate(llm_tokens_generated_total[5m])

      # 预计算 QPS
      - record: llm:qps:rate5m
        expr: rate(llm_requests_total[5m])

      # 预计算 SLA compliance
      - record: llm:sla_compliance:rate5m
        expr: |
          sum(rate(llm_request_duration_seconds_bucket{le="5.0"}[5m]))
          / sum(rate(llm_request_duration_seconds_count[5m]))

      # 预计算 error rate
      - record: llm:error_rate:rate5m
        expr: |
          rate(llm_requests_total{status="error"}[5m])
          / rate(llm_requests_total[5m])
```

---

## 8. Docker Compose 部署

```yaml
# docker-compose.yml
version: "3.8"
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml
      - ./monitoring/prometheus/recording_rules.yml:/etc/prometheus/recording_rules.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  gpu_exporter:
    build: ./monitoring/exporters
    ports:
      - "9400:9400"
    command: python gpu_metrics_exporter.py

  serving_exporter:
    build: ./monitoring/exporters
    ports:
      - "9401:9401"
    command: python serving_metrics_exporter.py
```

---

## 9. Prometheus 配置

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 10s
  evaluation_interval: 10s

rule_files:
  - "alert_rules.yml"
  - "recording_rules.yml"

scrape_configs:
  - job_name: "gpu_metrics"
    static_configs:
      - targets: ["gpu_exporter:9400"]

  - job_name: "serving_metrics"
    static_configs:
      - targets: ["serving_exporter:9401"]
```

---

## 10. 面试中的监控设计表达

### 系统设计回答模板

"我的 GPU serving 监控体系分三层：

1. **GPU 硬件层**：utilization, memory, temperature, power — 用于容量规划和硬件告警
2. **请求层**：TTFT/TPOT p50/p95/p99, throughput, queue depth — 用于 SLA 监控和性能分析
3. **系统层**：replica count, scaling events, error rate — 用于运维和自动化

Autoscaling 基于多指标联合判断（GPU util > 85% OR queue > 10 OR p95 > SLA），避免单指标误判。告警分 P1（立即响应：OOM, latency breach）和 P2（24h 内：高 util, queue buildup）。"
