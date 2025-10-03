# Metrics System Unit Testing Plan

## Overview
The metrics system consists of three main components:
1. **Core Metrics** (`/home/felipemello/forge/src/forge/observability/metrics.py`) - Core classes, accumulators, MetricCollector singleton, record_metric function
2. **Metric Actors** (`/home/felipemello/forge/src/forge/observability/metric_actors.py`) - LocalFetcherActor, GlobalLoggingActor coordination
3. **Main Usage** (`/home/felipemello/forge/apps/toy_rl/toy_metrics/main.py`) - Example usage with TrainActor and GeneratorActor

## Testing Challenges
- **MetricCollector Singleton**: Need MockBackend or proper setup/teardown to avoid state leakage between tests
- **Actor System**: Requires async testing with Monarch actor framework
- **Multi-rank simulation**: Need to test cross-rank behavior without actual distributed setup

## Complete Test Coverage

### 1. Core Metrics Module Tests

#### Metric Creation & Validation
- `Metric` object creation with automatic timestamp
- `Metric` object with custom timestamp
- `record_metric()` creates correct Metric object
- `record_metric()` with FORGE_DISABLE_METRICS=true (should be no-op)

#### Accumulator Classes
- `MeanAccumulator`: append(), get_value(), get_state(), reset()
- `SumAccumulator`: append(), get_value(), get_state(), reset()
- `MaxAccumulator`: append(), get_value(), get_state(), reset()
- `MinAccumulator`: append(), get_value(), get_state(), reset()
- `StdAccumulator`: append(), get_value(), get_state(), reset()
- Cross-accumulator state reduction via `get_reduced_value_from_states()`

#### Reduce Enum
- Each `Reduce` enum maps to correct accumulator class
- `reduce_metrics_states()` with mixed reduction types (should raise ValueError)
- `reduce_metrics_states()` with empty states
- `reduce_metrics_states()` with single and multiple states

#### MetricCollector Singleton Behavior
- Singleton per-rank behavior (same instance across calls)
- Different ranks get different instances
- `push()` without initialization (should raise ValueError)
- `push()` with invalid metric type (should raise TypeError)
- `flush()` without initialization (returns empty dict)
- `flush()` with no metrics (returns empty dict)

#### Backend Classes
- `ConsoleBackend`: init(), log(), log_immediate(), finish()
- `WandbBackend`: init() for different modes, log(), log_immediate(), get_metadata_for_secondary_ranks()
- Backend factory function `get_logger_backend_class()`

### 2. Metric Actors Module Tests

#### LocalFetcherActor
- `flush()` with return_state=True/False
- `init_backends()` with various configs
- `shutdown()` cleanup

#### GlobalLoggingActor
- `init_backends()` with valid/invalid configs
- `register_fetcher()` and `deregister_fetcher()`
- `flush()` coordination across multiple fetchers
- `shutdown()` cleanup
- `has_fetcher()` and `get_fetcher_count()`

#### Integration Function
- `get_or_create_metric_logger()` creates singleton correctly
- `get_or_create_metric_logger()` handles inconsistent state

### 3. Integration Tests
- End-to-end metric recording and flushing
- Multiple backends with different logging modes
- Cross-rank metric aggregation simulation

## Prioritized Test Implementation

Based on ease of testing and core functionality, here's the prioritized list:

### Priority 1: Core Functionality (Easily Testable)
1. **Test: Metric Creation & Basic Operations** - Tests Metric class, record_metric, accumulator basics
2. **Test: Accumulator State Management** - Tests all accumulator classes with state operations
3. **Test: MetricCollector with Mock Backend** - Tests singleton behavior with controlled backend
4. **Test: Reduce Operations** - Tests reduce_metrics_states and cross-accumulator operations

### Priority 2: Backend Testing (Medium Complexity)
5. **Test: Console Backend** - Tests simplest backend implementation
6. **Test: Backend Factory** - Tests get_logger_backend_class function

### Priority 3: Actor Integration (Most Complex)
7. **Test: Actor Coordination** - Tests LocalFetcherActor and GlobalLoggingActor with mocks

## Detailed Unit Tests

### Test 1: Metric Creation & Basic Operations
```python
import pytest
import time
from unittest.mock import patch, MagicMock
from forge.observability.metrics import Metric, record_metric, Reduce, MetricCollector

class MockBackend:
    def __init__(self):
        self.logged_metrics = []
        self.immediate_metrics = []

    def log_immediate(self, metric, step):
        self.immediate_metrics.append((metric, step))

    async def log(self, metrics, step):
        self.logged_metrics.extend(metrics)

@patch('forge.observability.metrics.current_rank')
def test_metric_creation(mock_rank):
    """Test Metric object creation with automatic and custom timestamps."""
    mock_rank.return_value = MagicMock(rank=0)

    # Test automatic timestamp
    before_time = time.time()
    metric = Metric("test_key", 42.0, Reduce.MEAN)
    after_time = time.time()

    assert metric.key == "test_key"
    assert metric.value == 42.0
    assert metric.reduction == Reduce.MEAN
    assert before_time <= metric.timestamp <= after_time

    # Test custom timestamp
    custom_time = 1234567890.0
    metric_custom = Metric("test_key2", 24.0, Reduce.SUM, timestamp=custom_time)
    assert metric_custom.timestamp == custom_time

@patch('forge.observability.metrics.current_rank')
@patch('forge.observability.metrics.MetricCollector')
def test_record_metric(mock_collector_class, mock_rank):
    """Test record_metric creates correct Metric and calls collector."""
    mock_rank.return_value = MagicMock(rank=0)
    mock_collector = MagicMock()
    mock_collector_class.return_value = mock_collector

    record_metric("loss", 1.5, Reduce.MEAN)

    mock_collector_class.assert_called_once()
    mock_collector.push.assert_called_once()

    # Verify the metric passed to push
    pushed_metric = mock_collector.push.call_args[0][0]
    assert pushed_metric.key == "loss"
    assert pushed_metric.value == 1.5
    assert pushed_metric.reduction == Reduce.MEAN

@patch.dict('os.environ', {'FORGE_DISABLE_METRICS': 'true'})
@patch('forge.observability.metrics.MetricCollector')
def test_record_metric_disabled(mock_collector_class):
    """Test record_metric is no-op when FORGE_DISABLE_METRICS=true."""
    record_metric("loss", 1.5, Reduce.MEAN)
    mock_collector_class.assert_not_called()

@patch.dict('os.environ', {'FORGE_DISABLE_METRICS': 'false'})
@patch('forge.observability.metrics.current_rank')
@patch('forge.observability.metrics.MetricCollector')
def test_record_metric_enabled_explicit(mock_collector_class, mock_rank):
    """Test record_metric works when FORGE_DISABLE_METRICS=false."""
    mock_rank.return_value = MagicMock(rank=0)
    mock_collector = MagicMock()
    mock_collector_class.return_value = mock_collector

    record_metric("loss", 1.5, Reduce.MEAN)
    mock_collector_class.assert_called_once()
    mock_collector.push.assert_called_once()
```

### Test 2: Accumulator State Management
```python
import pytest
from forge.observability.metrics import (
    MeanAccumulator, SumAccumulator, MaxAccumulator,
    MinAccumulator, StdAccumulator, Reduce
)

def test_mean_accumulator():
    """Test MeanAccumulator operations."""
    acc = MeanAccumulator(Reduce.MEAN)

    # Test initial state
    assert acc.get_value() == 0.0
    state = acc.get_state()
    assert state["sum"] == 0.0
    assert state["count"] == 0

    # Test append and get_value
    acc.append(10.0)
    acc.append(20.0)
    assert acc.get_value() == 15.0

    # Test state
    state = acc.get_state()
    assert state["sum"] == 30.0
    assert state["count"] == 2
    assert state["reduction_type"] == "mean"

    # Test reset
    acc.reset()
    assert acc.get_value() == 0.0
    assert acc.get_state()["sum"] == 0.0
    assert acc.get_state()["count"] == 0

def test_sum_accumulator():
    """Test SumAccumulator operations."""
    acc = SumAccumulator(Reduce.SUM)

    acc.append(5.0)
    acc.append(3.0)
    assert acc.get_value() == 8.0

    state = acc.get_state()
    assert state["total"] == 8.0
    assert state["reduction_type"] == "sum"

    acc.reset()
    assert acc.get_value() == 0.0

def test_max_accumulator():
    """Test MaxAccumulator operations."""
    acc = MaxAccumulator(Reduce.MAX)

    acc.append(5.0)
    acc.append(10.0)
    acc.append(3.0)
    assert acc.get_value() == 10.0

    state = acc.get_state()
    assert state["max_val"] == 10.0
    assert state["reduction_type"] == "max"

def test_min_accumulator():
    """Test MinAccumulator operations."""
    acc = MinAccumulator(Reduce.MIN)

    acc.append(5.0)
    acc.append(10.0)
    acc.append(3.0)
    assert acc.get_value() == 3.0

    state = acc.get_state()
    assert state["min_val"] == 3.0
    assert state["reduction_type"] == "min"

def test_std_accumulator():
    """Test StdAccumulator operations."""
    acc = StdAccumulator(Reduce.STD)

    # Test with zero/one values
    assert acc.get_value() == 0.0
    acc.append(5.0)
    assert acc.get_value() == 0.0  # std of single value is 0

    # Test with multiple values
    acc.append(7.0)  # values: 5, 7, mean=6, std=1
    assert abs(acc.get_value() - 1.0) < 0.001

    state = acc.get_state()
    assert state["sum"] == 12.0
    assert state["sum_sq"] == 74.0  # 5^2 + 7^2 = 25 + 49 = 74
    assert state["count"] == 2

def test_accumulator_state_reduction():
    """Test cross-accumulator state reduction."""
    # Test MeanAccumulator state reduction
    states = [
        {"reduction_type": "mean", "sum": 10.0, "count": 2},
        {"reduction_type": "mean", "sum": 20.0, "count": 3}
    ]
    result = MeanAccumulator.get_reduced_value_from_states(states)
    assert result == 30.0 / 5.0  # (10+20) / (2+3) = 6.0

    # Test SumAccumulator state reduction
    states = [
        {"reduction_type": "sum", "total": 10.0},
        {"reduction_type": "sum", "total": 15.0}
    ]
    result = SumAccumulator.get_reduced_value_from_states(states)
    assert result == 25.0

def test_reduce_enum_accumulator_mapping():
    """Test that Reduce enum correctly maps to accumulator classes."""
    assert Reduce.MEAN.accumulator_class == MeanAccumulator
    assert Reduce.SUM.accumulator_class == SumAccumulator
    assert Reduce.MAX.accumulator_class == MaxAccumulator
    assert Reduce.MIN.accumulator_class == MinAccumulator
    assert Reduce.STD.accumulator_class == StdAccumulator
```

### Test 3: MetricCollector with Mock Backend
```python
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from forge.observability.metrics import MetricCollector, Metric, Reduce

class MockBackend:
    def __init__(self):
        self.logged_metrics = []
        self.immediate_metrics = []

    def log_immediate(self, metric, step):
        self.immediate_metrics.append((metric, step))

    async def log(self, metrics, step):
        self.logged_metrics.extend([(m, step) for m in metrics])

@patch('forge.observability.metrics.current_rank')
def test_metric_collector_singleton(mock_rank):
    """Test MetricCollector singleton behavior per rank."""
    mock_rank.return_value = MagicMock(rank=0)

    collector1 = MetricCollector()
    collector2 = MetricCollector()
    assert collector1 is collector2

    # Different rank should get different instance
    mock_rank.return_value = MagicMock(rank=1)
    collector3 = MetricCollector()
    assert collector1 is not collector3

@patch('forge.observability.metrics.current_rank')
def test_metric_collector_uninitialized_push(mock_rank):
    """Test MetricCollector.push() raises error when uninitialized."""
    mock_rank.return_value = MagicMock(rank=0)

    # Clear any existing singleton
    MetricCollector._instances.clear()
    collector = MetricCollector()

    metric = Metric("test", 1.0, Reduce.MEAN)

    with pytest.raises(ValueError, match="Collector not initialized"):
        collector.push(metric)

@patch('forge.observability.metrics.current_rank')
def test_metric_collector_invalid_metric_type(mock_rank):
    """Test MetricCollector.push() raises error for invalid metric type."""
    mock_rank.return_value = MagicMock(rank=0)

    MetricCollector._instances.clear()
    collector = MetricCollector()

    # Initialize with mock backend
    collector._is_initialized = True
    collector.per_rank_no_reduce_backends = []
    collector.per_rank_reduce_backends = []

    with pytest.raises(TypeError, match="Expected Metric object"):
        collector.push("invalid_metric")

@patch('forge.observability.metrics.current_rank')
@patch('forge.observability.metrics.get_actor_name_with_rank')
async def test_metric_collector_push_and_flush(mock_actor_name, mock_rank):
    """Test MetricCollector push and flush with mock backends."""
    mock_rank.return_value = MagicMock(rank=0)
    mock_actor_name.return_value = "TestActor_abcd_r0"

    MetricCollector._instances.clear()
    collector = MetricCollector()

    # Setup mock backends
    no_reduce_backend = MockBackend()
    reduce_backend = MockBackend()

    collector._is_initialized = True
    collector.per_rank_no_reduce_backends = [no_reduce_backend]
    collector.per_rank_reduce_backends = [reduce_backend]
    collector.step = 0

    # Test push
    metric = Metric("loss", 1.5, Reduce.MEAN)
    collector.push(metric)

    # Should log immediately to no_reduce backend
    assert len(no_reduce_backend.immediate_metrics) == 1
    assert no_reduce_backend.immediate_metrics[0][0].key == "loss"
    assert no_reduce_backend.immediate_metrics[0][1] == 0  # step

    # Should not log to reduce backend yet
    assert len(reduce_backend.logged_metrics) == 0

    # Test flush
    result = await collector.flush(step=1, return_state=True)

    # Should have returned state
    assert "loss" in result
    assert result["loss"]["reduction_type"] == "mean"
    assert result["loss"]["sum"] == 1.5
    assert result["loss"]["count"] == 1

    # Should have logged to reduce backend
    assert len(reduce_backend.logged_metrics) == 1
    logged_metric, step = reduce_backend.logged_metrics[0]
    assert logged_metric.key == "loss"
    assert logged_metric.value == 1.5
    assert step == 1

@patch('forge.observability.metrics.current_rank')
async def test_metric_collector_flush_uninitialized(mock_rank):
    """Test MetricCollector.flush() returns empty dict when uninitialized."""
    mock_rank.return_value = MagicMock(rank=0)

    MetricCollector._instances.clear()
    collector = MetricCollector()

    result = await collector.flush(step=1, return_state=True)
    assert result == {}

@patch('forge.observability.metrics.current_rank')
async def test_metric_collector_flush_no_metrics(mock_rank):
    """Test MetricCollector.flush() returns empty dict when no metrics."""
    mock_rank.return_value = MagicMock(rank=0)

    MetricCollector._instances.clear()
    collector = MetricCollector()
    collector._is_initialized = True
    collector.per_rank_no_reduce_backends = []
    collector.per_rank_reduce_backends = []

    result = await collector.flush(step=1, return_state=True)
    assert result == {}
```

### Test 4: Reduce Operations
```python
import pytest
from forge.observability.metrics import reduce_metrics_states, Reduce

def test_reduce_metrics_states_empty():
    """Test reduce_metrics_states with empty input."""
    result = reduce_metrics_states([])
    assert result == {}

def test_reduce_metrics_states_single_state():
    """Test reduce_metrics_states with single state."""
    states = [
        {"loss": {"reduction_type": "mean", "sum": 10.0, "count": 2}}
    ]
    result = reduce_metrics_states(states)
    assert result == {"loss": 5.0}

def test_reduce_metrics_states_multiple_states():
    """Test reduce_metrics_states with multiple states."""
    states = [
        {"loss": {"reduction_type": "mean", "sum": 10.0, "count": 2}},
        {"loss": {"reduction_type": "mean", "sum": 20.0, "count": 3}},
        {"accuracy": {"reduction_type": "sum", "total": 15.0}}
    ]
    result = reduce_metrics_states(states)
    assert result["loss"] == 30.0 / 5.0  # 6.0
    assert result["accuracy"] == 15.0

def test_reduce_metrics_states_mismatched_types():
    """Test reduce_metrics_states raises error for mismatched reduction types."""
    states = [
        {"loss": {"reduction_type": "mean", "sum": 10.0, "count": 2}},
        {"loss": {"reduction_type": "sum", "total": 20.0}}
    ]
    with pytest.raises(ValueError, match="Mismatched reduction types"):
        reduce_metrics_states(states)

def test_reduce_metrics_states_partial_keys():
    """Test reduce_metrics_states with partial key overlap."""
    states = [
        {"loss": {"reduction_type": "mean", "sum": 10.0, "count": 2},
         "accuracy": {"reduction_type": "sum", "total": 5.0}},
        {"loss": {"reduction_type": "mean", "sum": 20.0, "count": 3}},
        {"throughput": {"reduction_type": "max", "max_val": 100.0}}
    ]
    result = reduce_metrics_states(states)

    assert result["loss"] == 30.0 / 5.0  # 6.0
    assert result["accuracy"] == 5.0
    assert result["throughput"] == 100.0
```

## Test Coverage Summary

The above tests cover:

**✅ Test 1**: `record_metric()` → `Metric` creation → `MetricCollector.push()`
**✅ Test 2**: All accumulator classes with state operations and cross-reduction
**✅ Test 3**: `MetricCollector` singleton behavior with mock backends
**✅ Test 4**: `reduce_metrics_states()` function with various scenarios

These 4 core tests validate the main functionality that `record_metric()` returns a `Metric`, accumulators work correctly, the singleton behaves properly, and cross-rank reduction works. This covers the most critical paths with minimal test code by focusing on core components rather than the full actor integration complexity.
