"""GPU memory a model needs, shared by the inference workers, the
orchestrator's feasibility filter and the allocation controller."""

# Workers load weights in bf16: 2 bytes per parameter, i.e. 2 GB per billion.
GB_PER_BILLION_PARAMS = 2.0


def estimated_size_gb(model_config):
    """Size of a model's bf16 weights, from its parameter_count (billions)."""
    params_billion = float((model_config or {}).get("parameter_count", 0) or 0)
    return GB_PER_BILLION_PARAMS * params_billion


def size_on_node_gb(telemetry, model_id, model_config):
    """Memory the model takes on a node: as measured there if the node has
    loaded it before, otherwise the bf16 estimate."""
    measured = (telemetry.get("model_memory_gb") or {}).get(model_id)
    if measured is not None:
        return float(measured)
    return estimated_size_gb(model_config)


def fits_on_node(telemetry, model_id, model_config):
    """True if the model fits into the node's memory budget once the node has
    unloaded other models. Nodes that report no budget are not restricted."""
    budget = telemetry.get("memory_budget_gb")
    if budget is None:
        return True
    return size_on_node_gb(telemetry, model_id, model_config) <= float(budget)
