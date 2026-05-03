# How to generate 


```python
    # regenerate fleet catalogue, posets, cost modules, and routing_policy
    uv run python -c "from experiments.mcdp.routing_bird import write_static_catalogues; write_static_catalogues(routing_implementations=None)"

    # run BiRD grid and write routing_service.dpc.yaml
    uv run python experiments/mcdp/routing_bird.py
```