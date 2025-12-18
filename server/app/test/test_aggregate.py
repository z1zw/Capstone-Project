def test_gold_pipeline_exists():
    from app.gold.build_gold import build_tensor, build_region_aggregation
    assert callable(build_tensor)
    assert callable(build_region_aggregation)
