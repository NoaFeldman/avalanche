from pseudoavalanche.config import AvalancheConfig


def test_param_hash_consistency():
    config1 = AvalancheConfig(seed=1)
    config2 = AvalancheConfig(seed=2)
    assert config1.param_hash == config2.param_hash
    config3 = AvalancheConfig(lambda_dope=1)
    assert config3.param_hash != config1.param_hash
