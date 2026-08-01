import pytest
from sentence_transformers.sentence_transformer.modules import Pooling
from torch import nn

from stochastic_retrieval.encoding import configure_dropout, configure_pooling


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_dropout = nn.Dropout(0.1)
        self.hidden_dropout = nn.Dropout(0.2)
        self.output = nn.Linear(2, 2)


def test_attention_scope_only_enables_attention_dropout() -> None:
    model = TinyModel()

    enabled = configure_dropout(
        model,
        stochastic=True,
        scope="attention",
        probability=0.3,
    )

    assert enabled == 1
    assert model.training is False
    assert model.attention_dropout.training is True
    assert model.attention_dropout.p == 0.3
    assert model.hidden_dropout.training is False
    assert model.output.training is False


def test_deterministic_mode_disables_all_dropout() -> None:
    model = TinyModel()
    model.train()

    enabled = configure_dropout(model, stochastic=False)

    assert enabled == 0
    assert model.attention_dropout.training is False
    assert model.hidden_dropout.training is False


def test_stochastic_mode_rejects_zero_probability_dropout() -> None:
    model = TinyModel()
    model.attention_dropout.p = 0.0
    model.hidden_dropout.p = 0.0

    with pytest.raises(RuntimeError, match="positive-probability"):
        configure_dropout(model, stochastic=True)


def test_pooling_contract_overrides_model_default() -> None:
    model = nn.Sequential(Pooling(4, pooling_mode="mean"))

    configured = configure_pooling(model, "cls")  # type: ignore[arg-type]
    pooling = model[0]

    assert configured == "cls"
    assert pooling.pooling_mode == "cls"
