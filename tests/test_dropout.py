import pytest
from sentence_transformers.sentence_transformer.modules import Pooling
from torch import nn

from stochastic_retrieval.encoding import (
    configure_dropout,
    configure_pooling,
    dropout_probability_summary,
)


class TinySelfAttention(nn.Module):
    """BERT-style: functional attention dropout gated by this module's training
    flag, with the probability held by an nn.Dropout child that is never called."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self = TinySelfAttention()


class TinyT5Attention(nn.Module):
    """T5-style: functional attention dropout with a float probability attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = 0.1


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = TinyAttention()
        self.SelfAttention = TinyT5Attention()
        self.hidden_dropout = nn.Dropout(0.2)
        self.output = nn.Linear(2, 2)


def test_attention_scope_flips_gate_modules_not_just_dropout_children() -> None:
    model = TinyModel()

    enabled = configure_dropout(
        model,
        stochastic=True,
        scope="attention",
        probability=0.3,
    )

    assert enabled == 2
    assert model.training is False
    # BERT-style: the gate is the self-attention module owning the dropout child.
    assert model.attention.self.training is True
    assert model.attention.self.dropout.p == 0.3
    # T5-style: the gate is the attention module with a float probability.
    assert model.SelfAttention.training is True
    assert model.SelfAttention.dropout == 0.3
    assert model.hidden_dropout.training is False
    assert model.output.training is False


def test_hidden_scope_leaves_attention_gates_in_eval_mode() -> None:
    model = TinyModel()

    enabled = configure_dropout(model, stochastic=True, scope="hidden")

    assert enabled == 1
    assert model.hidden_dropout.training is True
    assert model.attention.self.training is False
    assert model.SelfAttention.training is False


def test_deterministic_mode_disables_all_dropout() -> None:
    model = TinyModel()
    model.train()

    enabled = configure_dropout(model, stochastic=False)

    assert enabled == 0
    assert model.attention.self.training is False
    assert model.SelfAttention.training is False
    assert model.hidden_dropout.training is False


def test_stochastic_mode_rejects_zero_probability_dropout() -> None:
    model = TinyModel()
    model.attention.self.dropout.p = 0.0
    model.SelfAttention.dropout = 0.0
    model.hidden_dropout.p = 0.0

    with pytest.raises(RuntimeError, match="positive-probability"):
        configure_dropout(model, stochastic=True)


def test_probability_summary_covers_module_and_functional_sites() -> None:
    model = TinyModel()

    assert dropout_probability_summary(model, "all") == {"0.1": 2, "0.2": 1}
    assert dropout_probability_summary(model, "attention") == {"0.1": 2}
    assert dropout_probability_summary(model, "hidden") == {"0.2": 1}


def test_pooling_contract_overrides_model_default() -> None:
    model = nn.Sequential(Pooling(4, pooling_mode="mean"))

    configured = configure_pooling(model, "cls")  # type: ignore[arg-type]
    pooling = model[0]

    assert configured == "cls"
    assert pooling.pooling_mode == "cls"
