from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models as tk_models, pre_tokenizers, processors
from transformers import (
    M2M100Config,
    M2M100Model,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaModel,
)


EXPECTED_STATE_NAMES = ("INPUT", "ATTN", "POST_ATTN", "FFN", "OUTPUT")


def _build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "hello": 5,
        "tiny": 6,
        "multilingual": 7,
        "transformer": 8,
        "state": 9,
        "test": 10,
        "bonjour": 11,
        "monde": 12,
    }
    backend = Tokenizer(tk_models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )
    tokenizer.model_max_length = 32
    return tokenizer


def _build_roberta_model() -> RobertaModel:
    config = RobertaConfig(
        vocab_size=32,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    model = RobertaModel(config)
    model.eval()
    return model


def _build_m2m100_model() -> M2M100Model:
    config = M2M100Config(
        vocab_size=32,
        d_model=24,
        encoder_layers=2,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=48,
        decoder_ffn_dim=48,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        decoder_start_token_id=2,
        forced_bos_token_id=3,
    )
    model = M2M100Model(config)
    model.eval()
    return model


def _load_internal_state_module():
    if importlib.util.find_spec("src.features.extract_internal_states") is None:
        pytest.skip("src.features.extract_internal_states has not landed yet")
    return importlib.import_module("src.features.extract_internal_states")


def _build_adapter(module, *, model_name: str, model, tokenizer):
    builder = getattr(module, "build_model_state_adapter", None)
    if builder is None:
        pytest.skip("build_model_state_adapter is not implemented yet")
    return builder(model_name=model_name, model=model, tokenizer=tokenizer)


def _run_reference_forward(model_name: str, model, tokenizer, text: str):
    encoded = tokenizer(text, return_tensors="pt", return_special_tokens_mask=True)
    with torch.no_grad():
        if model_name == "nllb":
            outputs = model.get_encoder()(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
    return encoded, outputs


@pytest.mark.parametrize(
    ("model_name", "model_factory"),
    [
        ("xlmr", _build_roberta_model),
        ("nllb", _build_m2m100_model),
    ],
)
def test_internal_state_adapter_exposes_expected_state_keys_and_output_equivalence(
    model_name: str,
    model_factory,
) -> None:
    module = _load_internal_state_module()
    tokenizer = _build_tokenizer()
    if model_name == "nllb":
        tokenizer.src_lang = "eng_Latn"

    model = model_factory()
    adapter = _build_adapter(module, model_name=model_name, model=model, tokenizer=tokenizer)

    text = "hello tiny multilingual transformer test"
    bundle = adapter.encode_text_with_states(text)
    encoded, outputs = _run_reference_forward(model_name, model, tokenizer, text)

    expected_layer_count = len(outputs.hidden_states) - 1
    assert adapter.n_blocks == expected_layer_count
    assert tuple(adapter.state_names) == EXPECTED_STATE_NAMES

    block_indices = sorted(bundle.block_states)
    assert len(block_indices) == expected_layer_count

    reference_last_hidden = outputs.hidden_states[-1][0].detach().cpu().numpy()
    for block_index, hidden_state in zip(block_indices, outputs.hidden_states[1:], strict=True):
        block_states = bundle.block_states[block_index]
        assert set(block_states) == set(EXPECTED_STATE_NAMES)

        for state_name in EXPECTED_STATE_NAMES:
            tensor = block_states[state_name]
            assert tensor.shape == hidden_state[0].shape
            assert np.isfinite(tensor).all()

        np.testing.assert_allclose(
            block_states["OUTPUT"],
            hidden_state[0].detach().cpu().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )

    assert len(bundle.token_strings) == encoded["input_ids"].shape[1]
    assert len(bundle.token_offsets) == encoded["input_ids"].shape[1]
    assert all(len(offset) == 2 for offset in bundle.token_offsets)

    equivalence = adapter.verify_output_equivalence()
    assert isinstance(equivalence, dict)
    assert equivalence
    np.testing.assert_allclose(
        bundle.block_states[block_indices[-1]]["OUTPUT"],
        reference_last_hidden,
        atol=1e-5,
        rtol=1e-5,
    )
