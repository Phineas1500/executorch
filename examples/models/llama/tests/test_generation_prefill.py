# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from executorch.examples.models.llama.runner.generation import LlamaRunner


class _DummyTokenizer:
    n_words = 100
    eos_id = 2
    stop_tokens = [2]

    def encode(self, _text, bos=False, eos=False):
        del bos
        del eos
        return [10, 11]

    def decode_token(self, token_id):
        return str(token_id)


class _DummyRunner(LlamaRunner):
    def __init__(self, raise_on_parallel_prefill=False):
        self.calls = []
        self.raise_on_parallel_prefill = raise_on_parallel_prefill
        super().__init__(
            tokenizer_path="unused",
            tokenizer_config_path=None,
            max_seq_len=16,
            max_batch_size=1,
            use_kv_cache=True,
            vocab_size=100,
            device="cpu",
        )

    def forward(self, tokens: torch.Tensor, input_pos=None) -> torch.Tensor:
        self.calls.append((tokens.clone(), input_pos.clone() if input_pos is not None else None))
        if self.raise_on_parallel_prefill and tokens.shape[1] > 1:
            raise RuntimeError("parallel prefill failure")
        return torch.zeros((1, 8), dtype=torch.float32)


class TestGenerationPrefill(unittest.TestCase):
    @patch(
        "executorch.examples.models.llama.runner.generation.get_tokenizer",
        return_value=_DummyTokenizer(),
    )
    def test_static_prefill_uses_sequential_tokens(self, _mock_get_tokenizer):
        runner = _DummyRunner()
        runner.enable_dynamic_shape = False

        runner._prefill_with_kv_cache([5, 6, 7], pos_base=3)

        self.assertEqual(len(runner.calls), 3)
        for i, (tokens, input_pos) in enumerate(runner.calls):
            self.assertEqual(tuple(tokens.shape), (1, 1))
            self.assertEqual(tokens.item(), 5 + i)
            self.assertEqual(input_pos.item(), 3 + i)

    @patch(
        "executorch.examples.models.llama.runner.generation.get_tokenizer",
        return_value=_DummyTokenizer(),
    )
    def test_dynamic_prefill_uses_batched_prompt(self, _mock_get_tokenizer):
        runner = _DummyRunner()
        runner.enable_dynamic_shape = True

        runner._prefill_with_kv_cache([5, 6, 7], pos_base=4)

        self.assertEqual(len(runner.calls), 1)
        tokens, input_pos = runner.calls[0]
        self.assertEqual(tuple(tokens.shape), (1, 3))
        self.assertEqual(input_pos.item(), 4)

    @patch(
        "executorch.examples.models.llama.runner.generation.get_tokenizer",
        return_value=_DummyTokenizer(),
    )
    def test_dynamic_prefill_does_not_mask_runtime_errors(self, _mock_get_tokenizer):
        runner = _DummyRunner(raise_on_parallel_prefill=True)
        runner.enable_dynamic_shape = True

        with self.assertRaisesRegex(RuntimeError, "parallel prefill failure"):
            runner._prefill_with_kv_cache([5, 6], pos_base=0)

        self.assertEqual(len(runner.calls), 1)


if __name__ == "__main__":
    unittest.main()
