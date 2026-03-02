# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cuda.test.tester import CudaTester
from executorch.backends.test.suite.flow import TestFlow


def _create_cuda_flow(name: str = "cuda") -> TestFlow:
    """Create a test flow for the CUDA backend.

    The CUDA backend saves data externally (.so and weights blob in .ptd file).
    The test harness serialize stage has been updated to support loading external
    data via the data_map_buffer parameter of _load_for_executorch_from_buffer.
    """
    # Tests to skip due to known failures in CUDA backend
    # TODO: Fix these tests and remove from skip list
    skip_patterns = [
        # test_mean_output_dtype: float64 output dtype causes crash in AOTI compiled code
        "test_mean_output_dtype",
        # Adaptive pooling ops - not fully supported
        "test_adaptive_avgpool1d",
        "test_adaptive_avgpool2d",
        "test_adaptive_avgpool3d",
        "test_adaptive_maxpool3d",
        # 3D average pooling - not fully supported
        "test_avgpool3d",
        # Convolution with dilation - not fully supported
        "test_conv1d_dilation",
        "test_conv2d_dilation",
        "test_conv3d_dilation",
        # Transposed convolutions - not fully supported
        "test_convtranspose1d",
        "test_convtranspose2d",
        "test_convtranspose3d",
        # Embedding bag ops - not fully supported
        "test_embedding_bag",
        # Index put with multiple indices - not fully supported
        "test_index_put_in_place_two_indices",
        "test_index_put_two_indices",
        # Median ops - not fully supported
        "test_median",
        # Select/slice/split ops - not fully supported
        "test_select",
        "test_slice",
        "test_split",
        # Threshold op - not fully supported
        "test_threshold",
    ]

    return TestFlow(
        name,
        backend="cuda",
        tester_factory=CudaTester,
        quantize=False,
        skip_patterns=skip_patterns,
    )


CUDA_TEST_FLOW = _create_cuda_flow("cuda")
