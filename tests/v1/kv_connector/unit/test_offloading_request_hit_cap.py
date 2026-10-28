# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The OffloadingConnector loads no further than a request's hit cap."""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "cap,computed,expected",
    [(None, 0, None), (832, 0, 832), (832, 832, 0), (832, 900, 0)],
)
def test_max_loadable_tokens_honors_the_request_cap(cap, computed, expected):
    connector = object.__new__(OffloadingConnector)
    connector._bounding_group_ids = []
    request = SimpleNamespace(max_cache_hit_length=cap)
    assert connector._max_loadable_tokens(request, computed) == expected
