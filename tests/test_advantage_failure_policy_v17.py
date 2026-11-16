import pytest

from clickagent.advantage_runtimes import mvp_failed_output


MISSING_QUERY = ("Traceback: in get_top_attention_regions\n"
                 "batch_size, num_heads, q_len, head_dim = query_states.shape\n"
                 "AttributeError: 'NoneType' object has no attribute 'shape'")


def test_absent_generated_token_attention_is_an_output_failure():
    result = mvp_failed_output({'query_states': None}, MISSING_QUERY)
    assert result['point'] is None
    assert result['ground_truth_in_model_input'] is False


@pytest.mark.parametrize('diagnostic', [
    'RuntimeError: CUDA out of memory',
    'RuntimeError: shape mismatch',
    MISSING_QUERY + '\nRuntimeError: CUDA out of memory',
    'Warning: Image file missing, skipping',
])
def test_reference_exceptions_are_not_accuracy_misses(diagnostic):
    with pytest.raises(RuntimeError):
        mvp_failed_output({'query_states': None}, diagnostic)


def test_unexpected_failure_after_captured_query_stops_the_arm():
    with pytest.raises(RuntimeError):
        mvp_failed_output({'query_states': object()}, MISSING_QUERY)
