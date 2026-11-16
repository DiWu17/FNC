import pytest

from clickagent.model_baselines import parse_ui_tars_point


@pytest.mark.parametrize('text',["(336,192)","<point>336 192</point>","Action: click(point='<point>336 192</point>')",
    "Action: click(start_box='<|box_start|>(336,192)<|box_end|>')"])
def test_ui_tars_uses_actual_processed_pixels_not_normalized_1000(text):
    assert parse_ui_tars_point(text,(672,384),(3840,2160))==[1920,1080]


def test_official_box_wrapper_uses_its_center():
    assert parse_ui_tars_point("click(start_box='(300,160,372,224)')",(672,384),(3840,2160))==[1920,1080]


@pytest.mark.parametrize('text',["I think 336,192", "click(point=__import__('os').getcwd())", "(900,192)","(336,-1)","(336,192) unfinished"])
def test_no_prose_number_fallback_execution_or_out_of_bounds_repair(text):
    assert parse_ui_tars_point(text,(672,384),(3840,2160)) is None
