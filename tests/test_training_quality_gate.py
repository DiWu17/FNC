import pytest
from clickagent.training_quality_gate import unusable_instruction_reason, wording_tag


@pytest.mark.parametrize('instruction',['OK','No','0','+','←','Click Search','Move your cursor to Save','Point to the settings icon','\ufffc Search'])
def test_meaningful_target_descriptions_are_not_removed_for_surface_form(instruction):
    assert unusable_instruction_reason(instruction) is None


@pytest.mark.parametrize('instruction',['\ufffc',' \ufffc\ufffd\u200b ','\ufffd','\u200b\n ',None])
def test_unresolvable_instruction_content_is_rejected(instruction):
    assert unusable_instruction_reason(instruction) is not None


def test_wording_tags_do_not_claim_clickability():
    assert wording_tag('Move your cursor to Save')=='pointing_or_hover_wording'
    assert wording_tag('OK')=='short_label'
    assert wording_tag('Find the profile image')=='other_grounding_description'
