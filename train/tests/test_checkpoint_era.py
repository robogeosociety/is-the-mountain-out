import json

from train.checkpoint_era import (
    ERA_FILENAME,
    era_matches,
    read_checkpoint_era,
    write_checkpoint_era,
)

QUEEN_ANNE = "king5-queenanne"
UW = "uw-atg"


def test_reads_the_era_written_beside_the_weights(tmp_path):
    write_checkpoint_era(tmp_path, QUEEN_ANNE)
    assert read_checkpoint_era(tmp_path) == QUEEN_ANNE


def test_write_records_provenance_extras(tmp_path):
    path = write_checkpoint_era(tmp_path, QUEEN_ANNE, webcam_url="http://cam/x.jpg")
    saved = json.loads(path.read_text())
    assert path.name == ERA_FILENAME
    assert saved["era"] == QUEEN_ANNE
    assert saved["webcam_url"] == "http://cam/x.jpg"


def test_untagged_checkpoint_falls_back_to_the_configured_era(tmp_path):
    # Everything trained before the marker existed belongs to the dead camera.
    assert read_checkpoint_era(tmp_path, fallback=UW) == UW
    assert read_checkpoint_era(tmp_path) is None


def test_corrupt_marker_is_treated_as_absent(tmp_path):
    (tmp_path / ERA_FILENAME).write_text("{not json")
    assert read_checkpoint_era(tmp_path, fallback=UW) == UW


def test_blank_era_in_marker_is_treated_as_absent(tmp_path):
    (tmp_path / ERA_FILENAME).write_text(json.dumps({"era": "   "}))
    assert read_checkpoint_era(tmp_path, fallback=UW) == UW


def test_matching_requires_both_sides_to_be_known():
    assert era_matches(QUEEN_ANNE, QUEEN_ANNE) is True
    assert era_matches(QUEEN_ANNE, UW) is False
    # "We do not know" is not "yes": an untagged checkpoint never validates.
    assert era_matches(QUEEN_ANNE, None) is False
    assert era_matches(None, QUEEN_ANNE) is False
    assert era_matches(None, None) is False
    assert era_matches("", QUEEN_ANNE) is False


def test_whitespace_does_not_break_a_match():
    assert era_matches(" king5-queenanne ", "king5-queenanne\n") is True


def test_the_live_config_currently_refuses_its_own_checkpoint(tmp_path):
    """The whole point, asserted end to end on the real mountain.toml.

    The shipped config names the Queen Anne camera while the live (untagged)
    checkpoint is UW-era, so inference must refuse it. When a Queen Anne
    checkpoint is trained, `era.json` says so and this flips — that is the
    intended way for this test's premise to change.
    """
    from train.config_loader import ConfigLoader

    config = ConfigLoader("mountain.toml")
    checkpoint_era = read_checkpoint_era(
        tmp_path, fallback=config.checkpoint_era_fallback
    )
    assert config.camera_era == QUEEN_ANNE
    assert checkpoint_era == UW
    assert era_matches(config.camera_era, checkpoint_era) is False

    write_checkpoint_era(tmp_path, config.camera_era)
    assert era_matches(config.camera_era, read_checkpoint_era(tmp_path)) is True


def test_unserializable_extras_do_not_break_a_save(tmp_path):
    """Provenance must never cost a training run.

    The stamp is written right after the weights; raising here would abort a
    run whose real work is already done.
    """

    class Opaque:
        def __repr__(self):
            return "<opaque>"

    path = write_checkpoint_era(tmp_path, QUEEN_ANNE, note=Opaque())
    saved = json.loads(path.read_text())
    assert saved["era"] == QUEEN_ANNE
    assert saved["note"] == "<opaque>"
