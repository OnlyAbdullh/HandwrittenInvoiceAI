"""The GGUF header check: right files accepted, every common wrong file named.

A llama.cpp vision model is two files, and each failure here is one somebody
will actually make -- the two swapped, the same file twice, a text-only model's
projector missing, a download saved as an HTML error page or cut short.
"""

from __future__ import annotations

import pytest

from core.exceptions import ModelFileError
from ocr.llamacpp.gguf import check_vision_model_pair, read_gguf_info
from tests.llamacpp_support import write_gguf, write_model, write_projector


@pytest.fixture
def model(tmp_path):
    return write_model(tmp_path / "nuqta-ocr-4b-Q4_K_M.gguf")


@pytest.fixture
def projector(tmp_path):
    return write_projector(tmp_path / "nuqta-ocr-4b-mmproj-f16.gguf")


class TestReadInfo:
    def test_reads_the_identifying_metadata_past_the_tokenizer_arrays(self, model):
        info = read_gguf_info(model)

        assert info.version == 3
        assert info.architecture == "qwen35"
        assert info.metadata["general.name"].startswith("Nuqta")
        assert not info.is_projector

    def test_reads_a_projector(self, projector):
        info = read_gguf_info(projector)

        assert info.is_projector
        assert info.has_vision_encoder is True
        assert info.projector_type == "qwen3vl_merger"

    def test_a_missing_file_is_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            read_gguf_info(tmp_path / "absent.gguf")

    def test_a_directory_is_refused(self, tmp_path):
        with pytest.raises(ModelFileError, match="directory"):
            read_gguf_info(tmp_path)

    def test_an_html_page_saved_as_gguf_is_refused(self, tmp_path):
        page = tmp_path / "model.gguf"
        page.write_text("<!DOCTYPE html><title>404</title>", encoding="utf-8")
        with pytest.raises(ModelFileError, match="not a GGUF file"):
            read_gguf_info(page)

    def test_a_truncated_download_is_named_as_such(self, model):
        data = model.read_bytes()
        model.write_bytes(data[: len(data) // 2])
        with pytest.raises(ModelFileError, match="truncated"):
            read_gguf_info(model)

    def test_an_obsolete_gguf_version_is_refused(self, tmp_path):
        old = write_gguf(tmp_path / "old.gguf", [("general.architecture", "llama")], version=1)
        with pytest.raises(ModelFileError, match="version 1"):
            read_gguf_info(old)


class TestPair:
    def test_the_right_pair_is_accepted(self, model, projector):
        model_info, projector_info = check_vision_model_pair(model, projector)
        assert model_info.architecture == "qwen35"
        assert projector_info.is_projector

    def test_swapped_files_are_reported_as_swapped(self, model, projector):
        with pytest.raises(ModelFileError, match="swapped"):
            check_vision_model_pair(projector, model)

    def test_the_same_file_twice_is_refused(self, model):
        with pytest.raises(ModelFileError, match="both point at"):
            check_vision_model_pair(model, model)

    def test_a_language_model_given_as_the_projector_is_refused(self, model, tmp_path):
        other = write_model(tmp_path / "another.gguf", architecture="llama")
        with pytest.raises(ModelFileError, match="not a vision projector"):
            check_vision_model_pair(model, other)

    def test_an_audio_only_projector_is_refused(self, model, tmp_path):
        audio = write_projector(tmp_path / "audio-mmproj.gguf", vision=False)
        with pytest.raises(ModelFileError, match="no vision encoder"):
            check_vision_model_pair(model, audio)

    def test_a_missing_projector_is_file_not_found(self, model, tmp_path):
        with pytest.raises(FileNotFoundError):
            check_vision_model_pair(model, tmp_path / "missing-mmproj.gguf")

    def test_file_errors_are_not_value_errors(self):
        # The HTTP layer maps ValueError to 422 (the client's bad config). A
        # broken model install is the machine's state and must surface as 503.
        assert not issubclass(ModelFileError, ValueError)
