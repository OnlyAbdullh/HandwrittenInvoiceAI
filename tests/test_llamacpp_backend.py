"""The llama.cpp backend against a stand-in llama-server.

Everything up to the model itself is real here: the process is launched,
health-polled, authenticated against, sent PNG crops over HTTP and stopped.
Only the reading is fake (tests/fixtures/fake_llama_server.py), so these run on
any machine, with no model and no llama.cpp install.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from PIL import Image

from core.domain.geometry import BoundingBox
from core.domain.ocr import Recognition, RegionCrop, TextRegion
from core.domain.roles import ContentKind
from core.exceptions import ConfigurationError, InferenceServerError, ModelFileError, RecognitionTimeoutError
from ocr.decoding_policy import DATE_CHARS, NUMBER_CHARS
from ocr.llamacpp.client import (
    LlamaCppVisionBackend,
    charset_grammar,
    confidence_from_logprobs,
    encode_image,
    strip_reasoning,
)
from ocr.llamacpp.server import (
    LlamaServer,
    LlamaServerSettings,
    check_extra_args,
    diagnose,
    parse_build_number,
    resolve_executable,
)
from ocr.recognizers import llamacpp_recognizer
from ocr.recognizers.registry import recognizer_registry
from tests.llamacpp_support import fake_launcher, read_record, write_model, write_projector

EXPECTED_CONFIDENCE = round(math.sqrt(0.9 * 0.8), 4)


@pytest.fixture
def model_files(tmp_path):
    return (
        write_model(tmp_path / "nuqta-ocr-4b-Q4_K_M.gguf"),
        write_projector(tmp_path / "nuqta-ocr-4b-mmproj-f16.gguf"),
    )


@pytest.fixture
def record(tmp_path, monkeypatch):
    path = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(path))
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    return path


@pytest.fixture
def make_server(model_files, record):
    started: list[LlamaServer] = []

    def make(startup_timeout_s: float = 30.0, request_timeout_s: float = 10.0, **settings) -> LlamaServer:
        server = LlamaServer(
            LlamaServerSettings(
                launcher=fake_launcher(),
                model_path=model_files[0],
                mmproj_path=model_files[1],
                startup_timeout_s=startup_timeout_s,
                **settings,
            ),
            request_timeout_s=request_timeout_s,
        )
        started.append(server)
        return server

    yield make
    for server in started:
        server.close()


@pytest.fixture
def make_recognizer(model_files, record, monkeypatch):
    monkeypatch.setattr(llamacpp_recognizer, "_launcher", lambda server_path: fake_launcher())
    built = []

    def make(**params):
        params.setdefault("model_path", str(model_files[0]))
        params.setdefault("mmproj_path", str(model_files[1]))
        params.setdefault("startup_timeout_s", 30.0)
        params.setdefault("timeout_s", 10.0)
        recognizer = recognizer_registry.create("llamacpp", **params)
        built.append(recognizer)
        return recognizer

    yield make
    for recognizer in built:
        recognizer.close()


def crop(kind: ContentKind, seed: int = 0, size=(120, 40)) -> RegionCrop:
    rng = np.random.default_rng(seed)
    pixels = np.full((size[1], size[0]), 255, np.uint8)
    pixels[10:30, 10 : size[0] - 10] = rng.integers(0, 120, (20, size[0] - 20), dtype=np.uint8)
    return RegionCrop(
        region=TextRegion(bbox=BoundingBox(10, 20, size[0], size[1]), content_kind=kind), image=pixels
    )


def completions(record) -> list[dict]:
    return [e for e in read_record(record) if e["event"] == "completion"]


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------


class TestServerLifecycle:
    def test_starts_on_loopback_with_a_private_key_and_reports_ready(self, make_server, record):
        server = make_server()
        server.start()

        assert server.is_alive()
        assert server.is_ready()
        assert server.build_info == "b99999-fake"

        launch = read_record(record)[0]
        argv = launch["argv"]
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert argv[argv.index("--parallel") + 1] == "1"
        assert argv[argv.index("--ctx-size") + 1] == "4096"
        assert {"--no-webui", "--offline"} <= set(argv)
        # The key is in the child's environment, never on its command line.
        assert launch["api_key_in_env"] and not launch["api_key_on_command_line"]
        # GPU placement left to llama.cpp's own fit unless configured.
        assert "--n-gpu-layers" not in argv

    def test_gpu_and_cpu_settings_reach_the_command_line(self, make_server, record):
        make_server(gpu_layers="0", mmproj_offload=False, threads=6, ctx_size=2048).start()

        argv = read_record(record)[0]["argv"]
        assert argv[argv.index("--n-gpu-layers") + 1] == "0"
        assert "--no-mmproj-offload" in argv
        assert argv[argv.index("--threads") + 1] == "6"
        assert argv[argv.index("--ctx-size") + 1] == "2048"

    def test_start_is_idempotent(self, make_server, record):
        server = make_server()
        server.start()
        server.start()
        assert len([e for e in read_record(record) if e["event"] == "launch"]) == 1

    def test_waits_through_model_loading(self, make_server, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "load_delay=1.5")
        server = make_server()

        started = time.monotonic()
        server.start()

        assert time.monotonic() - started >= 1.0
        assert server.is_ready()

    def test_a_crash_at_startup_reports_the_servers_own_reason_and_a_hint(self, make_server, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "crash")
        server = make_server()

        with pytest.raises(InferenceServerError) as caught:
            server.start()

        message = str(caught.value)
        assert "exited during startup" in message
        assert "unknown model architecture" in message  # the server's log tail
        assert "b7990" in message  # the plain-language hint
        assert not server.is_alive()

    def test_a_server_that_never_becomes_ready_is_stopped_at_the_deadline(self, make_server, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "load_delay=60")
        server = make_server(startup_timeout_s=1.0)

        with pytest.raises(InferenceServerError, match="did not become ready within 1s"):
            server.start()
        assert not server.is_alive()

    def test_a_server_without_vision_is_refused(self, make_server, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "no_vision")
        server = make_server()

        with pytest.raises(InferenceServerError, match="without image support"):
            server.start()
        assert not server.is_alive()

    def test_close_stops_the_process_and_it_is_never_restarted(self, make_server):
        server = make_server()
        server.start()
        proc = server._proc

        server.close()

        assert proc.poll() is not None
        assert not server.is_ready()
        with pytest.raises(InferenceServerError, match="closed"):
            server.post_json("/v1/chat/completions", {})

    def test_a_server_nobody_references_is_stopped(self, model_files, record):
        # What happens at the end of `python main.py ...`: the orchestrator
        # goes out of scope without anyone calling close().
        import gc

        server = LlamaServer(
            LlamaServerSettings(launcher=fake_launcher(), model_path=model_files[0], mmproj_path=model_files[1])
        )
        server.start()
        proc = server._proc

        del server
        gc.collect()

        assert proc.wait(timeout=15) is not None

    def test_a_cli_run_that_never_closes_leaves_no_server_behind(self, model_files, record, tmp_path):
        import os
        import subprocess
        import sys

        from tests.llamacpp_support import FAKE_SERVER

        script = tmp_path / "cli.py"
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(FAKE_SERVER.parent.parent.parent)!r})\n"
            "from pathlib import Path\n"
            "from ocr.llamacpp.server import LlamaServer, LlamaServerSettings\n"
            f"server = LlamaServer(LlamaServerSettings(launcher=({sys.executable!r}, {str(FAKE_SERVER)!r}),"
            f" model_path=Path({str(model_files[0])!r}), mmproj_path=Path({str(model_files[1])!r})))\n"
            "server.start()\n"
            "print(server._proc.pid, flush=True)\n"
            # Module-level reference alive until interpreter shutdown.
            "keep = server\n",
            encoding="utf-8",
        )
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        pid = int(done.stdout.strip().splitlines()[-1])

        deadline = time.monotonic() + 10
        alive = True
        while alive and time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except OSError:
                alive = False
        assert not alive, "llama-server outlived the process that started it"

    def test_a_dead_server_is_relaunched_by_the_next_request(self, make_server, record, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "die_after=1")
        server = make_server()
        backend = LlamaCppVisionBackend(server, prompts={"any": "p", "number": "p"})
        server.start()

        backend.read(Image.new("RGB", (40, 20), "white"), "number")
        deadline = time.monotonic() + 5
        while server.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not server.is_alive()

        reading = backend.read(Image.new("RGB", (40, 20), "white"), "number")
        assert reading.text == "١٢"
        assert len([e for e in read_record(record) if e["event"] == "launch"]) == 2

    def test_readiness_probe_relaunches_a_dead_server_in_the_background(self, make_server, record):
        server = make_server()
        server.start()
        server._proc.kill()
        server._proc.wait()

        assert not server.is_ready()  # does not block on the relaunch
        deadline = time.monotonic() + 15
        while not server.is_ready() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert server.is_ready()

    def test_a_slow_answer_times_out_without_killing_a_healthy_server(self, make_server, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "slow=3")
        server = make_server(request_timeout_s=0.5)
        backend = LlamaCppVisionBackend(server, prompts={"any": "p"}, timeout_s=0.5)
        server.start()

        with pytest.raises(RecognitionTimeoutError, match="no answer within"):
            backend.read(Image.new("RGB", (40, 20), "white"), "any")
        assert server.is_alive()

    def test_an_http_error_carries_the_servers_message(self, make_server, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "error500")
        server = make_server()
        backend = LlamaCppVisionBackend(server, prompts={"any": "p"})
        server.start()

        with pytest.raises(InferenceServerError, match="HTTP 500: boom"):
            backend.read(Image.new("RGB", (40, 20), "white"), "any")


# --------------------------------------------------------------------------
# Requests and answers
# --------------------------------------------------------------------------


class TestRequestsAndAnswers:
    def test_number_cells_carry_a_grammar_and_no_repetition_penalty(self, make_server, record):
        server = make_server()
        backend = LlamaCppVisionBackend(server, prompts={"any": "p", "number": "digits please"})
        server.start()

        reading = backend.read(Image.new("RGB", (60, 30), "white"), "number")

        sent = completions(record)[0]
        assert sent["body"]["grammar"] == charset_grammar(NUMBER_CHARS)
        assert "repeat_penalty" not in sent["body"]
        assert sent["body"]["temperature"] == 0.0
        assert sent["body"]["max_tokens"] == 16
        assert sent["body"]["logprobs"] is True and sent["body"]["top_logprobs"] >= 1
        assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent["prompt"] == ["digits please"]
        assert reading == Recognition("١٢", confidence=EXPECTED_CONFIDENCE)

    def test_arabic_text_is_unconstrained_penalized_and_returned_intact(self, make_server, record):
        from ocr.engines.qwen_vlm_engine import PROMPTS

        server = make_server()
        backend = LlamaCppVisionBackend(server, prompts=PROMPTS)
        server.start()

        reading = backend.read(Image.new("RGB", (200, 40), "white"), "arabic_text")

        sent = completions(record)[0]
        assert "grammar" not in sent["body"]
        assert sent["body"]["repeat_penalty"] == 1.1
        assert sent["prompt"] == [PROMPTS["arabic_text"]]
        assert reading.text == "بوري شبابي زارا"
        assert reading.confidence == EXPECTED_CONFIDENCE

    def test_large_crops_are_capped_and_small_ones_sent_as_they_are(self, make_server, record):
        server = make_server()
        backend = LlamaCppVisionBackend(server, prompts={"any": "p"}, max_image_side=256)
        server.start()

        backend.read(Image.new("RGB", (1024, 200), "white"), "any")
        backend.read(Image.new("RGB", (100, 40), "white"), "any")

        sizes = [tuple(e["image_size"]) for e in completions(record)]
        assert sizes == [(256, 50), (100, 40)]

    def test_constraints_can_be_switched_off(self, make_server, record):
        server = make_server()
        backend = LlamaCppVisionBackend(server, prompts={"any": "p", "date": "d"}, constrained_decoding=False)
        server.start()

        backend.read(Image.new("RGB", (60, 30), "white"), "date")
        assert "grammar" not in completions(record)[0]["body"]

    def test_missing_logprobs_mean_unknown_confidence_not_zero(self, make_server, monkeypatch, caplog):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "no_logprobs")
        server = make_server()
        backend = LlamaCppVisionBackend(server, prompts={"any": "p"})
        server.start()

        with caplog.at_level("WARNING", logger="ocr.llamacpp.client"):
            first = backend.read(Image.new("RGB", (40, 20), "white"), "any")
            backend.read(Image.new("RGB", (40, 20), "white"), "any")

        assert first.confidence is None
        assert len([r for r in caplog.records if "no token log-probabilities" in r.message]) == 1

    def test_a_thinking_block_never_reaches_the_value(self, make_server, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "think")
        server = make_server()
        backend = LlamaCppVisionBackend(server, prompts={"any": "p", "number": "p"})
        server.start()

        assert backend.read(Image.new("RGB", (40, 20), "white"), "number").text == "١٢"


class TestPureHelpers:
    def test_grammar_lists_every_allowed_character_and_accepts_empty(self):
        grammar = charset_grammar(DATE_CHARS)
        assert grammar.startswith("root ::= char*\n")
        for ch in "٠٩/-.":
            assert f'"{ch}"' in grammar
        assert '"\\\\"' not in grammar  # no backslash allowed in a date

    def test_confidence_is_the_geometric_mean_of_token_probabilities(self):
        entries = [{"logprob": math.log(0.5)}, {"logprob": math.log(0.5)}]
        assert confidence_from_logprobs(entries) == 0.5
        assert confidence_from_logprobs([]) is None
        assert confidence_from_logprobs([{"logprob": None}]) == 0.0

    def test_reasoning_is_stripped_in_both_shapes(self):
        assert strip_reasoning("<think>hmm</think> 125") == "125"
        assert strip_reasoning("hmm</think>125") == "125"
        assert strip_reasoning("<think>never closed") == ""
        assert strip_reasoning("بوري") == "بوري"

    def test_images_are_sent_as_png(self):
        assert encode_image(Image.new("RGB", (10, 10))).startswith("data:image/png;base64,")

    def test_build_numbers_in_both_version_formats(self):
        assert parse_build_number("version: 7990 (abc123)\nbuilt with gcc") == 7990
        assert parse_build_number("version: 0.8.0-dev (build 12092, commit 20a51f24a)") == 12092
        assert parse_build_number("nonsense") is None

    def test_diagnosis_of_common_failures(self):
        assert "gpu_layers" in diagnose("ggml_cuda: cudaMalloc failed: out of memory")
        assert "b7990" in diagnose("error: unknown model architecture: 'qwen35'")
        assert diagnose("all fine") == ""

    def test_extra_args_cannot_move_the_server_off_loopback(self):
        assert check_extra_args(["--flash-attn", "on"]) == ("--flash-attn", "on")
        for forbidden in (["--host", "0.0.0.0"], ["--port=80"], ["--api-key", "x"], ["-m", "other.gguf"]):
            with pytest.raises(ValueError, match="may not set"):
                check_extra_args(forbidden)

    def test_a_winget_install_is_found_when_path_does_not_say(self, tmp_path, monkeypatch):
        package = tmp_path / "Microsoft" / "WinGet" / "Packages" / "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
        package.mkdir(parents=True)
        (package / "llama-server.exe").write_text("")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setenv("PATH", "")

        assert resolve_executable("llama-server") == str(package / "llama-server.exe")

    def test_executable_resolution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "no-winget-here"))
        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            resolve_executable("llama-server-that-does-not-exist")
        with pytest.raises(FileNotFoundError, match="not found at"):
            resolve_executable(str(tmp_path / "bin" / "llama-server.exe"))
        with pytest.raises(FileNotFoundError, match="No llama-server executable inside"):
            resolve_executable(str(tmp_path))

        binary = tmp_path / "llama-server"
        binary.write_text("")
        assert resolve_executable(str(tmp_path)) == str(binary)
        assert resolve_executable(str(binary)) == str(binary)


# --------------------------------------------------------------------------
# The recognizer: registry, configuration, the pipeline's contract
# --------------------------------------------------------------------------


class TestRecognizer:
    def test_is_registered_beside_the_transformers_recognizer(self):
        assert {"llamacpp", "qwen", "echo"} <= set(recognizer_registry.available())

    def test_reads_crops_with_the_prompt_their_kind_implies(self, make_recognizer, record):
        recognizer = make_recognizer()

        readings = recognizer.read_all(
            [crop(ContentKind.NUMBER, 1), crop(ContentKind.ARABIC_TEXT, 2), crop(ContentKind.CODE, 3)]
        )

        # The code reading went through output validation: caption stripped.
        assert [r.text for r in readings] == ["١٢", "بوري شبابي زارا", "00010"]
        assert all(r.confidence == EXPECTED_CONFIDENCE for r in readings)
        assert recognizer.is_ready()

    def test_a_failed_cell_comes_back_empty_without_failing_the_page(self, make_recognizer, monkeypatch):
        monkeypatch.setenv("FAKE_LLAMA_MODE", "error500")
        recognizer = make_recognizer()

        readings = recognizer.read_all([crop(ContentKind.NUMBER, 1), crop(ContentKind.ANY, 2)])

        assert readings == [Recognition(""), Recognition("")]

    def test_readings_that_cannot_fit_the_cell_are_rejected(self, make_recognizer, monkeypatch):
        recognizer = make_recognizer(constrained_decoding=False)
        # Without the grammar the fake answers a date cell with prose -- which
        # is exactly what output validation exists to refuse.
        monkeypatch.setattr(
            recognizer._backend, "recognition_from", lambda body: Recognition("وهو نائم", confidence=0.9)
        )

        assert recognizer.read(crop(ContentKind.DATE)) == Recognition("", confidence=0.0)

    def test_missing_model_files_fail_before_anything_starts(self, make_recognizer, tmp_path, record):
        with pytest.raises(FileNotFoundError):
            make_recognizer(model_path=str(tmp_path / "missing.gguf"))
        with pytest.raises(FileNotFoundError):
            make_recognizer(mmproj_path=str(tmp_path / "missing-mmproj.gguf"))
        assert read_record(record) == []  # no server was launched

    def test_swapped_model_files_fail_before_anything_starts(self, make_recognizer, model_files, record):
        with pytest.raises(ModelFileError, match="swapped"):
            make_recognizer(model_path=str(model_files[1]), mmproj_path=str(model_files[0]))
        assert read_record(record) == []

    @pytest.mark.parametrize(
        "params, message",
        [
            ({"gpu_layers": "most"}, "gpu_layers"),
            ({"ctx_size": 128}, "ctx_size"),
            ({"batch_size": 0}, "batch_size"),
            ({"port": 80}, "port"),
            ({"prompts": {"currency": "x"}}, "unknown content kind"),
            ({"extra_args": ["--host", "0.0.0.0"]}, "may not set"),
        ],
    )
    def test_bad_settings_are_rejected_before_a_model_loads(self, make_recognizer, record, params, message):
        with pytest.raises(ValueError, match=message):
            make_recognizer(**params)
        assert read_record(record) == []

    def test_an_unknown_parameter_is_rejected_by_the_registry(self, make_recognizer):
        with pytest.raises(ConfigurationError, match="backend"):
            make_recognizer(backend="llamacpp")

    @pytest.mark.parametrize(
        "value, expected", [("auto", None), (None, None), ("all", "all"), (0, "0"), ("24", "24")]
    )
    def test_gpu_layers_accepts_what_yaml_and_the_environment_produce(self, value, expected):
        assert llamacpp_recognizer._gpu_layers(value) == expected

    def test_prompts_can_be_overridden_per_kind(self, make_recognizer, record):
        recognizer = make_recognizer(prompts={"number": "اكتب الرقم فقط"})
        recognizer.read(crop(ContentKind.NUMBER))
        assert completions(record)[0]["prompt"] == ["اكتب الرقم فقط"]


class TestModelCardSettings:
    """Nuqta's model card: its own image size, no repeat penalty, its own prompt."""

    def _argv(self, record):
        return [e for e in read_record(record) if e["event"] == "launch"][-1]["argv"]

    def test_auto_respects_the_image_size_the_projector_declares(self, make_recognizer, tmp_path, record):
        declared = write_projector(tmp_path / "declares-mmproj.gguf", min_pixels=1_000_000)
        make_recognizer(mmproj_path=str(declared))
        # Nothing passed: llama.cpp then applies the projector's own ~1 MP floor.
        assert "--image-min-tokens" not in self._argv(record)

    def test_auto_falls_back_to_the_hf_floor_when_the_projector_says_nothing(self, make_recognizer, record):
        make_recognizer()
        argv = self._argv(record)
        assert argv[argv.index("--image-min-tokens") + 1] == "64"

    @pytest.mark.parametrize("setting, expected", [(256, "256"), (None, None)])
    def test_an_explicit_setting_wins(self, make_recognizer, record, setting, expected):
        make_recognizer(image_min_tokens=setting)
        argv = self._argv(record)
        assert (argv[argv.index("--image-min-tokens") + 1] if "--image-min-tokens" in argv else None) == expected

    def test_a_bad_image_size_is_refused(self, make_recognizer):
        with pytest.raises(ValueError, match="image_min_tokens"):
            make_recognizer(image_min_tokens="big")

    def test_the_repeat_penalty_override_applies_to_every_kind(self, make_recognizer, record):
        recognizer = make_recognizer(repetition_penalty=1.0)
        recognizer.read_all([crop(ContentKind.NUMBER, 1), crop(ContentKind.ARABIC_TEXT, 2)])
        assert [c["body"]["repeat_penalty"] for c in completions(record)] == [1.0, 1.0]

    def test_the_gguf_header_reports_the_declared_floor(self, tmp_path):
        from ocr.llamacpp.gguf import read_gguf_info

        assert read_gguf_info(write_projector(tmp_path / "a.gguf", min_pixels=1_000_000)).image_min_pixels == 1_000_000
        assert read_gguf_info(write_projector(tmp_path / "b.gguf")).image_min_pixels is None
