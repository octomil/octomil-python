"""Tests for edgeml.scanner — inference point detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from edgeml.scanner import InferencePoint, format_json, format_text, scan_directory


# ---------------------------------------------------------------------------
# Fixtures: sample source files
# ---------------------------------------------------------------------------

SWIFT_COREML = """\
import CoreML
import Vision

class ImageClassifier {
    var model: MLModel?

    func loadModel() {
        model = try? MLModel(contentsOf: url)
    }

    func loadAsync() {
        MLModel.load(contentsOf: url) { result in
            self.model = try? result.get()
        }
    }

    func predict(image: CVPixelBuffer) -> String {
        let visionModel = try! VNCoreMLModel(for: model!)
        let request = VNCoreMLRequest(model: visionModel)
        return ""
    }
}
"""

SWIFT_NLP = """\
import CoreML

class TextProcessor {
    func setupNLP() {
        let coreModel = try! MLModel(contentsOf: modelURL)
        let nlModel = NLModel(mlModel: coreModel)
        let array = MLMultiArray(shape: [1, 256], dataType: .float32)
        let provider: MLFeatureProvider = input
    }
}
"""

KOTLIN_TFLITE = """\
package com.example.ml

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

class InferenceEngine {
    private lateinit var interpreter: Interpreter

    fun loadModel(modelFile: File) {
        val options = Interpreter.Options()
        options.addDelegate(GpuDelegate())
        options.addDelegate(NnApiDelegate())
        val buffer = loadModelFile(modelFile)
        interpreter = Interpreter(buffer, options)
    }

    fun run(input: MappedByteBuffer): FloatArray {
        val output = FloatArray(10)
        interpreter.run(input, output)
        return output
    }
}
"""

PYTHON_PYTORCH = """\
import torch

model = torch.load("model.pt")
model.eval()

def run():
    output = model.forward(input_tensor)
    return output
"""

PYTHON_OPENAI = """\
import openai

client = openai.OpenAI(api_key="sk-test")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)
"""

PYTHON_ONNX = """\
import onnxruntime

session = onnxruntime.InferenceSession("model.onnx")
result = session.run(None, {"input": data})
"""

PYTHON_TFLITE = """\
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
"""

PYTHON_MLX = """\
import mlx.core as mx
import mlx_lm

model = mlx_lm.load("model")
"""

PYTHON_TRANSFORMERS = """\
from transformers import AutoModelForSequenceClassification, pipeline

model = AutoModelForSequenceClassification.from_pretrained("bert-base")
pipe = transformers.pipeline("text-classification", model=model)
"""

PYTHON_GENERAL = """\
def classify(image):
    return model.predict(image)

def detect(frame):
    return detector.inference(frame)
"""

# A file that only has comments — should produce no matches.
PYTHON_COMMENTS_ONLY = """\
# torch.load("model.pt")
# model.eval()
# openai.OpenAI(api_key="sk-test")
'''
torch.load("should not match")
'''
\"\"\"
onnxruntime.InferenceSession("nope")
\"\"\"
"""

# A file with no inference at all.
PYTHON_NO_INFERENCE = """\
import os
import sys

def hello():
    print("Hello, world!")

x = 1 + 2
"""


# ---------------------------------------------------------------------------
# Helper to set up temp directory with fixture files
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    """Create a temp directory mimicking a project with multi-platform sources."""
    # iOS files
    ios_dir = tmp_path / "src"
    ios_dir.mkdir()
    (ios_dir / "ImageClassifier.swift").write_text(SWIFT_COREML)
    (ios_dir / "TextProcessor.swift").write_text(SWIFT_NLP)

    # Android files
    android_dir = tmp_path / "app" / "src" / "main" / "kotlin"
    android_dir.mkdir(parents=True)
    (android_dir / "InferenceEngine.kt").write_text(KOTLIN_TFLITE)

    # Python files
    py_dir = tmp_path / "ml"
    py_dir.mkdir()
    (py_dir / "pytorch_model.py").write_text(PYTHON_PYTORCH)
    (py_dir / "openai_client.py").write_text(PYTHON_OPENAI)
    (py_dir / "onnx_runner.py").write_text(PYTHON_ONNX)
    (py_dir / "tflite_runner.py").write_text(PYTHON_TFLITE)
    (py_dir / "mlx_model.py").write_text(PYTHON_MLX)
    (py_dir / "hf_transformers.py").write_text(PYTHON_TRANSFORMERS)
    (py_dir / "general_inference.py").write_text(PYTHON_GENERAL)
    (py_dir / "comments_only.py").write_text(PYTHON_COMMENTS_ONLY)
    (py_dir / "no_inference.py").write_text(PYTHON_NO_INFERENCE)

    # Model files
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "classifier.mlmodel").write_bytes(b"fake")
    (models_dir / "detector.tflite").write_bytes(b"fake")
    (models_dir / "encoder.onnx").write_bytes(b"fake")
    (models_dir / "weights.pt").write_bytes(b"fake")
    (models_dir / "weights.safetensors").write_bytes(b"fake")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: scan_directory
# ---------------------------------------------------------------------------


class TestScanDirectory:
    """Tests for the core scan_directory function."""

    def test_finds_ios_coreml_patterns(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="ios")
        types = {pt.type for pt in points}
        assert "CoreML model loading" in types
        assert "CoreML Vision model" in types
        assert "CoreML Vision request" in types
        assert "CoreML NLP model" in types
        assert "CoreML import" in types

    def test_finds_android_tflite_patterns(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="android")
        types = {pt.type for pt in points}
        assert "TFLite model loading" in types
        assert "TFLite interpreter options" in types
        assert "TFLite GPU delegate" in types
        assert "TFLite NNAPI delegate" in types
        assert "TFLite import" in types

    def test_finds_python_pytorch(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        patterns = {pt.pattern for pt in points}
        assert "torch.load(" in patterns
        assert "model.eval()" in patterns
        assert "model.forward(" in patterns

    def test_finds_python_openai(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        patterns = {pt.pattern for pt in points}
        assert "openai.OpenAI(" in patterns
        assert "client.chat.completions" in patterns

    def test_finds_python_onnx(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        patterns = {pt.pattern for pt in points}
        assert "onnxruntime.InferenceSession" in patterns

    def test_finds_python_tflite(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        patterns = {pt.pattern for pt in points}
        assert "tf.lite.Interpreter" in patterns

    def test_finds_python_mlx(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        patterns = {pt.pattern for pt in points}
        assert "mlx.core" in patterns
        assert "mlx_lm" in patterns

    def test_finds_python_transformers(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        patterns = {pt.pattern for pt in points}
        assert "AutoModelFor*" in patterns
        assert "transformers.pipeline(" in patterns

    def test_finds_general_patterns(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        patterns = {pt.pattern for pt in points}
        assert "predict(" in patterns
        assert "classify(" in patterns
        assert "detect(" in patterns
        assert "inference(" in patterns

    def test_finds_model_files(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project))
        model_points = [pt for pt in points if pt.line == 0]
        model_files = {pt.pattern for pt in model_points}
        assert "classifier.mlmodel" in model_files
        assert "detector.tflite" in model_files
        assert "encoder.onnx" in model_files
        assert "weights.pt" in model_files
        assert "weights.safetensors" in model_files

    def test_platform_filter_ios(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="ios")
        for pt in points:
            assert pt.platform == "ios"

    def test_platform_filter_android(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="android")
        for pt in points:
            assert pt.platform == "android"

    def test_platform_filter_python(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        for pt in points:
            assert pt.platform == "python"

    def test_all_platforms_returns_all(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project))
        platforms = {pt.platform for pt in points}
        assert "ios" in platforms
        assert "android" in platforms
        assert "python" in platforms

    def test_no_false_positives_on_comments(self, sample_project: Path) -> None:
        """Comments-only file should produce zero inference points."""
        points = scan_directory(str(sample_project), platform="python")
        comment_file_points = [pt for pt in points if "comments_only" in pt.file]
        assert len(comment_file_points) == 0

    def test_no_matches_in_clean_file(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        clean_file_points = [pt for pt in points if "no_inference" in pt.file]
        assert len(clean_file_points) == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        points = scan_directory(str(tmp_path))
        assert points == []

    def test_nonexistent_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            scan_directory("/nonexistent/path/that/does/not/exist")

    def test_skips_hidden_and_build_dirs(self, tmp_path: Path) -> None:
        """Ensure .git, node_modules, __pycache__ etc. are skipped."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "hooks.py").write_text("torch.load('x')\n")

        nm_dir = tmp_path / "node_modules"
        nm_dir.mkdir()
        (nm_dir / "deep.py").write_text("openai.OpenAI()\n")

        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("model.eval()\n")

        points = scan_directory(str(tmp_path))
        assert len(points) == 0

    def test_inference_point_has_context(self, sample_project: Path) -> None:
        points = scan_directory(str(sample_project), platform="python")
        pytorch_points = [pt for pt in points if pt.pattern == "torch.load("]
        assert len(pytorch_points) > 0
        pt = pytorch_points[0]
        assert "torch.load" in pt.context
        assert pt.line > 0
        assert pt.file != ""


# ---------------------------------------------------------------------------
# Tests: formatting
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_no_points(self) -> None:
        output = format_text([])
        assert "No inference points found" in output

    def test_with_points(self) -> None:
        points = [
            InferencePoint(
                file="src/Model.swift",
                line=42,
                pattern="MLModel.load()",
                type="CoreML model loading",
                platform="ios",
                suggestion="Wrap with EdgeML.wrap(model) for telemetry",
                context="MLModel.load(contentsOf: url) { result in",
            ),
        ]
        output = format_text(points)
        assert "src/Model.swift:42" in output
        assert "MLModel.load()" in output
        assert "CoreML model loading" in output
        assert "Summary: 1 inference point(s)" in output
        assert "iOS (CoreML): 1" in output

    def test_summary_counts_platforms(self) -> None:
        points = [
            InferencePoint("a.swift", 1, "MLModel.load()", "CoreML", "ios", "", ""),
            InferencePoint("b.kt", 1, "Interpreter(", "TFLite", "android", "", ""),
            InferencePoint("c.py", 1, "torch.load(", "PyTorch", "python", "", ""),
            InferencePoint("d.py", 2, "model.eval()", "PyTorch", "python", "", ""),
        ]
        output = format_text(points)
        assert "4 inference point(s) found across 4 file(s)" in output
        assert "iOS (CoreML): 1" in output
        assert "Android (TFLite): 1" in output
        assert "Python: 2" in output


class TestFormatJson:
    def test_empty(self) -> None:
        data = json.loads(format_json([]))
        assert data["total"] == 0
        assert data["points"] == []

    def test_structure(self) -> None:
        points = [
            InferencePoint(
                file="src/Model.swift",
                line=42,
                pattern="MLModel.load()",
                type="CoreML model loading",
                platform="ios",
                suggestion="Wrap with EdgeML.wrap(model) for telemetry",
                context="MLModel.load(contentsOf: url)",
            ),
        ]
        data = json.loads(format_json(points))
        assert data["total"] == 1
        assert data["files"] == 1
        assert data["platforms"] == {"ios": 1}
        pt = data["points"][0]
        assert pt["file"] == "src/Model.swift"
        assert pt["line"] == 42
        assert pt["pattern"] == "MLModel.load()"
        assert pt["type"] == "CoreML model loading"
        assert pt["platform"] == "ios"


# ---------------------------------------------------------------------------
# Tests: CLI integration
# ---------------------------------------------------------------------------


class TestScanCLI:
    def test_scan_text_output(self, sample_project: Path) -> None:
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(sample_project)])
        assert result.exit_code == 0
        assert "Scanning for inference points" in result.output
        assert "inference point(s)" in result.output

    def test_scan_json_output(self, sample_project: Path) -> None:
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(sample_project), "--format", "json"])
        assert result.exit_code == 0
        # The JSON block starts after the "Scanning..." line
        lines = result.output.strip().split("\n")
        # Find the first line that starts with '{'
        json_start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("{"))
        json_str = "\n".join(lines[json_start:])
        data = json.loads(json_str)
        assert "total" in data
        assert "points" in data

    def test_scan_platform_filter(self, sample_project: Path) -> None:
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(sample_project), "--platform", "ios"])
        assert result.exit_code == 0
        assert "iOS (CoreML)" in result.output
        # Should NOT have Android or Python platform labels
        assert "Android (TFLite)" not in result.output
        assert "Python:" not in result.output

    def test_scan_nonexistent_path(self, tmp_path: Path) -> None:
        from edgeml.cli import main

        runner = CliRunner()
        bad_path = str(tmp_path / "does_not_exist")
        result = runner.invoke(main, ["scan", bad_path])
        assert result.exit_code != 0

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(tmp_path)])
        assert result.exit_code == 0
        assert "No inference points found" in result.output

    def test_scan_json_empty(self, tmp_path: Path) -> None:
        from edgeml.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(tmp_path), "--format", "json"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        json_start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("{"))
        json_str = "\n".join(lines[json_start:])
        data = json.loads(json_str)
        assert data["total"] == 0
