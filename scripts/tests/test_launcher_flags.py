#!/usr/bin/env python3
"""
Class A cover for the launcher's backend-flag validation (T12.1d). No accelerator, no
TornadoVM SDK, no model: the validation runs before any of those are touched, which is
the point of testing it here rather than through a launch.

Run with: python3 -m unittest discover -s scripts/tests
"""

import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_launcher():
    """Import `jllm` by path -- its name is not a valid module identifier."""
    spec = importlib.util.spec_from_loader(
        "llama_tornado_launcher",
        importlib.machinery.SourceFileLoader(
            "llama_tornado_launcher", str(REPO_ROOT / "jllm")
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher()


class Args:
    """Only the two fields the validation reads."""

    def __init__(self, backend_override, use_gpu):
        self.backend_override = backend_override
        self.use_gpu = use_gpu


class BackendFlagValidation(unittest.TestCase):

    def _reject(self, backend):
        args = Args(backend, use_gpu=False)
        out = io.StringIO()
        with self.assertRaises(SystemExit) as exit_ctx, redirect_stdout(out):
            launcher.validate_backend_selection(args)
        return exit_ctx.exception.code, out.getvalue()

    def test_every_backend_flag_without_gpu_is_rejected(self):
        for backend in launcher.Backend:
            with self.subTest(backend=backend.value):
                code, message = self._reject(backend)
                self.assertEqual(1, code)
                # The message must name --gpu: that is the fix the user has to apply.
                self.assertIn("--gpu", message)
                self.assertIn(f"--{backend.value}", message)

    def test_the_message_says_the_model_would_run_on_the_cpu(self):
        # The failure this guards against is silent and looks like slow GPU execution,
        # so naming the CPU is the part that makes it recognizable.
        _, message = self._reject(launcher.Backend.CUDA)
        self.assertIn("CPU", message)

    def test_backend_flag_with_gpu_is_accepted(self):
        for backend in launcher.Backend:
            with self.subTest(backend=backend.value):
                launcher.validate_backend_selection(Args(backend, use_gpu=True))

    def test_no_backend_flag_is_accepted_either_way(self):
        launcher.validate_backend_selection(Args(None, use_gpu=False))
        launcher.validate_backend_selection(Args(None, use_gpu=True))


if __name__ == "__main__":
    unittest.main()


class RemovedKernelSelectionFlags(unittest.TestCase):
    """Kernel selection is automatic; the scalar-prefill diagnostic is a Java property only."""

    def test_the_diagnostic_and_its_old_spellings_are_gone(self):
        text = launcher.create_parser().format_help()
        for flag in ("--diagnostic-scalar-batched-prefill", "--no-tensor-cores", "--tensor-cores"):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()):
                self.assertNotIn(flag, text)
                with self.assertRaises(SystemExit):
                    launcher.create_parser().parse_args(["--model", "m", flag])


class VerbosityOptions(unittest.TestCase):
    def test_verbose_aliases_and_quiet_default(self):
        for flags, expected in [([], False), (["-v"], True), (["--verbose"], True)]:
            args = launcher.create_parser().parse_args(["--model", "stub.gguf"] + flags)
            self.assertEqual([], launcher.resolve_verbosity(args))
            self.assertEqual(expected, args.verbose)

    def test_deprecated_alias_enables_the_same_output_and_warns(self):
        args = launcher.create_parser().parse_args(["--model", "stub.gguf", "--verbose-init"])
        warnings = launcher.resolve_verbosity(args)
        self.assertTrue(args.verbose)
        self.assertEqual(1, len(warnings))
        self.assertIn("deprecated", warnings[0])
        self.assertIn("--verbose", warnings[0])

    def test_verbose_forwards_only_the_report_property(self):
        args = launcher.create_parser().parse_args(["--model", "stub.gguf", "-v"])
        with tempfile.TemporaryDirectory() as sdk:
            open(os.path.join(sdk, "tornado-argfile"), "w").close()
            os.makedirs(os.path.join(sdk, "target"))
            open(os.path.join(sdk, "target", "jllm-1.0.0-jdk21.jar"), "w").close()
            runner = launcher.LlamaRunner.__new__(launcher.LlamaRunner)
            runner.tornado_sdk = sdk
            runner.java_home, runner.llama_root = "/stub/java", sdk
            args.installed_backends, args.backend = [launcher.Backend.CUDA], launcher.Backend.CUDA
            cmd = runner._build_base_command(args)
        self.assertIn("-Djllm.verbose=true", cmd)
        self.assertNotIn("-Djllm.EnableTimingForTornadoVMInit=true", cmd)

    def base_command(self, *flags):
        args = launcher.create_parser().parse_args(["--model", "stub.gguf", *flags])
        with tempfile.TemporaryDirectory() as sdk:
            open(os.path.join(sdk, "tornado-argfile"), "w").close()
            os.makedirs(os.path.join(sdk, "target"))
            open(os.path.join(sdk, "target", "jllm-1.0.0-jdk21.jar"), "w").close()
            runner = launcher.LlamaRunner.__new__(launcher.LlamaRunner)
            runner.tornado_sdk = sdk
            runner.java_home, runner.llama_root = "/stub/java", sdk
            args.installed_backends, args.backend = [launcher.Backend.CUDA], launcher.Backend.CUDA
            return runner._build_base_command(args)

    def test_kv_cache_is_fp16_unless_fp32_is_asked_for(self):
        default = self.base_command()
        self.assertFalse([a for a in default if "kvcache" in a], "FP16 is the Java default")
        self.assertIn("-Djllm.kvcache.fp32=true", self.base_command("--fp32-kv-cache"))
        self.assertNotIn("--fp16-kv-cache", launcher.create_parser().format_help())
        self.assertIn("--fp32-kv-cache", launcher.create_parser().format_help())

    def test_native_libraries_forward_the_property_only_when_asked(self):
        self.assertFalse([a for a in self.base_command() if "nativeLibraries" in a])
        self.assertIn("-Djllm.nativeLibraries=true", self.base_command("--with-native-libraries"))

    def test_help_exposes_only_the_new_verbosity_interface(self):
        text = launcher.create_parser().format_help()
        self.assertIn("--verbose", text)
        self.assertIn("-v", text)
        self.assertNotIn("--verbose-init", text)
        self.assertNotIn("--no-startup-summary", text)
