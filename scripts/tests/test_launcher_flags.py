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
from contextlib import redirect_stdout
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


class TensorCoreFlagCompatibility(unittest.TestCase):
    """Kernel selection is automatic; the old spellings survive as deprecated aliases."""

    def _args(self, tensor_cores=False, no_tensor_cores=False, diagnostic=False):
        class A:
            pass

        a = A()
        a.tensor_cores = tensor_cores
        a.no_tensor_cores = no_tensor_cores
        a.diagnostic_scalar_batched_prefill = diagnostic
        return a

    def test_defaults_select_nothing(self):
        a = self._args()
        self.assertEqual(launcher.resolve_deprecated_tensor_core_flags(a), [])
        self.assertFalse(a.diagnostic_scalar_batched_prefill)

    def test_tensor_cores_is_a_deprecated_no_op(self):
        a = self._args(tensor_cores=True)
        warnings = launcher.resolve_deprecated_tensor_core_flags(a)
        self.assertEqual(len(warnings), 1)
        self.assertIn("deprecated", warnings[0])
        self.assertFalse(a.diagnostic_scalar_batched_prefill)

    def test_no_tensor_cores_maps_to_the_diagnostic(self):
        a = self._args(no_tensor_cores=True)
        warnings = launcher.resolve_deprecated_tensor_core_flags(a)
        self.assertEqual(len(warnings), 1)
        self.assertIn("--diagnostic-scalar-batched-prefill", warnings[0])
        self.assertTrue(a.diagnostic_scalar_batched_prefill)

    def test_contradictory_flags_are_rejected(self):
        for kw in ({"no_tensor_cores": True}, {"diagnostic": True}):
            with self.subTest(kw=kw), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.resolve_deprecated_tensor_core_flags(self._args(tensor_cores=True, **kw))

    def test_help_hides_the_deprecated_spellings_and_shows_the_diagnostic(self):
        parser = launcher.create_parser()
        text = parser.format_help()
        self.assertNotIn("--tensor-cores", text)
        self.assertNotIn("--no-tensor-cores", text)
        self.assertIn("--diagnostic-scalar-batched-prefill", text)
        self.assertIn("int8", text)

    def test_only_the_diagnostic_reaches_the_jvm(self):
        parser = launcher.create_parser()
        runner = launcher.LlamaRunner.__new__(launcher.LlamaRunner)
        runner.tornado_sdk = "/nonexistent"
        for argv, expect in (
            (["--gpu", "--model", "m"], False),
            (["--gpu", "--model", "m", "--diagnostic-scalar-batched-prefill"], True),
        ):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                launcher.resolve_deprecated_tensor_core_flags(args)
                self.assertEqual(args.diagnostic_scalar_batched_prefill, expect)


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

    def test_help_exposes_only_the_new_verbosity_interface(self):
        text = launcher.create_parser().format_help()
        self.assertIn("--verbose", text)
        self.assertIn("-v", text)
        self.assertNotIn("--verbose-init", text)
        self.assertNotIn("--no-startup-summary", text)
