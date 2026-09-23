"""CLI contracts: modes, validation before loading, and Java argument forwarding."""
import io
import unittest
from contextlib import redirect_stderr
from test_launcher_flags import launcher


class Commands(unittest.TestCase):
    def parse(self, *args):
        with redirect_stderr(io.StringIO()):
            return launcher.parse_cli_args(list(args))

    def reject(self, *args):
        with self.assertRaises(SystemExit) as failure:
            self.parse(*args)
        self.assertEqual(2, failure.exception.code)

    def java_args(self, *args):
        parsed = self.parse(*args)
        runner = launcher.LlamaRunner.__new__(launcher.LlamaRunner)
        return runner._add_llama_args([], parsed)

    def test_subcommands_and_legacy_modes(self):
        for command, alias in [("run", "--instruct"), ("chat", "--interactive"),
                               ("serve", "--server"), ("bench", "--bench")]:
            prompt = ["--prompt", "hello"] if command == "run" else []
            for prefix in ([command], [alias]):
                with self.subTest(command=command, prefix=prefix):
                    args = self.parse(*prefix, "--model", "stub.gguf", *prompt)
                    self.assertEqual(command, args.command)
        self.assertEqual("run", self.parse("--model", "stub.gguf", "--prompt", "hello").command)
        self.assertEqual("chat", self.parse("--chat", "--model", "stub.gguf").command)

    def test_conflicting_modes_and_irrelevant_options_fail(self):
        for flags in [["--server", "--bench"], ["--interactive", "--instruct"],
                      ["chat", "--server"], ["run", "--prompt", "hi", "--port", "9000"],
                      ["serve", "--prompt", "ignored"], ["bench", "--temperature", "0"],
                      ["--server", "--temperature", "0"], ["chat", "--prompt", "ignored"]]:
            self.reject(*flags, "--model", "stub.gguf")

    def test_context_and_output_limit_are_independent(self):
        args = self.java_args("run", "--model", "stub.gguf", "--prompt", "hi",
                              "--ctx-size", "2048", "--max-new-tokens", "16", "--stream", "false")
        self.assertEqual("2048", args[args.index("--ctx-size") + 1])
        self.assertEqual("16", args[args.index("--max-new-tokens") + 1])
        self.assertEqual("false", args[args.index("--stream") + 1])
        legacy = self.parse("--model", "stub.gguf", "--prompt", "hi", "-n", "1024")
        self.assertEqual(1024, legacy.max_tokens)
        self.assertTrue(legacy.legacy_context)
        self.reject("run", "-m", "stub.gguf", "--prompt", "hi", "-c", "100", "-n", "200")
        self.reject("run", "-m", "stub.gguf", "--prompt", "hi", "--max-new-tokens", "0")
        self.reject("run", "-m", "stub.gguf")

    def test_server_forwards_host_capacity_and_request_slots(self):
        args = self.java_args("serve", "-m", "stub.gguf", "--gpu", "--host", "127.0.0.2",
                              "--port", "8081", "-c", "4096", "--parallel", "4",
                              "--max-queued-requests", "12", "--prefix-cache-entries", "8")
        for key, value in [("--host", "127.0.0.2"), ("--port", "8081"),
                           ("--ctx-size", "4096"), ("--parallel", "4"),
                           ("--max-queued-requests", "12"), ("--prefix-cache-entries", "8")]:
            self.assertEqual(value, args[args.index(key) + 1])
        self.assertNotIn("--prompt", args)
        self.assertNotIn("--temperature", args)
        self.reject("serve", "-m", "stub.gguf", "--parallel", "2")
        self.reject("serve", "-m", "stub.gguf", "--gpu", "--parallel", "2", "--cuda-graphs")
        self.reject("serve", "-m", "stub.gguf", "--gpu", "--parallel", "2", "--fp16-kv-cache")

    def test_server_context_defaults_to_model_and_accepts_ctx_aliases(self):
        args = self.java_args("--server", "-m", "stub.gguf", "--gpu", "--port", "8090")
        self.assertEqual("0", args[args.index("--ctx-size") + 1])
        for flag in ("--ctx", "--context-length"):
            args = self.java_args("serve", "-m", "stub.gguf", flag, "8192")
            self.assertEqual("8192", args[args.index("--ctx-size") + 1])
        self.assertEqual(512, self.parse("run", "-m", "stub.gguf", "--prompt", "hi").max_tokens)
        self.reject("run", "-m", "stub.gguf", "--prompt", "hi", "--ctx", "0")
        self.reject("bench", "-m", "stub.gguf", "--ctx", "1024")

    def test_benchmark_workloads_and_quoted_legacy_arguments(self):
        args = self.java_args("bench", "-m", "stub.gguf", "--gpu", "--pp", "32,64",
                              "--tg", "16", "--depth", "0,128", "--repetitions", "2",
                              "--output", "json", "--batch-prefill-size", "32",
                              "--bench-args=--expect 'qwen3 / F16 / batch-prefill-decode'")
        for key, value in [("-p", "32,64"), ("-n", "16"), ("-d", "0,128"),
                           ("-r", "2"), ("-o", "json"), ("-b", "32"),
                           ("--expect", "qwen3 / F16 / batch-prefill-decode")]:
            self.assertEqual(value, args[args.index(key) + 1])
        self.assertNotIn("--cpu", args)
        self.assertIn("--cpu", self.java_args("bench", "-m", "stub.gguf"))
        self.reject("bench", "-m", "stub.gguf", "--depth", "-1")

    def test_focused_help_and_global_help(self):
        for command in ("run", "chat", "serve", "bench"):
            help_text = launcher.create_parser(command).format_help()
            self.assertIn("--verbose", help_text)
            self.assertEqual(command == "serve", "--port" in help_text)
            self.assertEqual(command == "bench", "--pp" in help_text)
            self.assertEqual(command == "run", "--prompt " in help_text)
