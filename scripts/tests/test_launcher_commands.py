"""CLI contracts: modes, validation before loading, and Java argument forwarding."""
import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
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

    def test_server_forwards_host_capacity_and_continuous_batching(self):
        args = self.java_args("serve", "-m", "stub.gguf", "--gpu", "--fp32-kv-cache", "--host", "127.0.0.2",
                              "--port", "8081", "-c", "4096", "--continuous-batching", "4",
                              "--max-queued-requests", "12", "--prefix-cache-entries", "8")
        for key, value in [("--host", "127.0.0.2"), ("--port", "8081"),
                           ("--ctx-size", "4096"), ("--continuous-batching", "4"),
                           ("--max-queued-requests", "12"), ("--prefix-cache-entries", "8")]:
            self.assertEqual(value, args[args.index(key) + 1])
        self.assertNotIn("--prompt", args)
        self.assertNotIn("--temperature", args)
        # Off unless asked for, and then its options are not forwarded either.
        plain = self.java_args("serve", "-m", "stub.gguf", "--gpu")
        self.assertNotIn("--continuous-batching", plain)
        self.assertNotIn("--max-queued-requests", plain)
        self.reject("serve", "-m", "stub.gguf", "--continuous-batching", "2", "--fp32-kv-cache")
        self.reject("serve", "-m", "stub.gguf", "--gpu", "--fp32-kv-cache", "--continuous-batching", "1")
        self.reject("serve", "-m", "stub.gguf", "--gpu", "--fp32-kv-cache", "--continuous-batching", "2", "--cuda-graphs")
        # Either cache: the engine reads FP16 (the default) and FP32 pools alike.
        self.parse("serve", "-m", "stub.gguf", "--gpu", "--continuous-batching", "2")
        # Its options are refused on their own rather than silently doing nothing.
        self.reject("serve", "-m", "stub.gguf", "--gpu", "--prefix-cache-entries", "4")
        self.reject("serve", "-m", "stub.gguf", "--gpu", "--max-queued-requests", "4")
        self.reject("run", "-m", "stub.gguf", "--prompt", "hi", "--continuous-batching", "2")
        self.parse("serve", "-m", "stub.gguf", "--gpu", "--continuous-batching", "2", "--fp32-kv-cache")

    def test_native_libraries_are_opt_in_and_experimental(self):
        for command in ("run", "chat", "serve", "bench"):
            help_text = launcher.create_parser(command).format_help()
            advanced = help_text[help_text.index("Advanced Options"):]
            self.assertIn("--with-native-libraries", advanced)
            self.assertIn("[experimental] Use cuBLAS/cuDNN", advanced)
        self.assertFalse(self.parse("run", "-m", "stub.gguf", "--prompt", "hi").with_native_libraries)
        self.assertTrue(self.parse("run", "-m", "stub.gguf", "--prompt", "hi",
                                   "--with-native-libraries").with_native_libraries)
        # No native implementation in the continuous-batch engine.
        self.reject("serve", "-m", "stub.gguf", "--gpu", "--continuous-batching", "2",
                    "--with-native-libraries")

    def test_continuous_batching_is_listed_as_experimental(self):
        serve_help = launcher.create_parser("serve").format_help()
        advanced = serve_help[serve_help.index("Advanced Options"):]
        self.assertIn("--continuous-batching", advanced)
        self.assertIn("[experimental] (serve) Decode", advanced)
        self.assertNotIn("--parallel", serve_help)
        self.assertNotIn("--continuous-batching", launcher.create_parser("run").format_help())
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit):
            launcher.parse_cli_args(["--help"])
        self.assertIn("[experimental]", stdout.getvalue())
        self.assertIn("--continuous-batching", stdout.getvalue())

    def test_global_help_lists_every_command_and_option_in_order(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit):
            launcher.parse_cli_args(["--help"])
        text = stdout.getvalue()
        for command in ("run", "chat", "serve", "bench"):
            self.assertIn(f"  {command} ", text)
        for option in ("--prompt", "--ctx-size", "--max-new-tokens", "--port", "--pp", "--gpu",
                       "--profiler", "--print-kernel", "--cuda-graphs", "--with-native-libraries"):
            self.assertIn(option, text)
        titles = ["commands:", "Engine Configuration", "Server (serve)", "Benchmark (bench",
                  "Hardware Configuration", "Debug and Profiling", "TornadoVM Execution Verbose",
                  "Advanced Options"]
        positions = [text.index(t) for t in titles]
        self.assertEqual(sorted(positions), positions)
        for gone in ("--echo", "LLaMA Configuration", "Command Display", "Advanced CUDA",
                     "Prefill-Decode Optimizations", "Experimental:", "--context-length",
                     "--server", "--interactive"):
            self.assertNotIn(gone, text)

    def test_every_option_shares_a_line_with_its_description_at_80_columns(self):
        previous = os.environ.get("COLUMNS")
        os.environ["COLUMNS"] = "80"
        try:
            stdout = io.StringIO()
            with redirect_stdout(stdout), self.assertRaises(SystemExit):
                launcher.parse_cli_args(["--help"])
            texts = [stdout.getvalue()] + [launcher.create_parser(c).format_help()
                                           for c in ("run", "chat", "serve", "bench")]
        finally:
            if previous is None:
                os.environ.pop("COLUMNS")
            else:
                os.environ["COLUMNS"] = previous
        for text in texts:
            for line in text.splitlines():
                self.assertLessEqual(len(line), 80, line)
                if line.startswith("  -"):
                    self.assertRegex(line, r"^  \S.*?\S {2,}\S", "description pushed to the next line: " + line)
                self.assertFalse(line.endswith("-"), "broken at a hyphen: " + line)

    def test_taskgraph_chain_needs_the_accelerator(self):
        self.reject("run", "-m", "stub.gguf", "--prompt", "hi", "--print-taskgraph-chain")
        self.assertTrue(self.parse("run", "-m", "stub.gguf", "--prompt", "hi", "--gpu",
                                   "--print-taskgraph-chain").print_taskgraph_chain)

    def test_one_spelling_per_option_in_help_but_old_spellings_still_parse(self):
        help_text = launcher.create_parser("run").format_help()
        # "-c, --ctx-size N" on Python 3.13+, "-c N, --ctx-size N" before it.
        self.assertRegex(help_text, r"-c( N)?, --ctx-size N")
        self.assertNotIn("--ctx ", help_text)
        self.assertEqual(4096, self.parse("run", "-m", "stub.gguf", "--prompt", "hi", "--ctx", "4096").max_tokens)
        self.assertIn("usage: jitllm run --model FILE [options]", help_text)
        self.reject("run", "-m", "stub.gguf", "--prompt", "hi", "--echo", "true")
        # The profiler option takes a file; the old directory spelling still parses.
        self.assertIn("--profiler-dump-file FILE", help_text)
        self.assertNotIn("--profiler-dump-dir", help_text)
        for flag in ("--profiler-dump-file", "--profiler-dump-dir"):
            self.assertEqual("p.json", self.parse("run", "-m", "stub.gguf", "--prompt", "hi",
                                                  flag, "p.json").profiler_dump_dir)

    def test_removed_fp16_flag_is_refused_with_the_migration(self):
        for prefix in (["run", "--prompt", "hi"], ["chat"], ["serve"], ["bench"], ["--prompt", "hi"]):
            with self.subTest(prefix=prefix):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    launcher.parse_cli_args(prefix + ["-m", "stub.gguf", "--fp16-kv-cache"])
                self.assertIn("--fp32-kv-cache", stderr.getvalue())
                self.assertIn("default", stderr.getvalue())

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
            self.assertEqual(command == "serve", "--port N" in help_text)
            self.assertEqual(command == "bench", "--pp N[,N...]" in help_text)
            self.assertIn("-h, --help", help_text)
            self.assertIn("jitllm --help lists every command", " ".join(help_text.split()))
            self.assertEqual(command == "run", "--prompt " in help_text)
