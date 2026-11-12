---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
The exact command you ran, for example:
```bash
./jitllm run --gpu --verbose -m Llama-3.2-1B-Instruct-Q8_0.gguf --prompt "..."
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Output**
The full output of the command with `--verbose`, including any stack trace.

**Environment (please complete the following information):**
 - OS: [e.g. Ubuntu 24.04, macOS 15.5, Windows 11]
 - GPU and driver: [e.g. RTX 4090, NVIDIA driver 570.133; Apple M4 Pro]
 - JDK: [output of `java -version`]
 - TornadoVM: [contents of `$TORNADOVM_HOME/etc/tornado.release`]
 - TornadoVM backend: [contents of `$TORNADOVM_HOME/etc/tornado.backend`, and `--opencl`/`--cuda`/`--metal` if you forced one]
 - jitllm version: [e.g. 2.0.0-jdk22plus, or the commit you built]
 - Model file and quantization: [e.g. Llama-3.2-1B-Instruct-Q8_0.gguf, Q8_0]

**Additional context**
Add any other context about the problem here.
