# How to contribute

We welcome contributions!
Please follow the instructions below for your Pull Requests (PRs).

## How to submit your changes

1. **Fork** the repository in GitHub.
2. **Clone** the project.
3. **Create a new branch from the `main` branch**: 
    ```bash
    $ git checkout -b fix/my/branch 
    ```
4. **Commit your changes**:
    ```bash
    $ git add <yourFiles>
    $ git commit -a -m "My new feature/fix"
    ```
5. **Push** your work to your repository:
    ```bash
    $ git push -u myRepo feat/my/branch
    ```
6. Create a **Pull Request** (PR) to the `main` branch.
7. When you open PR, there are a few GitHub actions. One of them is the checker for the **Contributor License Agreement**, [CLA](https://cla-assistant.io/beehive-lab/jitllm), if you haven't signed before, you will be prompted with the link to sign the CLA. Use the same email as you commit email. 

Please, ensure that your changes are merged with the latest changes in the `main` branch, and the code follows the code conventions (see below).

### What's next? 

We check the PR and test it internally. Be aware we are a very small team. Thus, depending on the PR, it might take some time for us to review it since we check the PR with our regression tests and benchmarks for all backends (OpenCL, CUDA and Metal) as well as with different drivers and Operating Systems. 

## Code of Conduct 

For the PR process as well as any issues and discussions we follow this [CODE_OF_CONDUCT](https://github.com/beehive-lab/TornadoVM/blob/master/CODE_OF_CONDUCT.md).


## How is the review process?

1. We have a few GitHub actions, such as code formatter, documentation rendering and checks for the CLA (Contributor License Agreement).
2. As mentioned earlier, if you haven't signed the CLA yet, you will be redirected to the jitllm CLA webpage, where you can read and review it.
If you agree with the terms, then you will sign it.
3. After that, the jitllm team can process your PR to be able to merge it into jitllm's codebase.
4. At least two researchers/engineers from the jitllm team will review your PR.
**Expect a few comments, questions and possible changes.**
This is a totally normal process, and it tries not to introduce untested code for specific devices, better documentation, etc.
jitllm comes out of the Beehive Lab, behind 10+ years of active TornadoVM development with many different researchers and developers. Thus, we prioritize code maintainability and reproducibility, and we will work together to guarantee this as much as possible.
