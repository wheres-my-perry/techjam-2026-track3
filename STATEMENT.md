# Track 3: Implement a GPU Kernel for a Transformer Layer

> Source: [TikTok TechJam 2026 Tracks & Problem Statements](https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf#RNYvddBmXosHGbxr9jfmxYgOydd)  
> Source status: Early Bird Access; Track 3 problem statement last updated on Aug 27, 2026, at 6:25 PM.  
> Latest source update: added Appendix: Test Shapes and updated `torch_transformer_benchmark.py`.  
> Technical workshop and Q&A: Aug 28, 2026, 3:00–3:45 PM (SGT/GMT+8) — [join webinar](https://vc-my.larkoffice.com/j/484622806).

## 3.1 Background

Transformer is a widely used neural network architecture in modern AI. It is the core structure behind many natural language processing, computer vision, speech, recommendation, and large language model systems.

The main idea of Transformer is self-attention. Self-attention allows each token in a sequence to interact with other tokens directly. Compared with recurrent models, Transformer can process tokens in parallel, which makes it suitable for GPU acceleration.

Given an input sequence represented as a matrix:

$$
X \in \mathbb{R}^{N \times d}
$$

where $N$ is the sequence length and $d$ is the hidden dimension, the Transformer first projects the input into Query, Key, and Value matrices:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

The scaled dot-product attention is computed as:

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $d_k$ is the dimension of each attention head. The scaling factor $\sqrt{d_k}$ is used to prevent the dot-product values from becoming too large, which could make the softmax distribution unstable.

However, the computation of Transformer is expensive. Important operations include matrix multiplication, attention score calculation, softmax, normalization, and feed-forward layers. These operations may be limited by GPU compute throughput, memory bandwidth, cache efficiency, kernel launch overhead, and tensor core utilization.

In this competition, participants are asked to use AI-assisted methods to optimize the runtime efficiency of a Transformer structure on a given GPU model. The optimized implementation should improve performance while keeping the output numerically correct compared with the reference implementation.

Participants may consider optimization methods such as operator fusion, memory layout optimization, reduced-precision computation, tensor core usage, softmax optimization, and custom CUDA, Triton, TensorFlow, or PyTorch implementations.

The goal of this task is to explore how AI can help developers analyze Transformer workloads, identify bottlenecks, and generate more efficient implementations for specific GPU hardware.

## 3.2 Problem Statement

- Given a fixed formula of a Transformer layer, participants need to submit one or several GPU kernels that implement the layer and pass the given test cases.
- The test cases will be written in PyTorch or TensorFlow. Participants may modify the layer implementation if needed, including deciding which parts of the layer should be fused into one kernel.
- The test cases will compare participants' implementations with the original PyTorch/TensorFlow implementation. Each output element passes when **either** condition is satisfied (logical OR); both comparisons are strict:
  - Relative error: `< 0.02`
  - Absolute error: `< 0.002`
- The test cases will contain different input shapes, including large/small batch sizes, sequence lengths, and dimensions. Participants may choose different implementations for different shapes by adding shape checks in the layer implementation. All combinations of input shapes will be disclosed to participants.
- The use of AI tools is encouraged so participants can implement different kernels for different input shapes within the limited time.
- Optimize and test the code on your own machine. Different optimization methods may be appropriate depending on the machine/GPU used.
- Provide a clear technical report, including details of the AI skills/tools used, to earn bonus points.

> **Repository rule:** the local Torch harness follows the problem-statement wording exactly: strict `<`, `rtol=0.02`, `atol=0.002`, combined with OR. The Aug 27 attachment used `<=`; that comparator has been normalized locally to the published rule.

### What participants need to do

1. Download the benchmark scripts. Choose either Torch or TensorFlow; one is sufficient.
2. Implement the `customized-implementation` part and optimize it as much as possible, using AI or by hand.
3. Run the script on your own machine.
4. Provide a clear technical report describing:
   - The environment (CPU, GPU, disk, etc.)
   - The optimizations performed
   - The final test results

## 3.3 Constraints & Scope

| Category | Constraints & Scope Details |
|---|---|
| In scope | AI-based code generation, GPU kernel fusion, profiling-tool usage, etc. |
| Out of scope | Production-ready deployment. |

## 3.4 Available Resources / Data

Download one of the following benchmark scripts from the [original Lark document](https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf#RNYvddBmXosHGbxr9jfmxYgOydd), then run it on your own machine:

- Torch benchmark: `torch_transformer_benchmark.py`
- TensorFlow benchmark: `tensorflow_transformer_benchmark.py`

## 3.5 Deliverables

### 1. Written Project Description (via Devpost)

Provide a clear written description of the project that includes:

- How the solution addresses the problem statement
- Development tools used (for example, VS Code, Colab, or Jupyter)
- APIs used (for example, OpenAI GPT-4o or Google Maps API)
- Libraries and frameworks used (for example, Hugging Face Transformers, PyTorch, scikit-learn, or pandas)
- Datasets and assets used (for example, the Google Local Reviews dataset or manually labelled data)

### 2. Public Code/GitHub Repository

Project repository: [wheres-my-perry/techjam-2026-track3](https://github.com/wheres-my-perry/techjam-2026-track3).

Submit a link to a public code/GitHub repository containing:

- Well-structured, commented code covering all components of the solution
- A `README` that includes:
  - Project overview
  - Setup and installation instructions
  - Steps to reproduce the results
  - A brief reflection on the solution's limitations and what would be improved with more time
  - Team member contributions, if applicable

### 3. Demo Video

Submit a short video that:

- Demonstrates the solution working end-to-end (for example, inference results, a dashboard, or model predictions)
- Is uploaded to YouTube with public visibility
- Is linked in the Devpost description
- Does not include third-party trademarks or copyrighted content without permission

For backend/NLP tracks, if a front-end interface is not applicable, a walkthrough video showing API usage, inference examples, or result analysis is accepted.

## 3.6 Judging Criteria

| Criterion | Definition | Weight |
|---|---|---:|
| Technical Execution | The solution demonstrates strong engineering fundamentals, such as well-structured code, thoughtful architecture, and effective use of APIs or models. The demo runs reliably, and the technical complexity reflects deliberate, capable decision-making. | 35% |
| Innovation & Problem Insight | The project demonstrates originality in both idea and approach. It stands out for the sharpness of its problem understanding—how clearly the team has framed the challenge, why it matters, and how directly the solution addresses it. | 20% |
| Impact & Relevance | The project has clear potential to deliver value to real users or stakeholders, with meaningful reach, tangible benefit, and relevance that goes beyond solving the hackathon prompt alone. | 20% |
| Feasibility & Practicality | The solution is realistic and buildable beyond a prototype. The approach is technically and operationally sustainable: resource usage is proportionate, the architecture holds under real-world conditions, and the implementation is grounded rather than speculative. | 15% |
| Presentation & Communication | **Final event only.** The team communicates its work clearly. The pitch tells a coherent story from problem to solution to potential, and the team responds to questions with depth, demonstrating genuine understanding of its project. | 10% |

## 3.7 Appendix

### Test shapes

| # | Batch Size | QKV Dim | Heads | Seq Len | Layers | Causal | FFN Dim |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 64 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 2 | 1 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 3 | 4 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 4 | 16 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 5 | 128 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 6 | 10000 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 7 | 64 | 32 | 4 | 128 | 4 | TRUE | 32 |
| 8 | 64 | 1024 | 4 | 128 | 4 | TRUE | 1024 |
| 9 | 64 | 128 | 1 | 128 | 4 | TRUE | 128 |
| 10 | 64 | 128 | 2 | 128 | 4 | TRUE | 128 |
| 11 | 64 | 128 | 16 | 128 | 4 | TRUE | 128 |
| 12 | 64 | 128 | 4 | 32 | 4 | TRUE | 128 |
| 13 | 64 | 128 | 4 | 1024 | 4 | TRUE | 128 |
| 14 | 32 | 1024 | 16 | 100000 | 2 | TRUE | 1024 |
