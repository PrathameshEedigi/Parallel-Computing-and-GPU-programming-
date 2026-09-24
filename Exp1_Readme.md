# High-Performance Matrix Multiplication: Sequential vs. OpenMP

This repository evaluates and compares the performance of large-scale **4000 × 4000 Matrix Multiplication** using two computing paradigms:

1. **Sequential Execution** (Single-threaded CPU baseline)


2. **OpenMP Parallel Execution** (Shared-memory multi-threaded parallelism)



---

## 1. Overview of Implementations

### Problem Setup

* **Matrix Dimensions:** $4000 \times 4000$ dense matrices ($A, B, C$)


* **Initialization:** All elements of matrices $A$ and $B$ are set to `1.0`

* **Arithmetic Complexity:** $O(N^3) = 4000^3 = 6.4 \times 10^{10}$ floating-point operations
* **Expected Verification:** Every cell in matrix $C$ is the dot product of 4000 ones:

$$\sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$




---

### What is Sequential Matrix Multiplication?

The sequential approach performs standard row-by-column matrix multiplication on a single CPU thread. It uses three nested loops (`i`, `j`, `k`):

* The outer loop runs row by row.


* The middle loop traverses column by column.


* The inner loop computes the accumulation dot-product.



Because it is confined to a single core, it has no parallel coordination or synchronization overhead, serving as the benchmark baseline.

### What is OpenMP Matrix Multiplication?

OpenMP (Open Multi-Processing) is an API based on compiler directives (`#pragma omp`) for shared-memory multiprocessing in C/C++.

* It creates a team of worker threads running concurrently across available CPU cores.


* Using `#pragma omp parallel for private(j, k)`, the iterations of the outer loop (`i`) are distributed among the active threads.


* All threads read from shared input matrices $A$ and $B$, and compute distinct rows of the output matrix $C$ directly in shared memory without communication overhead or race conditions.



---

## 2. Key Differences

| Feature | Sequential CPU Baseline | OpenMP Shared-Memory Parallel |
| --- | --- | --- |
| **Execution Model** | Single process, single thread

 | Single process, multi-threaded fork-join

 |
| **Hardware Utilization** | 1 logical core | Multiple CPU cores/threads concurrently (8 threads)

 |
| **Memory Model** | Single address space

 | Shared address space across all threads

 |
| **Work Partitioning** | None (sequential execution of all $N$ rows)

 | Loop iterations (`i = 0` to `N-1`) partitioned across threads

 |
| **Coordination Overhead** | None

 | Thread creation, dynamic scheduling, and barrier synchronization

 |
| **Complexity** | Simple, reference implementation

 | Requires thread safety checks (making loop indices `j, k` private)

 |

---

## 3. Experimental Results & Performance Comparison

The benchmarks were executed in an **Ubuntu (WSL2)** environment on a multi-core CPU using GCC with `-O2` optimization.

### Performance Metrics

$$\text{Speedup} = \frac{\text{Sequential Execution Time}}{\text{Parallel Execution Time}}$$

$$\text{Parallel Efficiency} = \left( \frac{\text{Speedup}}{\text{Number of Threads}} \right) \times 100\%$$

| Implementation | Threads Used | Execution Time (s) | Speedup | Verification ($C[0][0]$) |
| --- | --- | --- | --- | --- |
| **Sequential** | 1 | **266.48 s** | **1.00×** (Baseline) | `4000.00`<br> |
| **OpenMP** | 8 | **41.02 s** | **6.50×** | `4000.00`<br> |

* **Speedup Achieved:** $\frac{266.477234\,\text{s}}{41.021555\,\text{s}} \approx \mathbf{6.50\times}$
* **Parallel Efficiency (8 threads):** $\left(\frac{6.50}{8}\right) \times 100\% \approx \mathbf{81.2\%}$

### Observations

1. **Significant Runtime Reduction:** Parallelizing loop iterations across 8 CPU threads reduced runtime from ~4.44 minutes down to ~41 seconds.


2. **Sub-linear Speedup (6.50× vs. 8.0× theoretical):** Near-linear scaling is hindered by memory bus bandwidth saturation (all 8 cores fetch heavy chunks of matrix $B$ simultaneously), cache evictions, and OS thread scheduling.
3. **Correctness Preserved:** Both implementations produced the identical check value `C[0][0] = 4000.00`, confirming arithmetic correctness.
