"""Hand-written paragraphs for the synthetic test papers.

Paper A is about prefetching.  Paper B is about on-chip networks but re-uses
A's material in three deliberately different ways so each detector layer has
exactly one thing to find:

  * B_COPIED     - A_COPIED verbatim              -> verbatim run, `exact`
  * B_EDITED     - A_EDITED with two words changed -> verbatim run, `near`
  * B_PARAPHRASE - A_PARAPHRASE rewritten           -> semantic only
"""

A_ABSTRACT = (
    "Abstract. Hardware prefetchers hide memory latency by predicting which cache "
    "lines a program will touch next. This paper proposes a stride-aware prefetcher "
    "that adapts its aggressiveness to observed bandwidth headroom. Across the SPEC "
    "CPU2017 suite it improves performance by 14 percent over a next-line baseline."
)

A_INTRO = (
    "The gap between processor and memory speed remains one of the defining "
    "constraints of modern computer architecture. Caches reduce the average access "
    "time but cannot eliminate compulsory misses, which prefetching targets directly. "
    "An effective prefetcher must be both accurate and timely, issuing requests early "
    "enough to cover the miss latency without evicting useful data."
)

A_COPIED = (
    "We observe that prefetch accuracy alone is a poor proxy for end-to-end benefit. "
    "A prefetcher that is accurate but late still stalls the pipeline, while one that "
    "is early but pollutes the cache wastes bandwidth that other cores need. Our "
    "design therefore tracks both the timeliness and the pollution of each prefetch "
    "stream and throttles streams whose net contribution turns negative."
)

A_EDITED = (
    "The stride detector tracks the difference between consecutive addresses issued "
    "by each load instruction and confirms a pattern after two matching deltas. Once "
    "confirmed, the detector issues prefetches for the next several strides ahead of "
    "the demand stream, with the distance chosen by the throttling controller."
)

A_PARAPHRASE = (
    "Our evaluation uses a cycle-accurate simulator configured to model an eight-core "
    "out-of-order processor with a shared last-level cache. We run the SPEC CPU2017 "
    "benchmarks to completion after a warm-up period of one billion instructions. "
    "Results are reported as speedup relative to a baseline without prefetching."
)

A_RESULTS = (
    "The proposed prefetcher improves geometric mean performance by 14 percent, with "
    "the largest gains on memory-bound workloads such as lbm and mcf. Bandwidth "
    "consumption grows by only 6 percent because the throttling controller suppresses "
    "streams that would otherwise saturate the memory channels."
)

A_CONCLUSION = (
    "We presented a stride-aware prefetcher whose aggressiveness follows the "
    "available bandwidth headroom. Future work will extend the controller to "
    "coordinate multiple prefetchers that share a memory subsystem."
)

A_REFERENCES = [
    "[1] J. Doe and R. Roe, A survey of hardware prefetching, ACM Computing Surveys, 2019.",
    "[2] A. Smith, Timeliness-aware prefetch throttling, in Proc. ISCA, 2021.",
    "[3] K. Lee et al., Bandwidth-adaptive stream prefetching, in Proc. MICRO, 2020.",
]

B_ABSTRACT = (
    "Abstract. Networks on chip connect the cores, caches and memory controllers of a "
    "many-core processor. We propose an adaptive routing scheme that steers packets "
    "away from congested links using locally observed buffer occupancy, improving "
    "saturation throughput by 22 percent on synthetic traffic."
)

B_INTRO = (
    "As core counts grow, the on-chip interconnect increasingly determines the "
    "latency of coherence traffic and the effective bandwidth of the shared cache. "
    "Deterministic dimension-order routing is simple but concentrates load on a few "
    "channels under adversarial patterns, motivating adaptive alternatives."
)

B_COPIED = A_COPIED

B_EDITED = (
    "The stride detector tracks the difference between successive addresses issued "
    "by each load instruction and confirms a pattern after three matching deltas. Once "
    "confirmed, the detector issues prefetches for the next several strides ahead of "
    "the demand stream, with the distance chosen by the throttling controller."
)

B_PARAPHRASE = (
    "We evaluate on a cycle-level simulator that models an out-of-order CPU with "
    "eight cores sharing a last-level cache. Each SPEC CPU2017 workload is executed "
    "after warming up for one billion instructions. We present performance as "
    "speedup over a system with prefetching disabled."
)

B_RESULTS = (
    "Under uniform random traffic the adaptive router matches the baseline until "
    "40 percent injection rate and then sustains 22 percent higher throughput. For "
    "transpose and bit-reversal patterns the improvement is larger because the "
    "baseline saturates early on the diagonal channels."
)

B_CONCLUSION = (
    "We described a congestion-aware router that needs only local buffer state. "
    "Extending the scheme to hierarchical topologies is left for future work."
)

B_REFERENCES = [
    "[1] W. Dally and B. Towles, Principles and Practices of Interconnection Networks, 2004.",
    "[2] J. Doe and R. Roe, A survey of hardware prefetching, ACM Computing Surveys, 2019.",
]

C_PARAGRAPHS = [
    "Abstract. This paper studies the energy efficiency of near-threshold voltage "
    "operation in GPU streaming multiprocessors and proposes a per-cluster voltage "
    "controller guided by occupancy counters.",
    "Graphics processors execute thousands of threads concurrently, and their energy "
    "consumption is dominated by datapath switching activity rather than by control "
    "logic. Lowering the supply voltage reduces dynamic energy quadratically but "
    "increases delay variation across the die.",
    "Our controller samples the warp occupancy of each streaming multiprocessor every "
    "ten microseconds and selects the lowest voltage that keeps the observed "
    "instruction throughput within a configurable slack of the nominal point.",
    "Across a set of compute and graphics kernels the controller reduces energy per "
    "instruction by 31 percent while keeping performance loss under 4 percent.",
]
