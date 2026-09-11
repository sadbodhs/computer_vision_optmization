# Use cases — who this is for, and which pipeline fits

prev: [Overview](../README.md) · next: [Methodology](methodology.md)

The [decision guide](../README.md#decision-guide) answers *"which flow is fastest
for X?"*. This page works the other way round: start from **what you are
building**, and find which constraint actually binds you — because that, not the
benchmark table, decides the pipeline.

---

## The one idea to take from this page

Every flow here keeps up with a 30 fps camera. **A2 at 1.23 ms and D at 74 ms are
both "fast enough" for a frame that arrives every 33.3 ms** — so throughput and
latency numbers alone cannot choose for you.

What separates the use cases is **what the number is spent on**:

| Binding constraint | What you are really buying | Question that decides it |
|---|---|---|
| **Closed loop** | Reaction time of a machine | Does something *move* because of the detection? |
| **Stream density** | Cameras per GPU — i.e. cost per camera | Is anyone waiting for the answer, or is it going into a database? |
| **Wall clock** | Hours to finish a corpus | Is the input a live feed at all? |
| **Engineering budget** | Time-to-ship and who maintains it | How many people will own this in two years? |

Pick your row first. The flow follows from it, and in three of the four rows the
lowest-latency pipeline is *not* the right answer.

---

## The map

| Use case | Binds on | Streams | Suggested flow | Why |
|---|---|---|---|---|
| **Mobile robotics** (AMR, drone, inspection bot) | closed loop | 1–4 | **A2** | in-process, no server to babysit, no network in the loop |
| **Autonomous driving / ADAS** | closed loop | 4–12 | **A2** | as above, plus determinism — see the caveat below |
| **Pick-and-place / robot arm vision** | closed loop | 1–2 | **A2** | cycle time is the product; 1.23 ms leaves the budget to the actuator |
| **Sports & broadcast tracking** | closed loop (soft) | 1–8 | **A2** or **B2** | live overlay must land within a frame; B2 if ops want the server |
| **AR / assistive / interactive** | closed loop | 1–2 | **A2** | motion-to-photon is the metric, and it is unforgiving |
| **Surveillance / VMS, site-scale** | stream density | 16–64 | **B2** | Triton ops (hot reload, metrics) + ≈A2 latency |
| **City traffic / ANPR** | stream density | 64+ | **D** | nobody is waiting; batching buys 37% cheaper GPU per frame |
| **Retail analytics** (footfall, queue, shelf) | stream density | 8–64 | **B2** or **D** | latency is irrelevant; pick by team, not by ms |
| **Manufacturing — line inspection** | closed loop | 1–8 | **A2** | reject actuator fires downstream; a late verdict is a scrapped part |
| **Manufacturing — process monitoring** | stream density | 8–32 | **B2** | dashboards, not actuators |
| **Video archive search / re-indexing** | wall clock | N/A | **D** | throughput is the only axis; 1665–1816 fps |
| **Dataset labelling / model eval** | wall clock | N/A | **D** | same, and latency is meaningless offline |
| **Multi-model production serving** | stream density | any | **D** | Triton's scheduler has no hand-rolled equivalent |
| **Python-only team, any of the above** | engineering | any | **C2** | 1.69 ms — within 0.4 ms of C++, pure-Python client |
| **Edge appliance / OEM product** | engineering | 1–16 | **E** DeepStream | zero custom code, NVIDIA-supported, config not C++ |

---

## The four constraints, in detail

### 1. Closed loop — something moves because of the detection

Robotics, ADAS, pick-and-place, AR, any actuator. Here latency is not a quality
metric, it is a **safety and correctness** metric: the world has moved by the
time you answer, and how far it moved is your error.

The trap is that people optimise the wrong term. Inference is one stage in a loop
that also contains sensor exposure, readout, decode, your control logic and the
actuator itself. **A2's 1.23 ms and B2's 1.28 ms differ by 0.05 ms** — against a
camera that takes tens of milliseconds to expose and deliver a frame, that
difference is invisible. Measure the whole loop before optimising this stage.

What *does* matter here, and is the real argument for A2:

- **No network in the loop.** B2's 1.28 ms is a median. A server introduces a
  tail — GC pauses, scheduler decisions, a noisy neighbour — and closed loops are
  judged on p99, not p50.
- **No batching, ever.** Every batching result in this study
  ([Batching](batching.md)) is a reason *not* to batch in a control loop. D's
  74 ms wait is two and a half frames of the world moving.
- **In-process control.** You own the CUDA stream, so you can order work against
  the rest of your robot's compute.

> **If something moves because of the detection: A2. Do not batch. Measure p99,
> not p50, and measure the whole loop.**

### 2. Stream density — cameras per GPU is the budget

Surveillance, retail, traffic, process monitoring. Nobody is waiting for the
answer; it lands in a database, a dashboard or an alert queue. Half a second of
latency on a shoplifting alert is not a defect.

So the question stops being "how fast" and becomes **"how many cameras before I
buy a second GPU"** — the answer is the cost of the deployment.

Two things dominate that number, and neither is the model:

- **Transport.** [Transport](transport.md) is the single biggest avoidable cost
  in the study: 3.6× from system shared memory on CPU paths, 3× from CUDA IPC on
  GPU paths. A naive gRPC client (B1, 496 fps) costs you **more than half your
  cameras** against B2's 1131 fps, with an identical server behind it.
- **Decode.** Capacity mode replays pre-made tensors, so the fps figures here are
  the *inference and plumbing* ceiling. Real cameras must be decoded first, and
  NVDEC capacity is a separate budget this study does not measure — check it
  before you size a box from these numbers.

If you are past ~64 streams and can tolerate latency, move to **D**: batching is
**37% cheaper GPU time per frame**, which is 37% more cameras on the same card.
That is the one place the batching trade is unambiguously worth taking.

Also relevant here: **INT8** is +32.8% throughput for −1.55 mAP points
([Precision](precision.md)) — a third more cameras per GPU, and it beats
downgrading to a smaller model for the same speed.

> **If the answer goes into a database: B2 up to a few dozen streams, D beyond
> that. Fix transport before you buy hardware.**

### 3. Wall clock — there is no camera

Archive re-indexing, retro-search after an incident, dataset labelling, model
evaluation. The input is a file, the metric is hours-to-finish, and latency has
no meaning at all.

This is **D**, without qualification. Every objection to batching disappears when
nothing is waiting: 1665 fps single-model, 1816 fps with three models sharing the
GPU, at 0.61 ms of GPU service per frame against the batch-1 engine's 0.97 ms.

The one thing to get right is that **dynamic batching needs async, in-flight
clients**. A synchronous client pays the queue window and never collects the
benefit — it is the fifth of the [six lessons](../README.md#six-things-wed-tell-ourselves-at-the-start),
and the easiest to get wrong by accident.

> **Offline means D. Async client, batch-8, and INT8 if the accuracy cost is
> acceptable.**

### 4. Engineering budget — who owns this in two years

Sometimes the binding constraint is not the GPU. Two honest cases:

**Python-only team → C2.** The interesting result is how little it costs:
**1.69 ms, 1038 fps**, within 0.4 ms of the C++ pipeline, using a pure-numpy
client and system shared memory. Python is not the problem — *transport* is the
problem, and C1's 6.4 ms / 225 fps is what Python looks like when you get
transport wrong. If your team is Python and the work is not a control loop, C2 is
a legitimate production answer, not a compromise.

**Shipping a product, small team → DeepStream (E).** 1.50 ms per frame and zero
custom code. You write configuration rather than CUDA. You get NVDEC→infer
integration, NVIDIA support, and a pipeline someone else maintains. The price is
that the abstraction owns your architecture: when you need something the
GStreamer graph does not offer, you are writing a plugin, and the per-stream
ceiling here was never reached because the *source* bound it at 45 fps
([DeepStream](deepstream.md)).

> **Count the people who will maintain this, not just the milliseconds.**

---

## Where this study does *not* transfer

Being explicit about this matters more than the mapping above.

| Situation | What holds | What does not |
|---|---|---|
| **Jetson / Orin edge devices** | The *ordering* — transport dominates, plumbing costs more than the model, batching trades latency for throughput | Every absolute number. Unified memory changes the transport picture; there is no PCIe hop to avoid |
| **Multi-GPU or cloud scale-out** | Per-GPU pipeline choice | Scheduling, placement and routing — untouched here; see [Roadmap](roadmap.md) |
| **Segmentation, pose, tracking, VLMs** | The plumbing analysis | The engine numbers, and possibly the conclusion: a heavier model shifts the balance back toward the GPU |
| **Detect-and-track pipelines** | Everything about the detection stage | The frame economics — tracking between keyframes can cut inference load by an order of magnitude, which beats any choice on this page |
| **Accuracy-critical work** (medical, metrology) | Nothing on this page | Start at [Accuracy](accuracy.md) instead. A 1.25% mAP resize bug survived every one of these flows undetected |

That last row deserves its own sentence. If being *right* matters more than being
fast, the pipeline choice is secondary and the preprocessing audit is primary —
this study shipped a nearest-neighbour resize defect across every flow, and no
throughput number revealed it.

---

## If you only answer one question

> **Does something move because of the detection?**

Yes → **A2**, no batching, measure p99.
No → **B2** for tens of streams, **D** for hundreds or for offline.
Neither, because you need to ship next quarter → **DeepStream**.

Everything else on this site is the evidence for those three lines.

---

prev: [Overview](../README.md) · next: [Methodology](methodology.md)
