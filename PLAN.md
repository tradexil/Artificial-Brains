# ARTIFICIAL BRAINS — COMPREHENSIVE PLAN

---

## 0. DESIGN PRINCIPLES

1. **Biology-first, then optimize.** Every design choice maps to a real neural mechanism. We deviate from biology only when it makes engineering sense AND we understand what we're losing.
2. **Sparse by default.** At any tick, <5% of neurons in a region fire. Sparsity is not a bug — it's how patterns stay distinguishable and computation stays fast.
3. **No training loop.** This is not a neural network. There's no loss function, no backprop, no optimizer. Learning happens through Hebbian co-activation, pruning, and consolidation — continuously and locally.
4. **Rust does physics, Python does policy.** Rust handles neuron firing, synapse propagation, and timing (the "physics"). Python handles learning rules, trace management, memory consolidation, input/output pipelines (the "policy").
5. **Everything is a trace.** Concepts, memories, percepts, words, actions — all represented as distributed activation patterns across regions. There is no other representation.

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 Four-Level Hierarchy

```
LEVEL 1 — NEURONS
  Fixed units, belong to exactly one region.
  Have potential, threshold, activation, refractory period.
  Two types: excitatory (~80%) and inhibitory (~20%).

LEVEL 2 — SYNAPSES
  Directed connections: neuron → neuron.
  Global pool. Weight, delay, plasticity.
  Excitatory neurons make positive-weight synapses only.
  Inhibitory neurons make negative-weight synapses only.
  (Dale's Law — matches biology, simplifies dynamics.)

LEVEL 3 — BINDINGS
  Pattern → pattern links ACROSS regions.
  Not neuron connections — higher-level "this pattern in region A
  co-occurs with that pattern in region B."
  Weight, fire count, time_delta (temporal offset).

LEVEL 4 — TRACES
  A concept = a set of neuron IDs across multiple regions.
  Traces don't contain synapses — they reference neurons.
  Synapses between trace-neurons exist in the global pool.
  Traces also reference bindings that define cross-region unity.
```

Why traces reference neurons, not synapses:
- A trace says WHAT fires (which neurons in which regions).
- The synapse pool says HOW STRONGLY they're connected (weights).
- This avoids duplication and lets synapses be shared across traces.
- When trace "hot" fires, its 39 neurons activate, and signal propagates through whatever synapses those neurons have — including synapses to neurons belonging to OTHER traces (which is how association works).

### 1.2 Region Map

```
REGION               NEURON RANGE        COUNT     ROLE
─────────────────────────────────────────────────────────────────
sensory              0       -  9,999    10,000    touch/pain/temp/pressure
visual               10,000  - 29,999    20,000    image processing
audio                30,000  - 44,999    15,000    sound processing
memory_short         45,000  - 54,999    10,000    working memory (capacity-limited)
memory_long          55,000  - 69,999    15,000    stored patterns
emotion              70,000  - 79,999    10,000    polarity/urgency/priority
attention            80,000  - 84,999     5,000    gain control filter
pattern              85,000  - 94,999    10,000    cross-modal pattern finder
integration          95,000  - 104,999   10,000    unified experience builder
language             105,000 - 119,999   15,000    symbols/logic/reasoning
executive            120,000 - 129,999   10,000    planning/conflict resolution
motor                130,000 - 139,999   10,000    action signals
speech               140,000 - 149,999   10,000    sound production
numbers              150,000 - 151,999    2,000    hardwired digit/magnitude
─────────────────────────────────────────────────────────────────
TOTAL                                   152,000
```

### 1.3 Signal Flow (Per Tick)

```
INPUTS (sensory, visual, audio)
         │
         ▼
    ┌─────────┐
    │ATTENTION │ ◄── gain control, not blocking
    │(multiply)│     attended signals amplified 2-5x
    └────┬────┘     unattended signals pass at 0.1-0.3x
         │
         ▼
  ┌──────────────┐
  │   PATTERN    │ ◄── compares activation against trace library
  │ RECOGNITION  │     finds matches, flags novelty on mismatches
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ INTEGRATION  │ ◄── merges multi-region patterns via temporal binding
  │              │     "all of this happened together = one experience"
  └──┬───┬───┬───┘
     │   │   │
     ▼   ▼   ▼
  MEMORY  EMOTION  LANGUAGE
     │      │        │
     └──────┼────────┘
            ▼
      ┌───────────┐
      │ EXECUTIVE  │ ◄── resolves conflicts, suppresses impulse
      └─────┬─────┘     plans multi-step sequences
            │
        ┌───┴───┐
        ▼       ▼
     MOTOR    SPEECH
```

**Critical routing rules:**
- Sensory/Visual/Audio NEVER go straight to Executive. Always through Pattern → Integration.
- Emotion can SHORT-CIRCUIT to Motor (reflexes: hand on hot stove → withdraw BEFORE conscious processing). This bypass has dedicated fast synapses.
- Language ↔ Executive form a LOOP for internal reasoning (thinking without speaking).
- Memory_long → Visual is the IMAGINATION pathway (reverse-firing visual from memory).

### 1.4 Internal Loops

These loops are what separates reactive stimulus-response from actual thinking:

```
REASONING LOOP:
  Language → Executive → Language → Executive → ... → resolution
  Internal deliberation. No motor output until executive resolves.
  This is "thinking through a problem."
  Working memory holds the intermediate states.

IMAGINATION LOOP:
  Memory_long → Visual (reverse activation) → Pattern → Memory
  Mental visualization. Visual region activated internally, not from input.
  This is "picturing something in your mind."

EMOTIONAL EVALUATION LOOP:
  Pattern → Emotion → Executive → Emotion → Executive → decision
  "Is this dangerous? How dangerous? Should I act?"
  Multiple iterations refine the emotional assessment.

REHEARSAL LOOP:
  Language → Memory_short → Language → Memory_short → ...
  Internal repetition that strengthens memory traces.
  This is "repeating something to remember it."
```

---

## 2. CORE DATA STRUCTURES

### 2.1 Neuron

```rust
struct Neuron {
    // IDENTITY (immutable after creation)
    id: u32,
    region: RegionId,           // which region this neuron belongs to
    neuron_type: NeuronType,    // Excitatory (80%) or Inhibitory (20%)

    // DYNAMICS (change every tick)
    potential: f32,             // accumulated input, rises toward threshold
    activation: f32,            // output signal (0.0 = silent, 1.0 = just fired)
    refractory: u16,            // ticks remaining until can fire again

    // PARAMETERS (fixed per region, can be modulated)
    threshold: f32,             // potential must exceed this to fire
    leak_rate: f32,             // potential *= leak_rate each tick (< 1.0)
    refractory_period: u16,     // how many ticks after firing before can fire again
}
```

**Firing rule (Leaky Integrate-and-Fire):**
```
each tick:
    if refractory > 0:
        refractory -= 1
        activation *= 0.5           // rapid decay after fire
        return

    potential += sum(incoming_synapse.weight * source.activation)
    potential *= leak_rate           // leak toward zero

    if potential >= threshold:
        activation = 1.0
        potential = 0.0              // reset
        refractory = refractory_period
    else:
        activation *= 0.8           // decay if didn't fire
```

**Why inhibitory neurons matter:**
- ATTENTION: inhibitory neurons suppress unattended signals (gain control)
- EXECUTIVE: inhibitory neurons suppress competing action plans
- PATTERN: inhibitory neurons create winner-take-all competition (only strongest pattern survives)
- MEMORY_SHORT: inhibitory neurons enforce capacity limit (~7 active traces)

**Sparsity enforcement:**
Within each region, lateral inhibition (inhibitory neurons connected to nearby excitatory neurons) ensures only ~2-5% of neurons fire per tick. This is NOT a hard cap — it's emergent from the inhibitory network. But if sparsity drifts above 10%, we add a global inhibitory gain boost as a safety valve.

### 2.2 Synapse

```rust
struct Synapse {
    from: u32,              // source neuron ID
    to: u32,                // target neuron ID
    weight: f32,            // strength (always positive; sign determined by neuron_type)
                            // excitatory neuron → positive effect on target
                            // inhibitory neuron → negative effect on target
    delay: u8,              // propagation delay in ticks (1-10)
                            // nearby neurons: 1-2 ticks
                            // cross-region: 3-8 ticks
                            // sensory→motor reflex: 2-3 ticks (fast path)
    plasticity: f32,        // 0.0 = fixed (hardwired reflex)
                            // 1.0 = fully plastic (learning synapse)
                            // numbers region synapses: plasticity ≈ 0.0
}
```

**Storage: Compressed Sparse Row (CSR) format for read-heavy access:**
```rust
struct SynapsePool {
    // Core CSR structure (fast traversal)
    targets: Vec<u32>,      // all target neuron IDs, contiguous
    weights: Vec<f32>,      // corresponding weights
    delays: Vec<u8>,        // corresponding delays
    plasticity: Vec<f32>,   // corresponding plasticity
    offsets: Vec<u32>,      // offsets[i] = index where neuron i's synapses start
                            // neuron i's synapses: targets[offsets[i]..offsets[i+1]]

    // Pending modifications buffer (applied during maintenance phase)
    pending_updates: Vec<SynapseUpdate>,    // weight changes
    pending_creates: Vec<SynapseCreate>,    // new synapses
    pending_prunes: Vec<(u32, u32)>,        // (from, to) pairs to remove

    // Stats
    total_count: u64,
    create_count: u64,
    prune_count: u64,
}
```

Why CSR over adjacency list:
- Cache-friendly contiguous memory (critical for 5M+ synapses)
- Fast iteration over a neuron's outgoing connections
- Tradeoff: modifications are batched and applied during rebuild
- Rebuild frequency: every 1000 ticks (or when pending buffer > 10% of pool)

### 2.3 Binding

```rust
struct Binding {
    id: u32,
    pattern_a: PatternRef,      // {region, neuron_ids[]}
    pattern_b: PatternRef,      // {region, neuron_ids[]}
    weight: f32,                // 0.0 - 1.0, how strongly linked
    fires: u32,                 // co-activation count
    time_delta: f32,            // temporal offset in ticks (positive = A before B)
                                // visual-audio binding for thunder: time_delta > 0
                                // (light before sound)
    last_fired: u64,            // tick of last co-activation
    confidence: f32,            // fires / opportunities (how reliable is this link)
}

struct PatternRef {
    region: RegionId,
    neurons: Vec<u32>,          // the specific neurons that form this pattern
    threshold: f32,             // what fraction must fire to count as "this pattern"
                                // 0.7 = 70% of neurons must be active
}
```

**Binding lifecycle:**
- FORMATION: When two patterns in different regions co-activate within a time window (±5 ticks) more than `binding_formation_threshold` times (default: 5), a binding is created.
- STRENGTHENING: Each co-activation increments `fires`, updates `weight` and `confidence`.
- WEAKENING: Each tick where one pattern fires but the other doesn't, `weight` decays slightly.
- DISSOLUTION: When `weight` drops below 0.05 AND `fires` < 10, binding is pruned.

**Binding competition:**
If pattern A has bindings to both B and C, and both B and C are active, both bindings fire. No winner-take-all at binding level — the competition happens at the integration region level, where the strongest combined signal wins.

### 2.4 Trace

```python
class Trace:
    # IDENTITY
    id: str                         # "trace_0047"
    label: str | None               # "hot" — attached from language exposure, None initially

    # STRUCTURE — what neurons define this concept in each region
    neurons: dict[str, list[int]]   # region_name → [neuron_ids]
                                    # NOT synapses — neurons
                                    # synapses between these neurons live in global pool

    # STRUCTURE — which bindings define cross-region unity
    binding_ids: list[int]          # binding IDs that connect this trace's patterns

    # DYNAMICS
    strength: float                 # 0.0 - 1.0, grows with repetition
    decay: float                    # 0.0 - 1.0, drops when not fired
    polarity: float                 # -1.0 to +1.0, emotional valence
    abstraction: float              # 0.0 (concrete) to 1.0 (abstract)
    novelty: float                  # drops every fire, drives attention

    # RELATIONSHIPS
    co_traces: list[str]            # conceptually related (what this IS)
    context_tags: list[str]         # situationally related (where this APPEARS)

    # BOOKKEEPING
    fire_count: int                 # total activations
    last_fired: int                 # tick of last activation
    formation_tick: int             # when this trace was born
```

**Trace activation (when is a trace considered "firing"?):**
```
For each trace:
    active_ratio = count(trace.neurons[region] that are active) / total_neurons_in_trace
    if active_ratio >= activation_threshold (default 0.6):
        trace is "firing"
        update: strength, decay, novelty, fire_count, last_fired
        propagate: co_traces get small activation bump
        propagate: context_tags get tiny activation bump
```

**Trace lookup (which traces match current activation?):**
Inverted index: for each neuron, store list of trace IDs containing it.
When neurons fire → look up candidate traces → score by overlap → threshold.
This is O(active_neurons × avg_traces_per_neuron) per tick, not O(all_traces).

### 2.5 Correction from Original Design

The original design stores `synapses: dict[region_id, list[synapse_id]]` on traces. This conflates two things:
- **Which neurons represent this concept** (the trace's identity)
- **How those neurons connect** (the synapse pool's job)

In this plan, traces store `neurons: dict[region, list[neuron_id]]`. The synapses between those neurons exist in the global synapse pool and are strengthened via Hebbian learning whenever the trace fires. This is cleaner because:
1. No duplication (synapse exists in one place).
2. Synapses are shared — neuron 10442 connects to neuron 70006 regardless of which trace activated it.
3. Trace "strength" becomes emergent — a strong trace has strong synapses between its neurons.

---

## 3. REGION SPECIFICATIONS

Each region has unique firing parameters and special behaviors.

### 3.1 Sensory Region (0 - 9,999)

```
Neurons:     10,000
Inhibitory:  15% (less inhibition — sensory signals should pass through)
Threshold:   0.3 (low — sensitive to input)
Leak rate:   0.85 (fast leak — sensory is transient)
Refractory:  2 ticks (fast recovery — continuous sensing)

SPECIAL BEHAVIOR:
- Direct input from sensory_input.py (temperature, pressure, pain values)
- Input values map to specific neuron sub-ranges:
    0-2499:     temperature (cold → hot gradient)
    2500-4999:  pressure (light → heavy)
    5000-7499:  pain (none → severe)
    7500-9999:  texture (smooth → rough, sharp, etc.)
- Has FAST PATH to motor (reflex arc): 2-tick delay synapses
  that bypass all processing — hand on hot stove = withdraw immediately
- Reflex synapses have plasticity = 0.05 (nearly hardwired)
```

### 3.2 Visual Region (10,000 - 29,999)

```
Neurons:     20,000 (largest input region)
Inhibitory:  20%
Threshold:   0.4
Leak rate:   0.9 (moderate persistence — visual afterimage)
Refractory:  3 ticks

SPECIAL BEHAVIOR:
- Hierarchical sub-regions (10k neurons is enough for basic hierarchy):
    10000-14999:  low-level (edges, colors, orientation)
    15000-19999:  mid-level (shapes, textures, contours)
    20000-24999:  high-level (objects, faces, scenes)
    25000-29999:  spatial (position, movement, depth)
- Signals flow low → mid → high → spatial (feedforward)
- BUT ALSO high → mid → low (feedback/prediction)
- Can be activated INTERNALLY by memory_long (imagination)
- When internally activated, inhibit low-level to prevent hallucination
  mixing with real input (imagination flag)
```

### 3.3 Audio Region (30,000 - 44,999)

```
Neurons:     15,000
Inhibitory:  20%
Threshold:   0.35
Leak rate:   0.88
Refractory:  2 ticks (fast — audio is temporal)

SPECIAL BEHAVIOR:
- Sub-regions:
    30000-34999:  frequency decomposition (pitch)
    35000-39999:  temporal patterns (rhythm, onset, duration)
    40000-44999:  complex (timbre, melody, speech phonemes)
- CRITICAL: audio is inherently temporal — patterns emerge over TIME
  not in a single tick. Audio pattern recognition needs a time window
  of ~50-100 ticks to identify a pattern.
- Stores recent activation in a circular buffer (last 100 ticks)
  for temporal pattern detection.
- Completely separate from visual until Integration region.
```

### 3.4 Memory Short / Working Memory (45,000 - 54,999)

```
Neurons:     10,000
Inhibitory:  25% (MORE inhibition — enforces capacity limit)
Threshold:   0.5 (higher — not everything gets into working memory)
Leak rate:   0.95 (slow leak — working memory persists while attended)
Refractory:  5 ticks (slower — working memory items are sustained)

SPECIAL BEHAVIOR:
- CAPACITY LIMIT: at most 7 ± 2 traces can be simultaneously active
  in working memory. Enforced by strong lateral inhibition.
  When an 8th trace tries to activate, it competes and the weakest
  current trace gets pushed out.
- Items in working memory have BOOSTED attention gain (2x)
- Items decay rapidly if not refreshed (rehearsal loop)
- Contents directly accessible to Executive and Language
- "What you're thinking about right now"
```

### 3.5 Memory Long (55,000 - 69,999)

```
Neurons:     15,000
Inhibitory:  20%
Threshold:   0.6 (HIGH — long-term memories don't activate easily)
Leak rate:   0.99 (very slow leak — long-term persistence)
Refractory:  8 ticks

SPECIAL BEHAVIOR:
- Pattern completion: partial activation → full pattern retrieval
  If 40%+ of a stored trace's memory_long neurons fire, the rest
  get a boost toward threshold. This IS memory recall.
- CUE-BASED retrieval: activation from other regions (language,
  sensory, emotion) triggers pattern completion.
- Transfer from memory_short happens during CONSOLIDATION CYCLES
  (idle/sleep periods, see section 6.5)
- Stores RICHER patterns than memory_short (more neurons per trace)
- Two implicit sub-systems (not separate regions, but different patterns):
    Episodic: events with context_tags (what happened and when/where)
    Semantic: facts without context (what something IS, stripped of occasion)
  Episodic → Semantic happens through repeated activation stripping
  context over time (you remember THAT fire is hot, forget WHEN you learned it)
```

### 3.6 Emotion (70,000 - 79,999)

```
Neurons:     10,000
Inhibitory:  15%
Threshold:   0.3 (LOW — emotions activate easily)
Leak rate:   0.92 (moderate — emotions linger)
Refractory:  4 ticks

SPECIAL BEHAVIOR:
- Sub-ranges for different emotional dimensions:
    70000-72499:  valence (positive ← → negative)
    72500-74999:  arousal (calm ← → excited)
    75000-77499:  urgency (can wait ← → act now)
    77500-79999:  social (safe ← → threat)
- Emotion TAGS traces with polarity:
  When a trace fires and emotion is co-active, the trace's polarity
  shifts toward current emotional state.
  polarity += learning_rate * (current_emotion_valence - polarity)
- Emotion can OVERRIDE attention:
  High-urgency emotion forces attention onto the triggering stimulus
  regardless of current focus. (Hearing your name in a crowd.)
- Emotion modulates GLOBAL thresholds (neuromodulator effect):
  High arousal → lower thresholds everywhere → more sensitive
  Low arousal → higher thresholds → less reactive
- Has SHORT-CIRCUIT to motor (panic → freeze/flee, bypasses executive)
```

### 3.7 Attention (80,000 - 84,999)

```
Neurons:     5,000 (smallest region — attention is a filter, not a processor)
Inhibitory:  40% (VERY HIGH — attention is primarily about suppression)
Threshold:   0.4
Leak rate:   0.93
Refractory:  3 ticks

SPECIAL BEHAVIOR:
- Attention is GAIN CONTROL, not gating.
  It doesn't block signals. It multiplies them.
  attended_signal = raw_signal × attention_gain
  gain range: 0.1 (suppressed) to 5.0 (hyper-focused)
- Three attentional drives compete:
    1. NOVELTY: high-novelty traces spike attention gain
    2. THREAT: emotion-flagged urgency forces attention
    3. RELEVANCE: executive sets top-down attention targets
       (you're looking for your keys → key-related patterns get boosted)
- Attention has INERTIA: once focused, takes 10-20 ticks to shift
  (cost of context switching)
- Attention outputs a gain map: one gain value per region
  that multiplies all incoming signals to that region
```

### 3.8 Pattern Recognition (85,000 - 94,999)

```
Neurons:     10,000
Inhibitory:  25%
Threshold:   0.5
Leak rate:   0.9
Refractory:  4 ticks

SPECIAL BEHAVIOR:
- Works ACROSS sensory, visual, and audio simultaneously
- Core job: compare current activation against trace library
  "Have I seen this pattern before?"
- Outputs:
    MATCH: existing trace identified → strengthen trace, reduce novelty
    PARTIAL MATCH: similar but not identical → flag for attention
    NO MATCH: novel pattern → high novelty signal → attention spike
              → candidate for new trace formation
- Uses the inverted neuron→trace index for fast matching
- Pattern PREDICTION (CRITICAL ADDITION):
    Based on current patterns + context, predict NEXT activation.
    prediction_error = |predicted - actual|
    High prediction error → novelty spike → learning signal
    Low prediction error → nothing interesting → suppress attention
    THIS IS THE MAIN DRIVER OF LEARNING.
    Not just "did I see this before" but "did I correctly predict what would happen next"
```

### 3.9 Integration (95,000 - 104,999)

```
Neurons:     10,000
Inhibitory:  20%
Threshold:   0.55
Leak rate:   0.92
Refractory:  5 ticks

SPECIAL BEHAVIOR:
- The BINDING PROBLEM solver.
  "Red ball moving left while a whistle blows" →
  Integration binds these into ONE experience, not three separate ones.
- Temporal binding window: patterns must fire within ±5 ticks
  to be bound into the same experience.
- Integration neurons fire when MULTIPLE regions are co-active:
    visual + audio + sensory → strong integration
    only visual → weak integration
  Integration strength = f(number of co-active regions)
- This region is where bindings are EVALUATED:
  When integration fires, check all active bindings.
  Matching bindings strengthen. Missing bindings weaken.
- Output feeds into Memory, Emotion, and Language simultaneously.
```

### 3.10 Language (105,000 - 119,999)

```
Neurons:     15,000
Inhibitory:  20%
Threshold:   0.5
Leak rate:   0.94 (persistent — you hold words in mind)
Refractory:  4 ticks

SPECIAL BEHAVIOR:
- Sub-regions:
    105000-109999:  symbols (words, labels, numbers-as-words)
    110000-114999:  relations (bigger-than, part-of, causes, before/after)
    115000-119999:  syntax (ordering rules, if-then structure, logic)
- Language LABELS traces: when a trace and a word co-activate,
  the trace gets a label. Before language exposure, traces are unlabeled.
- Language enables ABSTRACT traces: traces with heavy language neurons
  and sparse sensory neurons = abstract concepts (justice, time, infinity).
- Language ↔ Executive loop IS reasoning.
- Language can activate OTHER traces purely through relational logic:
  "If hot → danger" activates danger through language, not through
  actually experiencing danger. This is the power of symbolic thought.
```

### 3.11 Executive (120,000 - 129,999)

```
Neurons:     10,000
Inhibitory:  30% (high — executive is about SUPPRESSION of bad options)
Threshold:   0.6 (high — executive should be deliberate, not reactive)
Leak rate:   0.93
Refractory:  6 ticks (slowest — executive decisions take time)

SPECIAL BEHAVIOR:
- CONFLICT RESOLUTION: when multiple action traces are active,
  executive suppresses all but the strongest/best.
  "Fight or flight?" → executive evaluates, picks one, suppresses other.
- IMPULSE CONTROL: can suppress emotion-driven motor activation.
  Emotion says "punch him." Executive says "no, consequences."
  Implemented via strong inhibitory synapses from executive to motor.
- PLANNING: can activate future memory traces in sequence
  to simulate "what would happen if I..."
  This is prediction + imagination chained together.
- GOAL MAINTENANCE: holds active goal in working memory and
  biases attention toward goal-relevant stimuli.
  attention_gain[relevant_regions] += executive_bias
- DECISION OUTPUT: activates specific motor or speech traces.
```

### 3.12 Motor (130,000 - 139,999)

```
Neurons:     10,000
Inhibitory:  20%
Threshold:   0.55
Leak rate:   0.85 (fast leak — actions are discrete, not persistent)
Refractory:  3 ticks

SPECIAL BEHAVIOR:
- Receives commands from Executive (deliberate action) and
  directly from Sensory/Emotion (reflexes).
- REFLEX PATH: sensory → motor (2-3 tick delay, hardwired, low plasticity)
- DELIBERATE PATH: sensory → ... → executive → motor (20-50 tick delay)
- Motor sequences: some traces encode action SEQUENCES
  (reach → grab → pull = one motor trace)
- Output goes to motor_output.py which translates to actions.
```

### 3.13 Speech (140,000 - 149,999)

```
Neurons:     10,000
Inhibitory:  20%
Threshold:   0.5
Leak rate:   0.87
Refractory:  3 ticks

SPECIAL BEHAVIOR:
- Separate from Language. Language forms the thought, Speech produces sound.
- Input from Language (what to say) and Executive (permission to say it).
- Sub-regions:
    140000-144999:  phoneme patterns (sound units)
    145000-149999:  prosody (rhythm, stress, intonation)
- Can be suppressed by Executive (thinking without speaking).
- Output goes to speech_output.py.
```

### 3.14 Numbers (150,000 - 151,999)

```
Neurons:     2,000
Inhibitory:  10% (minimal — numbers is highly structured)
Threshold:   0.5
Leak rate:   0.96 (persistent — number concepts are stable)
Refractory:  4 ticks

SPECIAL BEHAVIOR:
- NOT randomly generated. Hardwired from birth.
- Structure encodes:
    150000-150099:  digits 0-9 (10 neurons each)
    150100-150399:  place values (ones, tens, hundreds)
    150400-150799:  magnitude sense (approximate number line)
    150800-151199:  operations (add, subtract, compare)
    151200-151599:  relational (greater, less, equal)
    151600-151999:  reserve
- All synapses in numbers region have plasticity ≈ 0.0 (hardwired truths).
- "3 is bigger than 2" is not learned. It's structure.
- Connects to Language (number words) and Pattern (counting patterns).
- Not in random trace generation (NEURONS_PER_REGION = 0 for numbers).
```

---

## 4. TICK CYCLE (THE BRAIN CLOCK)

### 4.1 Tick Definition

One tick ≈ 1ms of simulated brain time.
1000 ticks = 1 simulated second.
Target: 1000+ ticks/second real-time (Rust should achieve this easily with 152k neurons at 5% sparsity).

### 4.2 Tick Phases

Each tick has four sequential phases:

```
PHASE 1: PROPAGATE (Rust, parallel across regions)
───────────────────────────────────────────────────
For each active neuron:
    For each outgoing synapse (respecting delay):
        target.potential += source.activation * synapse.weight * attention_gain[target.region]
        (if source is inhibitory: subtract instead of add)

For each neuron:
    Apply leak: potential *= leak_rate
    Check threshold: if potential >= threshold → fire
    Update refractory counters
    Update activation (fire or decay)

Update active neuron lists (sparse tracking)

PHASE 2: EVALUATE (Rust + Python, parallel per system)
───────────────────────────────────────────────────
- Pattern Recognition: match current activation against trace library
- Binding evaluation: check which bindings co-activated
- Integration: compute binding strength for current tick
- Attention: compute new gain map
- Emotion: compute global arousal/urgency modulation
- Working memory: check capacity, evict if needed

PHASE 3: LEARN (Python, can overlap with next tick's propagate)
───────────────────────────────────────────────────
- Hebbian: detect co-active neuron pairs, queue synapse updates
- Trace updates: strength, decay, novelty, polarity adjustments
- Binding updates: weight, fires, confidence
- Prediction error: compute and distribute novelty signals
- Trace formation: if novel pattern persists > N ticks, create new trace

PHASE 4: MAINTAIN (Python, periodic — not every tick)
───────────────────────────────────────────────────
Every 100 ticks:
    - Apply queued synapse weight updates
    - Process synapse creation queue

Every 1000 ticks:
    - Rebuild CSR index if pending changes > threshold
    - Run pruning pass (see section 6.2)
    - Update trace decay values
    - Compute and log metrics

Every 10000 ticks:
    - Consolidation cycle (memory_short → memory_long transfer)
    - Binding pool cleanup
    - Trace pool cleanup (merge near-identical traces)
```

### 4.3 Neuromodulator System (Global State)

Not a separate region — a global modifier system that affects all regions:

```python
class NeuromodulatorState:
    arousal: float      # 0.0 (asleep) - 1.0 (panic)
                        # High arousal → lower ALL thresholds (more sensitive)
                        # Low arousal → higher thresholds (less reactive)

    valence: float      # -1.0 (negative) to 1.0 (positive)
                        # Affects polarity tagging on new/active traces

    focus: float        # 0.0 (scattered) to 1.0 (laser focused)
                        # High focus → attention gain range narrows
                        # (attend hard to one thing, ignore everything else)

    energy: float       # 0.0 (depleted) to 1.0 (full)
                        # Low energy → executive threshold rises (harder to plan)
                        # Low energy → triggers consolidation cycle (sleep)
```

These are updated by the Emotion region and affect every other region's parameters. This is analogous to neurotransmitter systems (dopamine, serotonin, norepinephrine, acetylcholine) in real brains.

---

## 5. LEARNING MECHANISMS

### 5.1 Hebbian Learning ("fire together, wire together")

```
Rule: if neuron A fires and neuron B fires within ±3 ticks,
      AND a synapse exists from A → B,
      THEN strengthen that synapse.

delta_weight = learning_rate × A.activation × B.activation × synapse.plasticity
synapse.weight += delta_weight
synapse.weight = clamp(synapse.weight, 0.0, 1.0)

learning_rate depends on:
    - Brain development phase (high during BLOOM, moderate during MATURE)
    - Novelty of the pattern (novel = higher learning rate)
    - Emotional arousal (high arousal = stronger memory formation)
    - Attention gain (attended signals learn faster)

ANTI-HEBBIAN: if A fires and B does NOT fire (within window),
    AND synapse exists from A → B,
    THEN slightly weaken:
    synapse.weight -= anti_rate × A.activation × (1 - B.activation) × synapse.plasticity
    This prevents everything from connecting to everything.
```

### 5.2 Pruning Lifecycle

This is the user's insight formalized into three phases:

```
PHASE 1: BLOOM (tick 0 — tick 500,000)
═══════════════════════════════════════
  Synapse creation threshold:  LOW (co-activation > 0.1 triggers new synapse)
  Synapse pruning:             NONE (no pruning during bloom)
  Net effect:                  Massive synapse growth
  Purpose:                     Cast wide net, explore all possible connections
  Expected synapse count:      5M → 15M+ (exponential growth)

  During bloom, the brain is OVER-CONNECTED. Most connections are weak
  and noisy. Pattern recognition is poor because everything activates
  everything else. This is INTENTIONAL — you need garbage to find gold.

PHASE 2: CRITICAL PRUNING (tick 500,000 — tick 2,000,000)
═══════════════════════════════════════════════════════════
  Trigger:                     synapse_count exceeds BLOOM_THRESHOLD (configurable, ~15M)
  Synapse creation threshold:  MEDIUM (co-activation > 0.3)
  Synapse pruning:             AGGRESSIVE
      Prune if: weight < 0.15  OR  (weight < 0.3 AND not fired in 50,000 ticks)
  Net effect:                  NEGATIVE (pruning >> growth)
  Expected synapse count:      15M → 5-8M

  This is the most important phase. The brain is SCULPTED by removing
  noise. Strong, frequently-used connections survive. Weak, random
  connections die. Pattern recognition improves dramatically because
  signals become clean.

  Like a sculptor: the statue was always inside the marble.
  Pruning reveals it.

PHASE 3: MATURE (tick 2,000,000+)
═════════════════════════════════
  Synapse creation threshold:  HIGH (co-activation > 0.5 sustained over 10+ ticks)
  Synapse pruning:             GENTLE
      Prune if: weight < 0.05  AND  not fired in 200,000 ticks
  Net effect:                  POSITIVE but slow (growth > pruning, barely)
  Expected synapse count:      Gradual slow climb from 5-8M upward

  The brain is now stable. New learning happens but requires stronger
  evidence than during bloom. Old unused connections slowly disappear.
  This is adult learning: slower, more deliberate, but more precise.
```

**Pruning is not just cleanup — it's a learning mechanism.** The information content of the brain INCREASES as synapses are pruned, because the remaining connections are meaningful.

### 5.3 Trace Formation

```
When does a NEW trace get created?

CONDITIONS (ALL must be met):
  1. Pattern Recognition reports NO MATCH above 0.6 threshold
  2. The novel pattern persists for > 20 consecutive ticks
  3. The pattern involves > 2 regions (cross-modal = worth remembering)
  4. novelty signal is high (prediction error was significant)
  5. Working memory has capacity (< 7 active traces)

PROCESS:
  1. Identify which neurons in which regions are part of the new pattern
  2. Create trace with those neuron assignments
  3. Set strength = 0.1 (fragile, needs reinforcement)
  4. Set novelty = 1.0 (brand new)
  5. Set decay = 1.0 (fully fresh)
  6. Set co_traces = traces that were also active during formation
  7. Set context_tags = current working memory contents
  8. Hebbian learning immediately strengthens synapses between trace neurons
  9. Register in inverted index (neuron → trace lookup)

TRACE MERGING:
  If two traces have > 80% neuron overlap and frequently co-activate,
  merge them into one trace. This is concept refinement.
  "Hot liquid" and "hot surface" might initially be separate traces,
  then merge into a general "hot" trace with context_tags differentiating.
```

### 5.4 Binding Formation

```
CONDITIONS:
  1. Two patterns in different regions co-activate within ±5 ticks
  2. This co-activation happens > 5 times
  3. No existing binding already covers this pair

PROCESS:
  1. Create binding with the two PatternRefs
  2. Set weight = 0.2 (initial)
  3. Record time_delta (avg temporal offset across the 5+ co-activations)
  4. Subsequent co-activations strengthen: weight += 0.05 × (1 - weight)
```

### 5.5 Consolidation (Memory Transfer)

```
TRIGGER: energy < 0.2 OR every 100,000 ticks (whichever comes first)
         This is analogous to sleep.

PROCESS:
  1. REPLAY: traces that fired during the awake period get re-activated
     in memory_long, in order of:
       (a) high emotional polarity (important stuff first)
       (b) high novelty (new stuff second)
       (c) high strength (frequent stuff third)
  2. For each replayed trace:
       - Strengthen its memory_long neurons and their synapses
       - If trace already exists in memory_long, reinforce existing neurons
       - If new, recruit new memory_long neurons for this trace
  3. CONTEXT STRIPPING: during consolidation, context_tags are partially
     dropped. Episodic (with context) → Semantic (without context) over
     repeated consolidations.
  4. INTERFERENCE RESOLUTION: if two traces competed during awake period
     (activated at same time, opposite polarities), weaken the loser.
  5. Clear memory_short: after consolidation, working memory items that
     were transferred to long-term get their short-term activations reduced.

DURATION: consolidation takes ~10,000 ticks
DURING CONSOLIDATION: input signals are gated (reduced attention gain to all inputs)
This is literally sleep — reduced external processing, internal memory reorganization.
```

### 5.6 Prediction Error (The Core Learning Signal)

```
THIS IS ARGUABLY THE MOST IMPORTANT MECHANISM.

The brain doesn't just react. It constantly PREDICTS the next moment.
When the prediction is wrong, that's what drives attention and learning.

PROCESS (every tick in Pattern Recognition):
  1. Based on current active traces + context + recent history:
     predicted_activation[region] = f(current_state, recent_history)
     (Simple version: weighted average of what happened after similar states before)

  2. Actual activation arrives from inputs.

  3. prediction_error = |predicted - actual| per region

  4. EFFECTS:
     error > 0.5  → SURPRISE    → attention spike, novelty boost,
                                    learning rate doubles for 50 ticks
     error > 0.8  → ALARM       → emotion arousal spike, executive activation,
                                    all thresholds drop, hypervigilance for 200 ticks
     error < 0.1  → EXPECTED    → attention drops, novelty zeroed,
                                    minimal learning (already know this)
     error 0.1-0.5 → INTERESTING → moderate attention, normal learning

  This creates a natural curriculum:
  - Brand new input → everything is surprising → massive learning
  - Familiar input → nothing surprising → minimal learning
  - Slightly novel twist on familiar → most interesting → strongest learning
    (This is why good teaching builds on what you already know.)
```

---

## 6. PARALLELIZATION STRATEGY

### 6.1 Where Parallelism Lives

```
LEVEL 1: INTER-REGION (coarse, 13+ threads)
─────────────────────────────────────────────
During Phase 1 (Propagate), each region's neurons can be updated independently.
No region reads another region's neurons during this phase — they read
the PREVIOUS tick's activations (double-buffered).

Implementation: one thread per region, barrier sync at end of Phase 1.
With 13+ regions, this maps well to modern CPUs (8-16 cores).

LEVEL 2: INTRA-REGION (fine, SIMD/vectorized)
──────────────────────────────────────────────
Within a region, all neurons can compute their new potential in parallel.
Each neuron reads (source activations + synapse weights) and writes
(own potential, own activation).
No neuron reads another neuron's CURRENT-tick state.

Implementation: SIMD vectorization via Rust's auto-vectorization or
explicit SIMD intrinsics. Process neurons in chunks of 8 (f32 × 8 = 256-bit AVX2).

LEVEL 3: SPARSE OPTIMIZATION (most important)
──────────────────────────────────────────────
At any tick, <5% of neurons are active = ~7,600 out of 152,000.
DON'T iterate all neurons. Iterate only active ones.

Maintain per-region active_list: Vec<u32> of neuron IDs with potential > 0.
Propagation only needs to:
  (a) iterate active neurons' outgoing synapses (apply signal to targets)
  (b) iterate neurons that RECEIVED signal this tick (update potential)
  (c) apply leak to neurons with non-zero potential

This turns O(152,000) per tick into O(~15,000) per tick.

LEVEL 4: PIPELINE PARALLELISM
──────────────────────────────
Phase 3 (Learn) for tick N can overlap with Phase 1 (Propagate) for tick N+1.
Learning reads tick N's activations (immutable by this point) while
propagation writes tick N+1's activations.

Two-stage pipeline: Propagate_N+1 | Learn_N running simultaneously.

LEVEL 5: TRACE MATCHING (embarrassingly parallel)
─────────────────────────────────────────────────
Pattern recognition compares ~100k traces against current activation.
Even with inverted index optimization, candidate scoring can be split
across threads. Each thread takes a chunk of candidates.
```

### 6.2 Rayon Integration (Rust)

```rust
// Region-level parallelism with rayon
regions.par_iter_mut().for_each(|region| {
    region.propagate(&synapse_pool, &prev_activations, &attention_gains);
});

// Neuron-level within a region
region.active_list.par_chunks(64).for_each(|chunk| {
    for &neuron_id in chunk {
        // compute new potential for each active neuron
    }
});

// Trace matching
trace_candidates.par_iter().map(|trace| {
    score_match(trace, &current_activations)
}).collect();
```

### 6.3 Memory Layout for Performance

```
Region data stored as Structure of Arrays (SoA) for SIMD:
    potentials:   [f32; N]     ← contiguous, vectorizes beautifully
    activations:  [f32; N]
    thresholds:   [f32; N]
    refractory:   [u16; N]

NOT Array of Structures:
    neurons: [Neuron; N]       ← poor cache utilization when operating on one field
```

---

## 7. SEED SYSTEM (INITIAL STATE)

### 7.1 Spawn Order

```
STEP 1: spawn_neurons.py
  - Create all 152,000 neurons with region assignments
  - 80% excitatory, 20% inhibitory per region
  - Initialize all potentials to 0, activations to 0

STEP 2: spawn_synapses.py (WITHIN-REGION)
  - For each region, create sparse random connections between neurons
  - ~10-30 synapses per neuron (region-dependent)
  - Weights initialized small random: uniform(0.01, 0.15)
  - Plasticity = 1.0 (fully learnable)
  - No cross-region synapses yet (those come from traces)

STEP 3: spawn_traces.py
  - Generate 100,000 random traces
  - Each trace randomly selects neurons per region:
      sensory: 3, visual: 6, audio: 3, memory_short: 3,
      memory_long: 4, emotion: 2, attention: 2, pattern: 3,
      integration: 4, language: 3, executive: 2, motor: 2, speech: 2
      (= 39 neurons per trace)
  - Initialize: strength=random(0.1,0.3), decay=1.0, polarity=0.0,
    abstraction=random(0.0,0.4), novelty=1.0, co_traces=[], context_tags=[]

STEP 4: spawn cross-region synapses
  - For each trace, create synapses between its neurons across regions
  - e.g., sensory neuron → pattern neuron, visual → integration, etc.
  - Following the signal flow diagram (section 1.3)
  - Weights initialized: uniform(0.05, 0.2)
  - These are the backbone connections that implement the architecture

STEP 5: physics_traces.py
  - Handcraft ~200 traces representing physical reality:
    gravity, hot, cold, hard, soft, bright, dark, heavy, light, fast, slow...
  - These have specific neuron assignments (not random)
  - Higher initial strength (0.5-0.8)
  - Specific polarity (pain = -0.9, warmth = +0.3)
  - Specific co_traces (hot → burn, cold → ice, etc.)

STEP 6: relational_traces.py
  - Handcraft ~200 traces for relational concepts:
    bigger, smaller, inside, outside, before, after, causes, prevents,
    same, different, part-of, contains, above, below...
  - Heavy in language region neurons, light in sensory
  - High abstraction (0.6-0.9)
  - These enable reasoning about relationships

STEP 7: numbers wiring
  - Hardwire the numbers region
  - Create digit neurons, place value neurons, magnitude neurons
  - All synapses plasticity ≈ 0.0
  - Create traces for numbers 0-9, 10, 100, 1000
  - Wire to language region (number words)

STEP 8: reflex wiring
  - Wire direct sensory → motor reflex arcs
  - Pain → withdraw, loud → startle, hot → retract
  - Low delay (2-3 ticks), low plasticity (0.05)
  - These work before any learning occurs
```

### 7.2 Initial Synapse Budget

```
Within-region:  152,000 neurons × ~20 synapses avg = ~3,000,000
Cross-region:   100,000 traces × ~15 cross-region synapses = ~1,500,000
Seed traces:    400 seed traces × ~50 specific synapses = ~20,000
Reflexes:       ~500 hardwired connections
Numbers:        ~5,000 hardwired connections

INITIAL TOTAL: ~4,525,000 synapses

During BLOOM phase, this grows to ~15,000,000
After CRITICAL PRUNING, settles to ~5,000,000 - 8,000,000
```

---

## 8. INPUT / OUTPUT SPECIFICATIONS

### 8.1 Input Pipeline

```python
# All inputs produce the same thing: a set of (neuron_id, activation_value) pairs
# that get injected into the appropriate region at Phase 1.

InputSignal = list[tuple[int, float]]  # [(neuron_id, activation), ...]

class VisualInput:
    """image/video frame → visual region activations"""
    def process(self, frame: np.ndarray) -> InputSignal:
        # Basic version: resize to fixed dims, edge detect, color extract
        # Map features to visual sub-regions (edges→low, shapes→mid, etc.)
        # Returns ~200-1000 neuron activations per frame

class AudioInput:
    """audio chunk → audio region activations"""
    def process(self, chunk: np.ndarray, sample_rate: int) -> InputSignal:
        # FFT → frequency bands → map to pitch neurons
        # Onset detection → temporal neurons
        # Returns ~100-500 neuron activations per chunk

class TextInput:
    """text → language region activations"""
    def process(self, text: str) -> InputSignal:
        # Character-level or phoneme-level encoding
        # Maps to language symbol sub-region
        # Also triggers audio (sub-vocalization) at reduced gain
        # Returns ~50-200 neuron activations per word

class SensoryInput:
    """numerical sensor values → sensory region activations"""
    def process(self, temperature: float, pressure: float,
                pain: float, texture: float) -> InputSignal:
        # Direct mapping to sensory sub-ranges
        # population coding: value activates a gaussian bump of ~20-50 neurons
        # centered on the neuron that represents that value range

class MultimodalInput:
    """synchronizes multiple input streams with timing"""
    def process(self, inputs: dict[str, InputSignal], tick: int) -> InputSignal:
        # Merges all inputs with tick timestamps
        # Handles different input rates (video @ 30fps, audio @ 100 chunks/sec)
        # Returns combined input for this tick
```

### 8.2 Output Pipeline

```python
class MotorOutput:
    """motor neuron activations → action decisions"""
    def read(self, activations: dict[int, float]) -> Action:
        # Decode which motor patterns are active
        # Map to action space (approach, withdraw, grab, push, etc.)

class SpeechOutput:
    """speech neuron activations → text/phonemes"""
    def read(self, activations: dict[int, float]) -> str:
        # Decode active phoneme patterns
        # Assemble into words (reverse of TextInput)

class ImaginationOutput:
    """internal visual activations → image (for debugging/visualization)"""
    def read(self, visual_activations: dict[int, float]) -> np.ndarray:
        # Reverse map visual neuron activations to image
        # For debugging: "what is the brain imagining right now?"
```

---

## 9. FILE STRUCTURE (REFINED)

```
brain/
│
├── core/                              ← RUST (performance critical)
│   ├── neuron.rs                      # Neuron struct, LIF model, fire/decay
│   ├── synapse.rs                     # Synapse struct, CSR pool, batch updates
│   ├── binding.rs                     # Binding struct, pattern matching, time_delta
│   ├── trace.rs                       # Trace struct (Rust-side for fast matching)
│   ├── region.rs                      # Region struct, SoA storage, active lists
│   ├── tick.rs                        # Tick cycle orchestration, 4 phases
│   ├── attention.rs                   # Gain map computation
│   ├── propagate.rs                   # Signal propagation (Phase 1 hot loop)
│   ├── neuromodulator.rs              # Global state (arousal, valence, focus, energy)
│   ├── brain.rs                       # Top-level struct, holds all regions + pools
│   └── lib.rs                         # PyO3 bindings, Python-callable functions
│
├── regions/                           ← RUST (region-specific firing rules)
│   ├── sensory.rs                     # Reflex shortcut, sub-range logic
│   ├── visual.rs                      # Hierarchical sub-regions, imagination flag
│   ├── audio.rs                       # Temporal buffer, frequency decomposition
│   ├── memory_short.rs               # Capacity limit, eviction, rehearsal
│   ├── memory_long.rs                # Pattern completion, cue retrieval
│   ├── emotion.rs                     # Polarity computation, arousal override
│   ├── attention.rs                   # Gain control, novelty/threat/relevance drives
│   ├── pattern.rs                     # Trace matching, prediction, novelty signal
│   ├── integration.rs                # Multi-region binding, temporal window
│   ├── language.rs                    # Symbol linking, relational activation
│   ├── executive.rs                   # Conflict resolution, inhibition, planning
│   ├── motor.rs                       # Action sequencing, reflex vs deliberate
│   ├── speech.rs                      # Phoneme sequencing, prosody
│   └── numbers.rs                     # Hardwired structure, magnitude comparison
│
├── structures/                        ← PYTHON (data management)
│   ├── trace_store.py                 # Create/load/save/query traces, inverted index
│   ├── synapse_pool.py                # Python-side synapse ops (create, prune, stats)
│   ├── binding_pool.py                # Binding management, formation/dissolution
│   ├── neuron_map.py                  # Region ranges, neuron type lookup
│   └── brain_state.py                 # Current global state, activation snapshot
│
├── learning/                          ← PYTHON (learning policy)
│   ├── hebbian.py                     # Co-activation → synapse strengthening
│   ├── anti_hebbian.py                # Decorrelation → synapse weakening
│   ├── pruning.py                     # Phase-aware pruning (bloom/critical/mature)
│   ├── trace_formation.py             # Novel pattern → new trace creation
│   ├── trace_merging.py               # Overlapping traces → merged concept
│   ├── binding_formation.py           # Co-pattern → new binding
│   ├── novelty.py                     # Novelty scoring, prediction error
│   ├── consolidation.py               # Memory transfer (short→long), replay, context stripping
│   └── prediction.py                  # Next-tick prediction, error computation
│
├── input/                             ← PYTHON (feeding data in)
│   ├── visual_input.py                # Image/video → visual neuron activations
│   ├── audio_input.py                 # Sound → audio neuron activations
│   ├── text_input.py                  # Text → language neuron activations
│   ├── sensory_input.py               # Sensor values → sensory activations
│   └── multimodal.py                  # Sync + merge inputs with timing
│
├── output/                            ← PYTHON (reading results out)
│   ├── motor_output.py                # Motor activations → actions
│   ├── speech_output.py               # Speech activations → text/sound
│   └── imagination.py                 # Internal visual → image (debug)
│
├── seed/                              ← PYTHON (bootstrap, run once)
│   ├── spawn_neurons.py               # Generate all 152k neurons
│   ├── spawn_synapses.py              # Random within-region + cross-region synapses
│   ├── spawn_traces.py                # Generate 100k random traces
│   ├── physics_traces.py              # Handcraft ~200 physics seed traces
│   ├── relational_traces.py           # Handcraft ~200 relational seed traces
│   ├── numbers_wiring.py              # Hardwire numbers region
│   ├── reflex_wiring.py               # Hardwire sensory→motor reflexes
│   └── seed_runner.py                 # Orchestrates all seed steps in order
│
├── utils/                             ← PYTHON (tools)
│   ├── serializer.py                  # Save/load entire brain state to disk
│   ├── metrics.py                     # Live stats: synapses, traces, firing rates
│   ├── visualizer.py                  # Firing pattern visualization (optional)
│   └── config.py                      # All hyperparameters in one place
│
├── Cargo.toml                         # Rust dependencies (PyO3, rayon, etc.)
├── pyproject.toml                     # Python project config
├── requirements.txt                   # Python dependencies
└── main.py                            # Entry point: seed → run loop
```

---

## 10. PyO3 BRIDGE (Rust ↔ Python API)

```rust
// What Python can call into Rust:

// === TICK CONTROL ===
fn tick() -> TickResult                          // advance one brain cycle (all 4 phases)
fn tick_propagate_only()                         // Phase 1 only (for pipeline overlap)
fn get_tick_count() -> u64                       // current tick

// === INPUT ===
fn inject_activations(signals: Vec<(u32, f32)>)  // set neuron activations from input
fn set_attention_gains(gains: HashMap<RegionId, f32>)  // set per-region attention multiplier

// === READ STATE ===
fn get_activations(region_id: RegionId) -> Vec<(u32, f32)>  // active neurons in region
fn get_all_activations() -> HashMap<RegionId, Vec<(u32, f32)>>
fn get_neuron_potential(neuron_id: u32) -> f32
fn get_active_count(region_id: RegionId) -> u32   // sparsity monitoring

// === SYNAPSE OPS ===
fn get_synapse_weight(from: u32, to: u32) -> Option<f32>
fn update_synapse(from: u32, to: u32, delta: f32)  // queue weight change
fn create_synapse(from: u32, to: u32, weight: f32, delay: u8, plasticity: f32)
fn prune_synapse(from: u32, to: u32)                // queue removal
fn get_synapse_count() -> u64
fn rebuild_synapse_index()                          // apply all queued changes

// === REGION OPS ===
fn set_region_threshold_modifier(region: RegionId, modifier: f32)
fn get_region_firing_rate(region: RegionId) -> f32  // % of neurons active

// === NEUROMODULATOR ===
fn set_neuromodulator(arousal: f32, valence: f32, focus: f32, energy: f32)
fn get_neuromodulator() -> (f32, f32, f32, f32)

// === BINDING OPS ===
fn create_binding(pattern_a: PatternRef, pattern_b: PatternRef) -> u32
fn get_binding_weight(binding_id: u32) -> f32
fn update_binding(binding_id: u32, delta_weight: f32)
fn check_binding_activation(binding_id: u32) -> bool   // did both patterns fire?

// === TRACE MATCHING (Rust-side for performance) ===
fn register_trace_neurons(trace_id: &str, neurons: HashMap<RegionId, Vec<u32>>)
fn get_matching_traces(threshold: f32) -> Vec<(String, f32)>  // (trace_id, match_score)
fn get_trace_activation(trace_id: &str) -> f32                // how active is this trace?
```

---

## 11. HYPERPARAMETERS (config.py)

All tunable values in one place. These WILL need tuning.

```python
# === NEURON PARAMETERS (per region) ===
REGION_CONFIG = {
    "sensory":      {"threshold": 0.30, "leak": 0.85, "refractory": 2, "inhibitory_pct": 0.15},
    "visual":       {"threshold": 0.40, "leak": 0.90, "refractory": 3, "inhibitory_pct": 0.20},
    "audio":        {"threshold": 0.35, "leak": 0.88, "refractory": 2, "inhibitory_pct": 0.20},
    "memory_short": {"threshold": 0.50, "leak": 0.95, "refractory": 5, "inhibitory_pct": 0.25},
    "memory_long":  {"threshold": 0.60, "leak": 0.99, "refractory": 8, "inhibitory_pct": 0.20},
    "emotion":      {"threshold": 0.30, "leak": 0.92, "refractory": 4, "inhibitory_pct": 0.15},
    "attention":    {"threshold": 0.40, "leak": 0.93, "refractory": 3, "inhibitory_pct": 0.40},
    "pattern":      {"threshold": 0.50, "leak": 0.90, "refractory": 4, "inhibitory_pct": 0.25},
    "integration":  {"threshold": 0.55, "leak": 0.92, "refractory": 5, "inhibitory_pct": 0.20},
    "language":     {"threshold": 0.50, "leak": 0.94, "refractory": 4, "inhibitory_pct": 0.20},
    "executive":    {"threshold": 0.60, "leak": 0.93, "refractory": 6, "inhibitory_pct": 0.30},
    "motor":        {"threshold": 0.55, "leak": 0.85, "refractory": 3, "inhibitory_pct": 0.20},
    "speech":       {"threshold": 0.50, "leak": 0.87, "refractory": 3, "inhibitory_pct": 0.20},
    "numbers":      {"threshold": 0.50, "leak": 0.96, "refractory": 4, "inhibitory_pct": 0.10},
}

# === LEARNING ===
HEBBIAN_RATE = 0.01              # base learning rate
ANTI_HEBBIAN_RATE = 0.003        # decorrelation rate
HEBBIAN_WINDOW = 3               # ticks for co-activation
NOVELTY_LEARNING_BOOST = 2.0     # multiply learning rate by this when novel
AROUSAL_LEARNING_BOOST = 1.5     # multiply learning rate when aroused

# === PRUNING ===
BLOOM_END_TICK = 500_000
CRITICAL_END_TICK = 2_000_000
BLOOM_SYNAPSE_THRESHOLD = 15_000_000

BLOOM_CREATE_THRESHOLD = 0.1
CRITICAL_CREATE_THRESHOLD = 0.3
MATURE_CREATE_THRESHOLD = 0.5

CRITICAL_PRUNE_WEIGHT = 0.15
CRITICAL_PRUNE_DORMANT_TICKS = 50_000
CRITICAL_PRUNE_DORMANT_WEIGHT = 0.3
MATURE_PRUNE_WEIGHT = 0.05
MATURE_PRUNE_DORMANT_TICKS = 200_000

# === ATTENTION ===
ATTENTION_GAIN_MIN = 0.1
ATTENTION_GAIN_MAX = 5.0
ATTENTION_INERTIA_TICKS = 15      # ticks to shift focus
NOVELTY_ATTENTION_WEIGHT = 0.4
THREAT_ATTENTION_WEIGHT = 0.4
RELEVANCE_ATTENTION_WEIGHT = 0.2

# === WORKING MEMORY ===
WORKING_MEMORY_CAPACITY = 7
WORKING_MEMORY_DECAY_RATE = 0.02   # per tick without rehearsal

# === TRACES ===
TRACE_ACTIVATION_THRESHOLD = 0.6   # fraction of neurons that must fire
TRACE_FORMATION_PERSISTENCE = 20   # ticks novel pattern must persist
TRACE_FORMATION_MIN_REGIONS = 2    # minimum regions involved
TRACE_MERGE_OVERLAP = 0.8          # neuron overlap for merging

# === BINDINGS ===
BINDING_TEMPORAL_WINDOW = 5        # ticks for co-activation detection
BINDING_FORMATION_COUNT = 5        # co-activations before binding created
BINDING_DISSOLUTION_WEIGHT = 0.05  # prune below this
BINDING_DISSOLUTION_MIN_FIRES = 10

# === PREDICTION ===
PREDICTION_SURPRISE_THRESHOLD = 0.5
PREDICTION_ALARM_THRESHOLD = 0.8
PREDICTION_BORING_THRESHOLD = 0.1
PREDICTION_SURPRISE_DURATION = 50   # ticks of boosted learning
PREDICTION_ALARM_DURATION = 200     # ticks of hypervigilance

# === CONSOLIDATION ===
CONSOLIDATION_TRIGGER_ENERGY = 0.2
CONSOLIDATION_TRIGGER_TICKS = 100_000
CONSOLIDATION_DURATION = 10_000
CONSOLIDATION_INPUT_GAIN = 0.1     # reduce input during consolidation

# === SEED ===
INITIAL_TRACES = 100_000
NEURONS_PER_TRACE = {
    "sensory": 3, "visual": 6, "audio": 3, "memory_short": 3,
    "memory_long": 4, "emotion": 2, "attention": 2, "pattern": 3,
    "integration": 4, "language": 3, "executive": 2, "motor": 2,
    "speech": 2, "numbers": 0,
}
INITIAL_WITHIN_REGION_SYNAPSES_PER_NEURON = 20
INITIAL_CROSS_REGION_SYNAPSES_PER_TRACE = 15
SEED_PHYSICS_TRACES = 200
SEED_RELATIONAL_TRACES = 200
```

---

## 12. IMPLEMENTATION PHASES

### Phase 1: Foundation (Rust core compiles, Python can call it)
```
BUILD:
  core/neuron.rs        — Neuron struct, LIF fire/decay
  core/region.rs        — Region struct, SoA storage, active lists
  core/synapse.rs       — SynapsePool with CSR, batch ops
  core/propagate.rs     — Signal propagation hot loop
  core/tick.rs          — Basic tick cycle (Phase 1 only)
  core/brain.rs         — Top-level holder
  core/lib.rs           — PyO3 bridge (inject, tick, read)

TEST:
  Create 1000 neurons, 5000 synapses, inject signal, tick, verify propagation.
  Verify LIF dynamics: potential accumulates, fires at threshold, refractory works.
  Verify inhibitory neurons subtract.
  Benchmark: ticks/second with full 152k neurons at 5% sparsity.

MILESTONE: can call brain.tick() from Python, inject signals, read activations.
```

### Phase 2: Seed & Structure (Brain has initial state)
```
BUILD:
  utils/config.py       — All hyperparameters
  structures/neuron_map.py — Region ranges, type assignments
  seed/spawn_neurons.py  — Generate 152k neurons
  seed/spawn_synapses.py — Random within-region + cross-region
  seed/spawn_traces.py   — Generate 100k random traces
  structures/trace_store.py — Trace management, inverted index

TEST:
  Seed a full brain, serialize to disk, load, verify integrity.
  Verify neuron counts per region match spec.
  Verify synapses exist and initial weights are correct.
  Verify trace→neuron index and neuron→trace inverted index.

MILESTONE: brain has 152k neurons, ~4.5M synapses, 100k traces.
```

### Phase 3: Learning Core (Brain can learn)
```
BUILD:
  learning/hebbian.py    — Co-activation detection, synapse strengthening
  learning/anti_hebbian.py — Decorrelation
  learning/pruning.py    — Phase-aware pruning
  learning/novelty.py    — Novelty scoring
  structures/brain_state.py — Activation snapshots

TEST:
  Inject same signal repeatedly → verify synapses strengthen.
  Inject uncorrelated signals → verify synapses weaken.
  Run 1M ticks → verify pruning lifecycle (bloom → critical → mature).
  Monitor synapse count over time → should follow expected curve.

MILESTONE: brain learns from repeated stimuli, prunes unused connections.
```

### Phase 4: Attention & Pattern Recognition (Brain can focus and recognize)
```
BUILD:
  core/attention.rs       — Gain map
  regions/attention.rs    — Novelty/threat/relevance drives
  regions/pattern.rs      — Trace matching, prediction, novelty signal
  learning/prediction.py  — Next-tick prediction, error computation

TEST:
  Inject novel pattern → verify attention spikes.
  Inject familiar pattern → verify attention drops.
  Inject threatening pattern (high polarity) → verify attention override.
  Verify pattern recognition finds matching traces above threshold.

MILESTONE: brain attends to novel/important stimuli, recognizes familiar ones.
```

### Phase 5: Integration & Memory (Brain can bind and remember)
```
BUILD:
  core/binding.rs           — Binding struct, evaluation
  regions/integration.rs    — Multi-region merging
  regions/memory_short.rs   — Working memory, capacity limit
  regions/memory_long.rs    — Pattern completion, cue retrieval
  learning/consolidation.py — Short→long transfer
  learning/trace_formation.py
  learning/binding_formation.py

TEST:
  Present multi-modal input → verify integration binds it.
  Verify working memory holds max 7 traces.
  Verify consolidation transfers important memories to long-term.
  Verify pattern completion: partial cue → full recall.

MILESTONE: brain has working memory, long-term memory, multi-modal binding.
```

### Phase 6: Emotion & Executive (Brain can decide)
```
BUILD:
  regions/emotion.rs       — Polarity, arousal, urgency
  core/neuromodulator.rs   — Global state modulation
  regions/executive.rs     — Conflict resolution, planning, inhibition
  learning/trace_merging.py

TEST:
  Present threat → verify emotion activates, arousal rises, attention shifts.
  Present conflicting options → verify executive resolves.
  Verify impulse suppression: emotion fires motor, executive blocks.

MILESTONE: brain has emotional responses and deliberate decision-making.
```

### Phase 7: Language & Speech (Brain can think in symbols)
```
BUILD:
  regions/language.rs      — Symbol processing, relational logic
  regions/speech.rs        — Phoneme patterns
  input/text_input.py      — Text → language activations
  output/speech_output.py  — Activations → text

TEST:
  Feed text → verify language region activates appropriate traces.
  Verify language↔executive reasoning loop works.
  Verify speech output produces coherent phoneme sequences.

MILESTONE: brain can process and produce language.
```

### Phase 8: Full Input/Output (Brain meets the world)
```
BUILD:
  input/visual_input.py
  input/audio_input.py
  input/sensory_input.py
  input/multimodal.py
  output/motor_output.py
  output/imagination.py
  seed/physics_traces.py
  seed/relational_traces.py
  seed/numbers_wiring.py
  seed/reflex_wiring.py
  seed/seed_runner.py

TEST:
  Full pipeline: image input → visual → pattern → integration → executive → motor output.
  Multi-modal: simultaneous image + sound → verify binding.
  Reflexes: pain input → motor output in < 5 ticks.

MILESTONE: brain can perceive multi-modal input and produce actions.
```

### Phase 9: Polish & Observe
```
BUILD:
  utils/serializer.py
  utils/metrics.py
  utils/visualizer.py (optional)
  main.py

TEST:
  Full lifecycle: seed → bloom → prune → mature → continuous learning.
  Save/load brain state, verify continuity.
  Long-run stability: 10M+ ticks without divergence or collapse.

MILESTONE: brain runs continuously, learns, remembers, decides, speaks.
```

---

## 13. CRITICAL DESIGN DECISIONS & TRADE-OFFS

### 13.1 Why NOT a Neural Network

This system has NO:
- Loss function (nothing is "wrong" — learning is local and unsupervised)
- Backpropagation (signals go forward, learning is Hebbian)
- Training/inference split (always learning, always running)
- Gradient computation (weight updates are local co-activation rules)
- Batch processing (one tick at a time, real-time)

This means: slower at narrow tasks (an NN trained on ImageNet will crush this at image classification). But: no catastrophic forgetting, continuous learning, genuine multi-modal integration, and the capacity for novel reasoning rather than pattern interpolation.

### 13.2 Why Rust Core + Python Policy

- Signal propagation is O(active_neurons × avg_synapses) per tick. At 5% sparsity with 152k neurons and 5M synapses, this is ~500k-2M operations per tick. Needs to hit 1000+ ticks/second. Python can't do this. Rust can.
- Learning rules, trace management, and consolidation are complex policy that changes during development. Python lets us iterate fast.
- PyO3 gives near-zero-cost FFI. The bridge is not the bottleneck.

### 13.3 Why 152k Neurons (Not More, Not Less)

- 152k is large enough for meaningful patterns across 14 regions.
- Small enough to simulate in real-time on a single machine.
- Each region has 2k-20k neurons — enough for ~100-500 distinct activation patterns per region at 5% sparsity.
- Can scale later by multiplying all region sizes by the same factor.

### 13.4 Open Questions to Resolve During Implementation

1. **Prediction mechanism specifics**: How exactly does pattern region predict next tick? Simple linear extrapolation? Trace-sequence lookup? This needs experimentation.

2. **Imagination gating**: When memory reverse-fires visual, how do we prevent the brain from confusing imagination with real input? The "imagination flag" needs concrete implementation.

3. **Consolidation fidelity**: How lossy is short→long transfer? Does compression happen (fewer neurons in long-term than short-term)? Needs tuning.

4. **Multi-trace interference**: When 5+ traces are active simultaneously, how do we prevent activation soup where everything bleeds into everything? Lateral inhibition should handle this but needs verification.

5. **Time perception**: The brain has no explicit clock. How does it perceive duration? Possibly through decay rates (fresh activation = recent, decayed = old). Needs design.

6. **Attention shift cost**: How expensive is shifting focus? Too cheap = scattered. Too expensive = rigid. The 15-tick inertia parameter needs tuning.

7. **Emotional homeostasis**: How does arousal return to baseline? Simple exponential decay toward 0.5? Or does it need active regulation?

8. **Trace saturation**: With 100k traces and continuous learning, eventually we run out of "unique neuron combinations." When does this happen and what's the strategy? (Answer: probably at ~500k-1M traces. Strategy: aggressive merging of similar traces.)

---

## 14. WHAT MAKES THIS DIFFERENT FROM EXISTING APPROACHES

| Feature | LLMs | Spiking Neural Nets | This System |
|---------|-------|-------------------|-------------|
| Representation | Token embeddings | Spike trains | Distributed traces |
| Learning | Backprop (offline) | STDP (local) | Hebbian + pruning + consolidation |
| Memory | Context window | Short-term only | Working + long-term + consolidation |
| Attention | Softmax over tokens | None typically | Gain control filter |
| Emotion | None | None | Priority tagger + neuromodulator |
| Multi-modal | Separate encoders merged | Possible but rare | Built-in from architecture |
| Prediction | Next token | None typically | Next-tick activation prediction |
| Reasoning | In-context pattern | None | Language↔Executive loop |

This isn't better or worse than any of these. It's attempting something different: a unified cognitive architecture where perception, memory, emotion, attention, reasoning, and action emerge from the same substrate rather than being bolted together.

---

*End of plan. Build order starts at Phase 1: Rust core.*
