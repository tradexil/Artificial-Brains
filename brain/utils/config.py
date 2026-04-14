"""All hyperparameters in one place. Never hardcode these in modules."""

# === REGION DEFINITIONS ===

REGIONS = {
    "sensory":      (0,       9_999),
    "visual":       (10_000,  29_999),
    "audio":        (30_000,  44_999),
    "memory_short": (45_000,  54_999),
    "memory_long":  (55_000,  69_999),
    "emotion":      (70_000,  79_999),
    "attention":    (80_000,  84_999),
    "pattern":      (85_000,  94_999),
    "integration":  (95_000,  104_999),
    "language":     (105_000, 119_999),
    "executive":    (120_000, 129_999),
    "motor":        (130_000, 139_999),
    "speech":       (140_000, 149_999),
    "numbers":      (150_000, 151_999),
}

TOTAL_NEURONS = 152_000

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
    "executive":    {"threshold": 0.50, "leak": 0.93, "refractory": 6, "inhibitory_pct": 0.30},
    "motor":        {"threshold": 0.55, "leak": 0.85, "refractory": 3, "inhibitory_pct": 0.20},
    "speech":       {"threshold": 0.50, "leak": 0.87, "refractory": 3, "inhibitory_pct": 0.20},
    "numbers":      {"threshold": 0.50, "leak": 0.96, "refractory": 4, "inhibitory_pct": 0.10},
}

# === LEARNING ===
HEBBIAN_RATE = 0.005
ANTI_HEBBIAN_RATE = 0.005
HEBBIAN_WINDOW = 3  # ticks for co-activation
NOVELTY_LEARNING_BOOST = 1.5
AROUSAL_LEARNING_BOOST = 1.2
COACTIVE_TRACK_INTERVAL = 1000  # align dormancy snapshots with pruning cadence

# === WAVE / MULTI-RATE SCHEDULING ===
PERCEPTION_LANE_CADENCE = 1
COGNITION_LANE_CADENCE = 2
CONTROL_LANE_CADENCE = 2
MEMORY_LONG_REGION_CADENCE = 4

# === MAINTENANCE CADENCE ===
LIGHT_MAINTENANCE_INTERVAL = 8
BINDING_TRACKER_CLEANUP_INTERVAL = LIGHT_MAINTENANCE_INTERVAL
PRUNE_INTERVAL = 1000
BINDING_MAINTENANCE_INTERVAL = 500
REBUILD_INTERVAL = 10000

# === SYNAPSE UPDATE RELEASE ===
# Keep learning deltas queued continuously, but release them in smaller bounded
# batches so longer runs do not hit a single maintenance cliff.
SYNAPSE_UPDATE_RELEASE_INTERVAL = 8
SYNAPSE_UPDATE_MAX_BATCH_SYNAPSE_MULTIPLIER = 2.0
SYNAPSE_UPDATE_TARGET_DEFERRED_SYNAPSE_MULTIPLIER = 1.0

# === PRUNING LIFECYCLE ===
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
ATTENTION_INERTIA_TICKS = 15
NOVELTY_ATTENTION_WEIGHT = 0.4
THREAT_ATTENTION_WEIGHT = 0.4
RELEVANCE_ATTENTION_WEIGHT = 0.2

# === WORKING MEMORY ===
WORKING_MEMORY_CAPACITY = 7
WORKING_MEMORY_DECAY_RATE = 0.02

# === TRACES ===
TRACE_ACTIVATION_THRESHOLD = 0.6
TRACE_ACTIVE_NEURON_BUDGET_FRACTION = 0.05
TRACE_ACTIVE_NEURON_BUDGET = int(TOTAL_NEURONS * TRACE_ACTIVE_NEURON_BUDGET_FRACTION)
TRACE_FRESHNESS_RETENTION = 0.85
TRACE_FRESHNESS_FLOOR = 0.15
TRACE_FRESHNESS_MIN_SCORE = 0.12
TRACE_REFRESH_MAX_BOOST = 1.2
TRACE_AGE_DECAY_WINDOW = 20
TRACE_AGE_FLOOR_CEILING = 0.4
TRACE_FORMATION_PERSISTENCE = 5
TRACE_FORMATION_JACCARD_THRESHOLD = 0.4
TRACE_FORMATION_MIN_REGIONS = 2
TRACE_MERGE_OVERLAP = 0.8
# Keep the specificity gate live in the default runtime path.
TRACE_FORMATION_BASELINE_WINDOW = HEBBIAN_WINDOW
TRACE_FORMATION_MIN_BASELINE_DELTA = 0.15
TRACE_FORMATION_MIN_BASELINE_RATIO = 1.75
TRACE_FORMATION_MAX_NEURONS_PER_REGION = 12
TRACE_FORMATION_MAX_TOTAL_NEURONS = 48
TRACE_FORMATION_EXCLUDED_REGIONS = {"attention", "memory_short"}
TRACE_FORMATION_EXCLUDED_REGION_PREFIX_NEURONS = {"integration": 100}
TRACE_FORMATION_REGION_MAX_NEURONS = {
    "emotion": 4,
    "integration": 8,
    "language": 8,
    "memory_long": 8,
    "pattern": 8,
    "visual": 24,
}
TRACE_FORMATION_VISUAL_FAMILY_MAX_NEURONS = {
    "low": 8,
    "mid": 8,
    "spatial": 8,
}

# === BINDINGS ===
# Effective binding horizon is `BINDING_TEMPORAL_WINDOW * BINDING_FORMATION_COUNT`.
# Keep this within roughly 1-2 samples of sustained text input to avoid
# cross-sample temporal bleed during formation.
BINDING_TEMPORAL_WINDOW = 2
BINDING_FORMATION_COUNT = 5
BINDING_DISSOLUTION_WEIGHT = 0.10
BINDING_DISSOLUTION_MIN_FIRES = 10
BINDING_CANDIDATE_BUDGET = 60
BINDING_CANDIDATE_AUDIO_CROSS_MODAL_RESERVE = 16
BINDING_RECALL_MIN_RELATIVE_WEIGHT = 0.85
BINDING_RECALL_BOOST_SCALE = 0.25
BINDING_RECALL_PATTERN_COMPLETION_THRESHOLD = 0.25
BINDING_RECALL_PATTERN_COMPLETION_BOOST = 0.8
BINDING_RECALL_TRACE_MATCH_THRESHOLD = 0.4

# === CUE COLLISION BENCHMARK ===
CUE_COLLISION_STRONG_REINFORCEMENTS = 1
CUE_COLLISION_WEAK_MISSES = 0

# === TEXT INPUT ===
TEXT_INPUT_KNOWN_REGION_SCALES = {
    "language": 1.0,
    "pattern": 0.75,
    "integration": 0.65,
    "memory_long": 0.45,
    "memory_short": 0.35,
    "sensory": 0.55,
    "visual": 0.55,
    "audio": 0.55,
    "emotion": 0.4,
    "attention": 0.35,
    "numbers": 0.75,
}
TEXT_INPUT_UNKNOWN_REGION_BOOSTS = {
    "language": 0.6,
    "pattern": 0.7,
    "integration": 0.65,
    "attention": 0.45,
}
TEXT_INPUT_UNKNOWN_REGION_COUNTS = {
    "language": 3,
    "pattern": 3,
    "integration": 3,
    "attention": 2,
}
TEXT_INPUT_MAX_MATCHED_TRACES_PER_SPAN = 4

# === PREDICTION ===
PREDICTION_SURPRISE_THRESHOLD = 0.5
PREDICTION_ALARM_THRESHOLD = 0.8
PREDICTION_BORING_THRESHOLD = 0.1
PREDICTION_SURPRISE_DURATION = 50
PREDICTION_ALARM_DURATION = 200

# === CONSOLIDATION ===
CONSOLIDATION_TRIGGER_ENERGY = 0.2
CONSOLIDATION_TRIGGER_TICKS = 100_000
CONSOLIDATION_DURATION = 10_000
CONSOLIDATION_INPUT_GAIN = 0.1

# === HOMEOSTASIS & SLEEP (Phase 9) ===
HOMEOSTASIS_AROUSAL_REG_RATE = 0.005
HOMEOSTASIS_VALENCE_REG_RATE = 0.002
HOMEOSTASIS_FOCUS_REG_RATE = 0.003
SLEEP_PRESSURE_RATE = 0.00002
SLEEP_DISSIPATION_RATE = 0.0001
CIRCADIAN_PERIOD = 100_000
SLEEP_DROWSY_DURATION = 2_000
SLEEP_LIGHT_DURATION = 5_000
SLEEP_DEEP_DURATION = 8_000
SLEEP_REM_DURATION = 5_000
DREAM_REPLAY_PER_TICK = 3
WAKE_ALARM_PAIN_THRESHOLD = 0.8

# === SPEECH DECODER ===
SPEECH_OUTPUT_THRESHOLD = 0.5
SPEECH_DECODE_TOP_K = 10
SPEECH_BOOST_MULTIPLIER = 0.8  # voltage injected into speech neurons per matched trace

# === SCHEMA ===
SCHEMA_MIN_SEQUENCE_LENGTH = 2
SCHEMA_MAX_SEQUENCE_LENGTH = 8
SCHEMA_FORMATION_COUNT = 3  # times a causal sequence must repeat
SCHEMA_CAUSAL_WINDOW_TICKS = 15  # max delay between sequential traces
SCHEMA_PREDICTION_BONUS = 0.3  # reward for confirmed prediction
SCHEMA_SURPRISE_PENALTY = 0.4  # surprise signal for missed prediction

# === CHUNKING ===
CHUNK_MIN_WORDS = 5    # never produce a chunk smaller than this
CHUNK_MAX_WORDS = 60   # hard cap on words per chunk (adaptive target stays below)

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
SEED_WITHIN_REGION_DEFAULT_DELAY_RANGE = (1, 2)
SEED_WITHIN_REGION_DEFAULT_WEIGHT_RANGE = (0.01, 0.15)
SEED_CROSS_REGION_DEFAULT_DELAY_RANGE = (3, 8)
SEED_CROSS_REGION_DELAY_RANGE_BY_CONNECTION = {
    ("emotion", "executive"): (1, 3),
    ("pattern", "executive"): (1, 3),
    ("integration", "executive"): (1, 3),
    ("language", "executive"): (1, 3),
}
SEED_CROSS_REGION_DEFAULT_WEIGHT_RANGE = (0.05, 0.2)
SEED_CROSS_REGION_WEIGHT_RANGE_BY_CONNECTION = {
    ("executive", "speech"): (0.002, 0.02),
}
SEED_WITHIN_REGION_SYNAPSES_PER_NEURON_BY_REGION = {
    "memory_long": 5,
}
SEED_WITHIN_REGION_DISABLED_REGIONS = (
    "visual",
    "audio",
    "emotion",
    "pattern",
    "integration",
    "language",
    "executive",
)
SEED_WITHIN_REGION_DELAY_RANGE_BY_REGION = {
    # Preserve a recurrent scaffold only where longer-horizon integration is structural.
    "memory_long": (9, 10),
}
SEED_WITHIN_REGION_WEIGHT_RANGE_BY_REGION = {
    # Restore a latent same-region scaffold in the remaining hot regions without
    # reintroducing the original high-gain recurrence.
    "emotion": (0.05, 0.05),
    "pattern": (0.05, 0.05),
    "integration": (0.05, 0.05),
}
SEED_PHYSICS_TRACES = 200
SEED_RELATIONAL_TRACES = 200
SEED_TRACE_BUCKET_JITTER_FRACTION = 0.35
SEED_WITHIN_REGION_LOCAL_WINDOW_FRACTION = 0.05
SEED_MIN_LOCAL_WINDOW = 64

# === SIGNAL FLOW (cross-region synapse routing) ===
# Defines which regions connect to which, following the architecture diagram.
# (source_region, target_region)
SIGNAL_FLOW_CONNECTIONS = [
    # Input → Attention (gain modulation, not direct synapses — but pattern gets fed)
    # Input → Pattern Recognition
    ("sensory",     "pattern"),
    ("visual",      "pattern"),
    ("audio",       "pattern"),
    # Pattern → Integration / Executive
    ("pattern",     "integration"),
    ("pattern",     "executive"),
    # Integration → Memory / Emotion / Language / Executive
    ("integration", "memory_short"),
    ("integration", "emotion"),
    ("integration", "language"),
    ("integration", "executive"),
    # Memory interactions
    ("memory_short", "memory_long"),
    ("memory_long",  "memory_short"),
    # Emotion → Executive, Attention
    ("emotion",     "executive"),
    ("emotion",     "attention"),
    # Language ↔ Executive (reasoning loop)
    ("language",    "executive"),
    ("executive",   "language"),
    # Executive → Motor / Speech
    ("executive",   "motor"),
    ("executive",   "speech"),
    # Language → Speech
    ("language",    "speech"),
    # Attention → (modulates all, but has direct connections to pattern)
    ("attention",   "pattern"),
    # Memory → Visual (imagination pathway)
    ("memory_long", "visual"),
    # Sensory → Motor (reflex shortcut — handled separately with low delay)
]
