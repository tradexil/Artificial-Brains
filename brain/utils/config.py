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
    "executive":    {"threshold": 0.60, "leak": 0.93, "refractory": 6, "inhibitory_pct": 0.30},
    "motor":        {"threshold": 0.55, "leak": 0.85, "refractory": 3, "inhibitory_pct": 0.20},
    "speech":       {"threshold": 0.50, "leak": 0.87, "refractory": 3, "inhibitory_pct": 0.20},
    "numbers":      {"threshold": 0.50, "leak": 0.96, "refractory": 4, "inhibitory_pct": 0.10},
}

# === LEARNING ===
HEBBIAN_RATE = 0.01
ANTI_HEBBIAN_RATE = 0.003
HEBBIAN_WINDOW = 3  # ticks for co-activation
NOVELTY_LEARNING_BOOST = 2.0
AROUSAL_LEARNING_BOOST = 1.5

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
TRACE_FORMATION_PERSISTENCE = 3
TRACE_FORMATION_MIN_REGIONS = 2
TRACE_MERGE_OVERLAP = 0.8

# === BINDINGS ===
BINDING_TEMPORAL_WINDOW = 5
BINDING_FORMATION_COUNT = 5
BINDING_DISSOLUTION_WEIGHT = 0.05
BINDING_DISSOLUTION_MIN_FIRES = 10

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

# === SIGNAL FLOW (cross-region synapse routing) ===
# Defines which regions connect to which, following the architecture diagram.
# (source_region, target_region)
SIGNAL_FLOW_CONNECTIONS = [
    # Input → Attention (gain modulation, not direct synapses — but pattern gets fed)
    # Input → Pattern Recognition
    ("sensory",     "pattern"),
    ("visual",      "pattern"),
    ("audio",       "pattern"),
    # Pattern → Integration
    ("pattern",     "integration"),
    # Integration → Memory / Emotion / Language
    ("integration", "memory_short"),
    ("integration", "emotion"),
    ("integration", "language"),
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
