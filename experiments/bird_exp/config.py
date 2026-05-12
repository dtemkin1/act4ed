from formulation.bird_adapter import BirdAdapterConfig

config = BirdAdapterConfig(
    # ── Cohort / assignment mode ──────────────────────────────────────────────
    # Which student population to route.
    cohort="all",                   # "conventional"        – non-SPED, non-wheelchair students
    # cohort="sped_no_wheelchair",  # "sped_no_wheelchair"  – SPED students who are not wheelchair users
    # cohort="sped_and_wheelchair", # "sped_and_wheelchair" – all SPED and/or wheelchair students
    # cohort="all",                 # "all"                 – fleet-aware staged routing (wheelchair → SPED → conventional)

    # ── Bus selection ─────────────────────────────────────────────────────────
    # Homogeneous (old BiRD) mode – pick a single bus type for all routes:
    # bus_type="C",                 # "C"   – standard bus (no monitor / no wheelchair lift)
    # bus_type="B",                 # "B"   – monitor bus
    # bus_type="BWC",               # "BWC" – monitor + wheelchair lift
    # bus_type="WC",                # "WC"  – wheelchair lift only

    # Fleet-aware mode – uses concrete fleet buses/capacities/depots/monitor flags:
    fleet_aware=True,               # enable fleet-aware routing (default: False)
    # bus_type=None,                # None means use all bus types from the fleet
    conventional_spillover=False,   # allow conventional students onto remaining monitor buses (default: False)
    
    # ── Monitor Policy ─────────────────────────────────────────────────────────
    # monitor_policy="route_assigned",

    # ── Capacity / time constraints ───────────────────────────────────────────
    max_time_on_bus=60,             # max minutes a student may spend on the bus (default: 120)
    school_dwell_time=10,           # minutes the bus waits at the school after arriving (default: 0)
    earliest_arrival_buffer=40,     # bus must arrive ≥ N min before bell (default: None)
    # latest_arrival_buffer=10,     # bus must arrive ≤ N min before bell (default: None)

    # ── Stop dwell times ─────────────────────────────────────────────────────
    # constant_stop_time=0,                 # fixed dwell per stop in minutes (default: 0)
    stop_time_per_student=0.5,              # additional dwell per boarding student in minutes (default: 0.3)
    stop_time_per_sped=1.0,                 # additional dwell per boarding student in minutes (default: 0.3)
    stop_time_per_wheelchair_student=1.0,   # additional dwell per wheelchair student in minutes (default: 0)

    # ── Routing parameters ───────────────────────────────────────────────────
    bus_mph=20,                     # assumed bus travel speed (default: _DEFAULT_BUS_MPH)
    # lambda_value=1.0e4,           # trade-off weight between distance and ride time (default: _DEFAULT_BIRD_LAMBDA_VALUE)

    # ── Partial assignment ───────────────────────────────────────────────────
    allow_partial=True,             # allow some students to remain unassigned (default: False)

    # ── Optional stop reassignment ───────────────────────────────────────────
    # reassign_stops=True,          # re-optimise stop assignments before routing (default: False)
    # stop_assignment_lambda=1.0e4, # weight for stop reassignment optimisation (default: _DEFAULT_STOP_ASSIGNMENT_LAMBDA)
    # max_walking_distance_km=1.0,  # max walk distance when reassigning stops in km (default: None)
    
    # ── Optimization method ──────────────────────────────────────────────────
    # method="lbh",                 # fast greedy route construction
    method="scenario",              # scenario mode; fleet-aware currently delegates to concrete fleet path
)
