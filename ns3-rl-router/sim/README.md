# fso-rl-env — ns3-ai Gym environment for FSO routing

An ns-3 program exposing the Phase 3 FSO mesh (`ns3-fso-channel/`) to a
Python RL agent through ns3-ai's Gym interface (shared memory, no sockets).

## Environment definition

**Topology.** 5 nodes on an 800 m radius pentagon, 7 FSO links (the ring
`0-1-2-3-4-0` plus cross links `0-2` and `1-3`), built with
`FsoTopologyHelper`. Every link's packet error rate follows Gamma-Gamma
block fading at the configured `C2n`. With `coherenceLarge` /
`coherenceSmall` at their `0ms` defaults successive fading blocks are
i.i.d.; positive values make each link direction's irradiance a
temporally correlated Gamma-Gamma process with those component
coherence times (Phase 6, see
`ns3-fso-channel/model/correlated-gamma-gamma-fading.h`). One UDP flow
runs from node 0 to node 3.

**Observation** — `Box(low=-1e6, high=1e6, shape=[28], float64)`, four
features per link, links in install order
`(0,1) (1,2) (2,3) (3,4) (4,0) (0,2) (1,3)`:

| offset | feature | meaning |
|---|---|---|
| `4i+0` | `snrMarginDb` | mean SNR margin: `TxPowerDbm − extinctionDb(d) − NoiseDbm` |
| `4i+1` | `linkPer` | current packet error rate of the link: mean of the two directions' `RateErrorModel` rates, i.e. the fading bridge's latest channel state. Defined for idle links too (physically: FSO transceivers track per-link beacon power continuously), so the agent can compare off-route link quality; under correlated fading it predicts the link's near future. (Until Phase 6 this slot held the empirical per-step drop rate, which was a `1.0` sentinel on idle links — invisible off-route state.) |
| `4i+2` | `scintIndex` | `1/α + 1/β + 1/(αβ)` from the loss model's α, β at the current `C2n` and link distance |
| `4i+3` | `queuePkts` | packets waiting in the two device TX queues |

**Action** — `Discrete(4)`: route of the 0→3 flow, applied as
`Ipv4StaticRouting` host routes on every node along the path (stale host
routes are removed first). Routes: `0: 0-2-3`, `1: 0-1-3`, `2: 0-4-3`,
`3: 0-1-2-3`. The initial route is action 0.

**Reward** per decision step:

```
r = − dropWeight   · phyDrops          (PhyRxDrop total, all links)
    − delayWeight  · meanDelayMs       (mean e2e delay of packets delivered
                                        this step; 0 if none arrived)
    − flapPenalty  · routeChanged      (1 if the last action switched routes)
    − energyWeight · hops · txPackets  (energy proxy: a packet sent on an
                                        h-hop route costs ≈h laser
                                        transmissions; the plan's
                                        "energy_saved" realised as a
                                        per-transmission cost)
```

**Episode.** Fixed number of decision steps (`episodeSteps`, one per
`stepTime`); the ns-3 process then signals `done` and exits. `C2n`,
fading coherence times, episode length, decision interval, link budget
and reward weights are all command-line arguments (defaults mirror
`../config/sim_config.yaml`).

## Build mechanism

The directory is symlinked into ns-3's `scratch/` — ns-3's
`scratch/CMakeLists.txt` runs `add_subdirectory()` on any scratch subdir
containing a `CMakeLists.txt`, and ours builds the `fso-rl-env` executable
linked against the `ai` and `fso-channel` contrib modules. This avoids
turning `sim/` into an ns-3 module or editing ns3-ai's example lists.

Setup (idempotent):

```bash
./setup/link_fso_modules.sh     # symlinks fso-channel into contrib/,
                                # sim/ into scratch/, reconfigures + builds
```

## Smoke test

```bash
source ~/fso-tools/ns3ai-venv/bin/activate   # python3 must be the venv's 3.11
python check_env.py --steps 10
```

Prints the spaces, the reset observation and one transition per random
action, then exits with the ns-3 subprocess cleaned up.

Note for Python consumers: `ns3ai_gym_env` uses the `np.float`/`np.int`
aliases removed in NumPy ≥ 1.24, and its editable install does not expose
its top-level `messages_pb2`/`ns3ai_gym_msg_py` modules. Importing
`ns3ai_shim.py` (from this directory) applies both fixes; `check_env.py`
and `agent/ns3_env.py` use it.
