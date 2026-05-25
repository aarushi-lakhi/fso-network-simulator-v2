# fso-rl-env — ns3-ai Gym environment for FSO routing

An ns-3 program exposing the Phase 3 FSO mesh (`ns3-fso-channel/`) to a
Python RL agent through ns3-ai's Gym interface (shared memory, no sockets).

## Environment definition

**Topology** (`topology`, default `pentagon`). Two layouts, both 5 nodes /
7 FSO links / 4 candidate routes, built with `FsoTopologyHelper`:

* `pentagon` — 5 nodes on an 800 m radius pentagon, 7 links (the ring
  `0-1-2-3-4-0` plus cross links `0-2` and `1-3`). Candidate routes share
  links and nodes, which is why Phase 6 found route switching never pays
  here: one fade epoch degrades several routes at once.
* `disjoint` (Phase 7a) — the four 0→3 routes are pairwise
  **link-disjoint** (they share only the endpoints): the direct link
  `0-3` plus three 2-hop relays `0-1-3`, `0-2-3`, `0-4-3`, i.e. links
  `(0,3) (0,1) (1,3) (0,2) (2,3) (0,4) (4,3)` in install order. Geometry:
  node 0 at the origin, node 3 at `(D, 0, 0)` with
  `D = disjointDirectM` (default 1500 m); relays sit halfway across at
  offset `h = sqrt(R² − (D/2)²)` where `R = disjointRelayM` (default
  900 m) is the per-hop relay link length — node 1 at `(D/2, +h, 0)`,
  node 2 at `(D/2, −h, 0)`, node 4 at `(D/2, 0, +h)` (an elevated
  relay). The direct link is deliberately *longer* than a relay hop:
  Rytov variance grows as `d^(11/6)`, so at `C2n = 1e-13` the 1.5 km
  direct link fades much deeper (σ²_R ≈ 4.2) than each 0.9 km relay hop
  (≈ 1.6). That is what makes the 1-hop direct route and the 2-hop
  relays genuinely compete instead of one dominating a priori — and
  because the routes share no links, their fade epochs are independent,
  so a well-timed switch actually has somewhere clean to go.

Every link's packet error rate follows Gamma-Gamma block fading at the
configured `C2n`. With `coherenceLarge` / `coherenceSmall` at their `0ms`
defaults successive fading blocks are i.i.d.; positive values make each
link direction's irradiance a temporally correlated Gamma-Gamma process
with those component coherence times (Phase 6, see
`ns3-fso-channel/model/correlated-gamma-gamma-fading.h`).

**Traffic** (`trafficProtocol`, default `udp`). One flow runs from node 0
to node 3 at `trafficRate` offered load: a constant-rate OnOff flow over
UDP, or over `TcpSocketFactory` into a `PacketSink` when
`trafficProtocol=tcp`. Under TCP the segment size is pinned to
`packetSize` so a segment carries one application packet (the PHY error
model is packet-mode — smaller segments would mean more drop
opportunities per byte), and route changes also re-install the reverse
host routes so the ACK stream follows the flow. TCP is the Phase 7a
lever on loss *shape*: PHY drops trigger retransmission and congestion
backoff, so losses compound and dodging a fade epoch pays non-linearly
through sustained goodput.

**Observation** — `Box(low=-1e6, high=1e6, shape=[28], float64)`, four
features per link, links in install order — pentagon
`(0,1) (1,2) (2,3) (3,4) (4,0) (0,2) (1,3)`, disjoint
`(0,3) (0,1) (1,3) (0,2) (2,3) (0,4) (4,3)`:

| offset | feature | meaning |
|---|---|---|
| `4i+0` | `snrMarginDb` | mean SNR margin: `TxPowerDbm − extinctionDb(d) − NoiseDbm` |
| `4i+1` | `linkPer` | current packet error rate of the link: mean of the two directions' `RateErrorModel` rates, i.e. the fading bridge's latest channel state. Defined for idle links too (physically: FSO transceivers track per-link beacon power continuously), so the agent can compare off-route link quality; under correlated fading it predicts the link's near future. (Until Phase 6 this slot held the empirical per-step drop rate, which was a `1.0` sentinel on idle links — invisible off-route state.) |
| `4i+2` | `scintIndex` | `1/α + 1/β + 1/(αβ)` from the loss model's α, β at the current `C2n` and link distance |
| `4i+3` | `queuePkts` | packets waiting in the two device TX queues |

**Action** — `Discrete(4)`: route of the 0→3 flow, applied as
`Ipv4StaticRouting` host routes on every node along the path (stale host
routes are removed first; under TCP the reverse ACK routes are chained
along the same path). Routes — pentagon: `0: 0-2-3`, `1: 0-1-3`,
`2: 0-4-3`, `3: 0-1-2-3`; disjoint: `0: 0-3`, `1: 0-1-3`, `2: 0-2-3`,
`3: 0-4-3`. The initial route is action 0.

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

Under `trafficProtocol=tcp` the drops term is swapped for a
goodput-relative term (the rest of the shape is unchanged):

```
  − goodputWeight · (offeredPkts − deliveredPkts)
```

with `offeredPkts = trafficRate · stepTime / (8 · packetSize)` and
`deliveredPkts = sink bytes this step / packetSize`. Rationale: under TCP
a PHY drop is retransmitted, so its true cost is the goodput the
connection fails to sustain, not the drop itself — charging the
shortfall against the offered load makes fade-dodging pay exactly when
it preserves goodput. The term is not clamped at zero: per episode it
telescopes to `total offered − total delivered` (the payload TCP failed
to deliver), and post-fade catch-up bursts above the offered rate earn
the shortfall back. With the default weights the term has the same
scale and units as the UDP drops term (packets per step). `txPackets`
in the energy term counts data segments including retransmissions; ACK
transmissions are not billed (same honest-differences spirit as the
AODV baseline). The TCP info string gains `goodputMbps`, `sinkPkts`,
and `retx` (retransmitted data segments this step, counted from the
socket's `Tx` trace).

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
