/*
 * Copyright (c) 2026 FSO Network Simulator Project
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

// ns3-ai Gym environment over the Phase 3 FSO mesh (see sim/README.md).
//
// Topology (--topology):
//   pentagon (default): 5 nodes on an 800 m radius pentagon, 7 FSO links
//     (ring plus the 0-2 and 1-3 cross links). Candidate routes share links,
//     so a fade epoch on one link degrades several routes at once (the
//     Phase 6 finding).
//   disjoint (Phase 7a): 5 nodes, same 7-link / 4-route budget, but the four
//     0->3 routes are pairwise link-disjoint: the direct link 0-3 plus three
//     2-hop relays 0-1-3, 0-2-3, 0-4-3. Node 0 sits at the origin, node 3 at
//     (D, 0, 0) with D = disjointDirectM; relays 1/2/4 sit halfway across at
//     offset h = sqrt(R^2 - (D/2)^2) (R = disjointRelayM per-hop length):
//     1 at (D/2, +h, 0), 2 at (D/2, -h, 0), 4 at (D/2, 0, +h) (an elevated
//     relay). The direct path is deliberately LONGER than each relay hop:
//     Rytov variance grows as d^(11/6), so at C2n = 1e-13 the 1.5 km direct
//     link fades much deeper (sigma_R^2 ~ 4.2) than a 0.9 km relay hop
//     (~1.6), which is what lets one 2-hop relay route compete with the
//     1-hop direct route instead of being dominated a priori.
//
// One flow runs from node 0 to node 3 (--trafficProtocol udp | tcp; UDP is
// a constant-rate OnOff flow, TCP an OnOff flow over TcpSocketFactory into
// a PacketSink, segment size pinned to packetSize so drops-per-byte stay
// comparable). Every stepTime the environment publishes a per-link
// observation to the Python agent and applies the agent's route choice via
// Ipv4StaticRouting host routes (for TCP, reverse host routes for the ACK
// stream follow the same path).
//
// Observation (Box, double, shape [numLinks * 4] = [28]; with
// --routeInObs=true a one-hot of the currently installed route is appended,
// shape [numLinks * 4 + numRoutes] = [32]), for link i:
//   [4i+0] snrMarginDb  mean SNR margin, TxPowerDbm - extinctionDb(d) - NoiseDbm
//   [4i+1] linkPer      current packet error rate of the link (mean of the
//                       two directions' RateErrorModel rates, i.e. the fading
//                       bridge's latest channel state). Unlike an empirical
//                       drop rate this is defined for links carrying no
//                       traffic, so the agent can see off-route link quality
//                       (physically: FSO transceivers track beacon power per
//                       link continuously). Under correlated fading (positive
//                       coherence times) it predicts the link's near future.
//   [4i+2] scintIndex   1/alpha + 1/beta + 1/(alpha*beta) at (C2n, d)
//   [4i+3] queuePkts    packets queued in the two device TX queues
//
// Route one-hot (--routeInObs, Phase 10): obs[4*numLinks + r] is 1.0 for the
// route installed while this step's channel state was measured — the route
// the agent currently HOLDS at decision time — and 0.0 elsewhere. At episode
// start it marks the initial route (0). Phases 7-9 showed every learned
// policy fails to express the greedy-PER teacher's hysteresis because
// hold-vs-switch is indistinguishable without this state; the flag defaults
// to false so all earlier studies stay reproducible.
//
// Action (Discrete(4)): route for the 0->3 flow,
//   pentagon: 0: 0-2-3    1: 0-1-3    2: 0-4-3    3: 0-1-2-3
//   disjoint: 0: 0-3      1: 0-1-3    2: 0-2-3    3: 0-4-3
//
// Reward per step (UDP):
//   - dropWeight   * phyDrops          (packets lost to fading, all links)
//   - delayWeight  * meanDelayMs       (mean e2e delay of packets delivered
//                                       this step; 0 if none delivered)
//   - flapPenalty  * routeChanged      (1 if last action switched routes)
//   - energyWeight * hops * txPackets  (energy proxy: every packet sent on an
//                                       h-hop route costs ~h laser transmissions)
//
// Under TCP the drops term is swapped for a goodput-relative term:
//   - goodputWeight * (offeredPkts - deliveredPkts)
// where offeredPkts = trafficRate * stepTime / (8 * packetSize) and
// deliveredPkts = sink-received bytes this step / packetSize. Raw PHY drops
// are the wrong cost under TCP (a drop is retransmitted, its real price is
// paid in stalled goodput), so the reward charges the goodput shortfall
// against the offered load instead. The term is deliberately not clamped at
// zero: over an episode it telescopes to (total offered - total delivered),
// i.e. exactly the payload TCP failed to deliver, and post-fade catch-up
// bursts (goodput above the offered rate while the backlog drains) earn the
// shortfall back. Delay/flap/energy terms are unchanged; ACK-stream
// transmissions are not billed by the energy proxy (mirrors the AODV
// baseline's honest-differences note).

#include "ns3/ai-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/fso-topology-helper.h"
#include "ns3/gamma-gamma-fso-loss-model.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/queue.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"

#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FsoRlEnv");

/**
 * \brief Per-link bookkeeping for observations.
 */
struct FsoLinkRecord
{
    uint32_t nodeA;                     //!< First endpoint node id
    uint32_t nodeB;                     //!< Second endpoint node id
    Ptr<PointToPointNetDevice> devA;    //!< Device on nodeA
    Ptr<PointToPointNetDevice> devB;    //!< Device on nodeB
    Ptr<RateErrorModel> errA;           //!< Error model on devA (B->A PER)
    Ptr<RateErrorModel> errB;           //!< Error model on devB (A->B PER)
    Ipv4Address addrA;                  //!< nodeA's address on this link
    Ipv4Address addrB;                  //!< nodeB's address on this link
    uint32_t ifA;                       //!< nodeA's interface index
    uint32_t ifB;                       //!< nodeB's interface index
    double snrMarginDb;                 //!< Mean SNR margin [dB]
    double scintIndex;                  //!< Scintillation index at (C2n, d)
    uint64_t rxOk{0};                   //!< Cumulative PhyRxEnd, both directions
    uint64_t rxDrop{0};                 //!< Cumulative PhyRxDrop, both directions
    uint64_t prevRxOk{0};               //!< rxOk at the previous step
    uint64_t prevRxDrop{0};             //!< rxDrop at the previous step
};

/**
 * \brief Retransmission counter fed by a TcpSocketBase "Tx" trace.
 */
struct TcpRetxCounter
{
    uint64_t retx{0};             //!< Data segments sent below the high-water mark
    SequenceNumber32 highTx{0};   //!< Highest sequence number sent so far
};

/**
 * \brief Scalar knobs shared between main() and the environment.
 */
struct FsoRlEnvConfig
{
    Time stepTime;              //!< Agent decision interval
    uint32_t episodeSteps;      //!< Number of decision steps per episode
    double dropWeight;          //!< Reward weight per dropped packet (UDP)
    double delayWeight;         //!< Reward weight per ms of mean delivery delay
    double flapPenalty;         //!< Reward penalty for switching routes
    double energyWeight;        //!< Reward weight per packet-hop transmitted
    bool tcp{false};            //!< Whether the flow runs over TCP
    double goodputWeight{1.0};  //!< Reward weight per undelivered packet (TCP)
    double offeredPktsPerStep{0.0}; //!< Offered application packets per step
    uint32_t packetSize{1024};  //!< Application payload size [bytes]
    uint16_t flowPort{9000};    //!< Destination port identifying the flow
    bool routeInObs{false};     //!< Append a current-route one-hot to the obs
};

/**
 * \brief Gym environment: per-link FSO state in, next-hop route choice out.
 */
class FsoRlEnv : public OpenGymEnv
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * \brief Wire the environment to the simulated mesh.
     *
     * \param nodes the mesh nodes
     * \param links per-link records (trace counters are updated externally)
     * \param routes candidate routes, each a node-id sequence from src to dst
     * \param dstAddr flow destination address (host routes point here)
     * \param srcAddr flow source address (TCP only: reverse host routes for
     *        the ACK stream point here; pass Ipv4Address() for UDP)
     * \param monitor flow monitor tracking the src->dst flow
     * \param classifier flow classifier used to isolate the src->dst flow
     * \param sink packet sink of the flow (goodput source; may be null)
     * \param retx TCP retransmission counter (null for UDP)
     * \param config scalar configuration
     */
    void Setup(NodeContainer nodes,
               std::vector<FsoLinkRecord>* links,
               std::vector<std::vector<uint32_t>> routes,
               Ipv4Address dstAddr,
               Ipv4Address srcAddr,
               Ptr<FlowMonitor> monitor,
               Ptr<Ipv4FlowClassifier> classifier,
               Ptr<PacketSink> sink,
               TcpRetxCounter* retx,
               const FsoRlEnvConfig& config);

    /**
     * \brief Install the initial route (action 0) and schedule the first step.
     */
    void Start();

    // OpenGym interface
    Ptr<OpenGymSpace> GetActionSpace() override;
    Ptr<OpenGymSpace> GetObservationSpace() override;
    bool GetGameOver() override;
    Ptr<OpenGymDataContainer> GetObservation() override;
    float GetReward() override;
    std::string GetExtraInfo() override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

  private:
    /**
     * \brief Per-step hook: refresh metrics, notify Python, execute action.
     */
    void Step();

    /**
     * \brief Snapshot per-link counters and flow stats; compute obs and reward.
     */
    void CollectStepMetrics();

    /**
     * \brief Point the flow's host routes along the given candidate route.
     *
     * For TCP (srcAddr set) reverse host routes to the source address are
     * installed along the same path so the ACK stream follows the flow.
     *
     * \param routeIndex index into the candidate route set
     */
    void ApplyRoute(uint32_t routeIndex);

    /**
     * \brief Chain host routes to \p dst hop by hop along \p path.
     * \param path node-id sequence to route along
     * \param dst destination address of the host routes
     */
    void InstallHostRoutes(const std::vector<uint32_t>& path, Ipv4Address dst);

    NodeContainer m_nodes;                        //!< Mesh nodes
    std::vector<FsoLinkRecord>* m_links{nullptr}; //!< Link records (owned by main)
    std::vector<std::vector<uint32_t>> m_routes;  //!< Candidate routes
    Ipv4Address m_dstAddr;                        //!< Flow destination
    Ipv4Address m_srcAddr;                        //!< Flow source (TCP reverse routes)
    Ptr<FlowMonitor> m_monitor;                   //!< Flow statistics source
    Ptr<Ipv4FlowClassifier> m_classifier;         //!< Flow tuple lookup
    Ptr<PacketSink> m_sink;                       //!< Flow sink (goodput source)
    TcpRetxCounter* m_retx{nullptr};              //!< TCP retransmission counter
    FsoRlEnvConfig m_config;                      //!< Scalar configuration

    uint32_t m_stepCount{0};      //!< Steps taken so far
    uint32_t m_currentRoute{0};   //!< Active route index
    bool m_flapped{false};        //!< Route switched since last reward
    uint64_t m_prevFlowTx{0};     //!< Flow tx packets at previous step
    uint64_t m_prevFlowRx{0};     //!< Flow rx packets at previous step
    Time m_prevDelaySum;          //!< Flow delay sum at previous step
    uint64_t m_prevSinkBytes{0};  //!< Sink rx bytes at previous step
    uint64_t m_prevRetx{0};       //!< Retransmission count at previous step
    std::vector<double> m_obs;    //!< Latest observation vector
    float m_reward{0.0};          //!< Latest reward
    std::string m_info;           //!< Latest extra info
};

NS_OBJECT_ENSURE_REGISTERED(FsoRlEnv);

TypeId
FsoRlEnv::GetTypeId()
{
    static TypeId tid = TypeId("ns3::FsoRlEnv")
                            .SetParent<OpenGymEnv>()
                            .SetGroupName("FsoRlRouter")
                            .AddConstructor<FsoRlEnv>();
    return tid;
}

void
FsoRlEnv::Setup(NodeContainer nodes,
                std::vector<FsoLinkRecord>* links,
                std::vector<std::vector<uint32_t>> routes,
                Ipv4Address dstAddr,
                Ipv4Address srcAddr,
                Ptr<FlowMonitor> monitor,
                Ptr<Ipv4FlowClassifier> classifier,
                Ptr<PacketSink> sink,
                TcpRetxCounter* retx,
                const FsoRlEnvConfig& config)
{
    m_nodes = nodes;
    m_links = links;
    m_routes = std::move(routes);
    m_dstAddr = dstAddr;
    m_srcAddr = srcAddr;
    m_monitor = monitor;
    m_classifier = classifier;
    m_sink = sink;
    m_retx = retx;
    m_config = config;
    m_obs.assign(m_links->size() * 4 + (m_config.routeInObs ? m_routes.size() : 0), 0.0);
    if (m_config.routeInObs)
    {
        // Before the first step the one-hot marks the initial route
        m_obs[m_links->size() * 4 + m_currentRoute] = 1.0;
    }
}

void
FsoRlEnv::Start()
{
    ApplyRoute(m_currentRoute);
    Simulator::Schedule(m_config.stepTime, &FsoRlEnv::Step, this);
}

Ptr<OpenGymSpace>
FsoRlEnv::GetActionSpace()
{
    return CreateObject<OpenGymDiscreteSpace>(static_cast<int>(m_routes.size()));
}

Ptr<OpenGymSpace>
FsoRlEnv::GetObservationSpace()
{
    std::vector<uint32_t> shape = {static_cast<uint32_t>(m_obs.size())};
    return CreateObject<OpenGymBoxSpace>(-1e6, 1e6, shape, TypeNameGet<double>());
}

bool
FsoRlEnv::GetGameOver()
{
    // Episode end is signalled through OpenGymInterface::NotifySimulationEnd()
    return false;
}

Ptr<OpenGymDataContainer>
FsoRlEnv::GetObservation()
{
    std::vector<uint32_t> shape = {static_cast<uint32_t>(m_obs.size())};
    auto box = CreateObject<OpenGymBoxContainer<double>>(shape);
    box->SetData(m_obs);
    return box;
}

float
FsoRlEnv::GetReward()
{
    return m_reward;
}

std::string
FsoRlEnv::GetExtraInfo()
{
    return m_info;
}

bool
FsoRlEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    auto discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    NS_ABORT_MSG_IF(!discrete, "FsoRlEnv expects a discrete action");
    uint32_t route = discrete->GetValue();
    NS_ABORT_MSG_IF(route >= m_routes.size(), "route index out of range");
    if (route != m_currentRoute)
    {
        ApplyRoute(route);
        m_currentRoute = route;
        m_flapped = true;
    }
    return true;
}

void
FsoRlEnv::Step()
{
    m_stepCount++;
    CollectStepMetrics();
    Notify();
    if (m_stepCount < m_config.episodeSteps)
    {
        Simulator::Schedule(m_config.stepTime, &FsoRlEnv::Step, this);
    }
}

void
FsoRlEnv::CollectStepMetrics()
{
    uint64_t dropDelta = 0;
    for (std::size_t i = 0; i < m_links->size(); i++)
    {
        FsoLinkRecord& link = (*m_links)[i];
        uint64_t drop = link.rxDrop - link.prevRxDrop;
        link.prevRxOk = link.rxOk;
        link.prevRxDrop = link.rxDrop;
        dropDelta += drop;

        DoubleValue perBtoA;
        DoubleValue perAtoB;
        link.errA->GetAttribute("ErrorRate", perBtoA);
        link.errB->GetAttribute("ErrorRate", perAtoB);
        double linkPer = 0.5 * (perBtoA.Get() + perAtoB.Get());
        uint32_t queued =
            link.devA->GetQueue()->GetNPackets() + link.devB->GetQueue()->GetNPackets();

        m_obs[4 * i + 0] = link.snrMarginDb;
        m_obs[4 * i + 1] = linkPer;
        m_obs[4 * i + 2] = link.scintIndex;
        m_obs[4 * i + 3] = queued;
    }

    if (m_config.routeInObs)
    {
        // One-hot of the route held while this step's state was measured;
        // ExecuteActions runs after Notify, so this is the pre-decision route
        std::size_t base = m_links->size() * 4;
        for (std::size_t r = 0; r < m_routes.size(); r++)
        {
            m_obs[base + r] = (r == m_currentRoute) ? 1.0 : 0.0;
        }
    }

    uint64_t flowTx = 0;
    uint64_t flowRx = 0;
    Time delaySum;
    for (const auto& [flowId, stats] : m_monitor->GetFlowStats())
    {
        // Skip the reverse ACK flow under TCP (ephemeral destination port)
        if (m_classifier->FindFlow(flowId).destinationPort != m_config.flowPort)
        {
            continue;
        }
        flowTx += stats.txPackets;
        flowRx += stats.rxPackets;
        delaySum += stats.delaySum;
    }
    uint64_t txDelta = flowTx - m_prevFlowTx;
    uint64_t rxDelta = flowRx - m_prevFlowRx;
    double meanDelayMs =
        rxDelta > 0 ? (delaySum - m_prevDelaySum).GetSeconds() * 1e3 / double(rxDelta) : 0.0;
    m_prevFlowTx = flowTx;
    m_prevFlowRx = flowRx;
    m_prevDelaySum = delaySum;

    uint64_t sinkBytes = m_sink ? m_sink->GetTotalRx() : 0;
    uint64_t sinkBytesDelta = sinkBytes - m_prevSinkBytes;
    m_prevSinkBytes = sinkBytes;
    double deliveredPkts = double(sinkBytesDelta) / double(m_config.packetSize);
    uint64_t retxTotal = m_retx ? m_retx->retx : 0;
    uint64_t retxDelta = retxTotal - m_prevRetx;
    m_prevRetx = retxTotal;

    double lossTerm = m_config.tcp
                          ? m_config.goodputWeight * (m_config.offeredPktsPerStep - deliveredPkts)
                          : m_config.dropWeight * double(dropDelta);
    double hops = static_cast<double>(m_routes[m_currentRoute].size() - 1);
    m_reward = static_cast<float>(-lossTerm - m_config.delayWeight * meanDelayMs -
                                  m_config.flapPenalty * (m_flapped ? 1.0 : 0.0) -
                                  m_config.energyWeight * hops * double(txDelta));
    m_flapped = false;

    std::ostringstream info;
    info << "step=" << m_stepCount << " route=" << m_currentRoute << " drops=" << dropDelta
         << " txPkts=" << txDelta << " rxPkts=" << rxDelta << " meanDelayMs=" << meanDelayMs;
    if (m_config.tcp)
    {
        double goodputMbps = double(sinkBytesDelta) * 8.0 / m_config.stepTime.GetSeconds() / 1e6;
        info << " goodputMbps=" << goodputMbps << " sinkPkts=" << deliveredPkts
             << " retx=" << retxDelta;
    }
    m_info = info.str();
    NS_LOG_INFO(m_info << " reward=" << m_reward);
}

void
FsoRlEnv::ApplyRoute(uint32_t routeIndex)
{
    Ipv4StaticRoutingHelper helper;

    // Drop every stale host route to the flow endpoints
    for (uint32_t n = 0; n < m_nodes.GetN(); n++)
    {
        Ptr<Ipv4StaticRouting> routing =
            helper.GetStaticRouting(m_nodes.Get(n)->GetObject<Ipv4>());
        for (int32_t r = static_cast<int32_t>(routing->GetNRoutes()) - 1; r >= 0; r--)
        {
            Ipv4Address dest = routing->GetRoute(r).GetDest();
            if (dest == m_dstAddr || (m_srcAddr != Ipv4Address() && dest == m_srcAddr))
            {
                routing->RemoveRoute(r);
            }
        }
    }

    const std::vector<uint32_t>& path = m_routes[routeIndex];
    InstallHostRoutes(path, m_dstAddr);
    if (m_srcAddr != Ipv4Address())
    {
        // TCP: route the ACK stream back along the same path
        std::vector<uint32_t> reversed(path.rbegin(), path.rend());
        InstallHostRoutes(reversed, m_srcAddr);
    }
    NS_LOG_INFO("route " << routeIndex << " installed");
}

void
FsoRlEnv::InstallHostRoutes(const std::vector<uint32_t>& path, Ipv4Address dst)
{
    Ipv4StaticRoutingHelper helper;

    // Chain host routes hop by hop along the given path
    for (std::size_t h = 0; h + 1 < path.size(); h++)
    {
        uint32_t u = path[h];
        uint32_t v = path[h + 1];
        const FsoLinkRecord* link = nullptr;
        for (const auto& candidate : *m_links)
        {
            if ((candidate.nodeA == u && candidate.nodeB == v) ||
                (candidate.nodeA == v && candidate.nodeB == u))
            {
                link = &candidate;
                break;
            }
        }
        NS_ABORT_MSG_IF(!link, "no FSO link between nodes " << u << " and " << v);

        Ipv4Address nextHop = (link->nodeA == v) ? link->addrA : link->addrB;
        uint32_t ifIndex = (link->nodeA == u) ? link->ifA : link->ifB;
        Ptr<Ipv4StaticRouting> routing =
            helper.GetStaticRouting(m_nodes.Get(u)->GetObject<Ipv4>());
        routing->AddHostRouteTo(dst, nextHop, ifIndex);
    }
}

/**
 * \brief Trace sink counting successful receptions on a link.
 * \param link the link record
 * \param packet the received packet (unused)
 */
static void
LinkRxOkTrace(FsoLinkRecord* link, Ptr<const Packet> packet)
{
    link->rxOk++;
}

/**
 * \brief Trace sink counting fading-corrupted receptions on a link.
 * \param link the link record
 * \param packet the dropped packet (unused)
 */
static void
LinkRxDropTrace(FsoLinkRecord* link, Ptr<const Packet> packet)
{
    link->rxDrop++;
}

/**
 * \brief Trace sink counting TCP data-segment retransmissions.
 *
 * A data segment whose sequence number falls below the connection's
 * high-water mark is a retransmission.
 *
 * \param counter the retransmission counter
 * \param packet the transmitted segment payload
 * \param header the TCP header
 * \param socket the transmitting socket (unused)
 */
static void
TcpTxTrace(TcpRetxCounter* counter,
           Ptr<const Packet> packet,
           const TcpHeader& header,
           Ptr<const TcpSocketBase> socket)
{
    if (packet->GetSize() == 0)
    {
        return; // SYN/FIN/pure ACK
    }
    if (header.GetSequenceNumber() < counter->highTx)
    {
        counter->retx++;
    }
    else
    {
        counter->highTx = header.GetSequenceNumber() + packet->GetSize();
    }
}

/**
 * \brief Attach TcpTxTrace to the OnOff application's socket.
 *
 * Must run shortly after the application starts: the socket does not exist
 * before StartApplication.
 *
 * \param app the OnOff application driving the flow
 * \param counter the retransmission counter to feed
 */
static void
ConnectTcpRetxTrace(Ptr<OnOffApplication> app, TcpRetxCounter* counter)
{
    Ptr<Socket> socket = app->GetSocket();
    NS_ABORT_MSG_IF(!socket, "OnOff socket not created yet");
    socket->TraceConnectWithoutContext("Tx", MakeBoundCallback(&TcpTxTrace, counter));
}

int
main(int argc, char* argv[])
{
    double c2n = 1e-15;
    uint32_t episodeSteps = 100;
    double stepTime = 0.1;
    double updateIntervalMs = 1.0;
    Time coherenceLarge = Seconds(0);
    Time coherenceSmall = Seconds(0);
    double txPowerDbm = 10.0;
    double noiseDbm = -8.0;
    double wavelength = 1550e-9;
    double extinction = 1e-5;
    double meshRadius = 800.0;
    std::string topology = "pentagon";
    double disjointDirectM = 1500.0;
    double disjointRelayM = 900.0;
    std::string dataRate = "100Mbps";
    std::string trafficProtocol = "udp";
    std::string trafficRate = "2Mbps";
    uint32_t packetSize = 1024;
    double dropWeight = 1.0;
    double delayWeight = 0.1;
    double flapPenalty = 5.0;
    double energyWeight = 0.01;
    double goodputWeight = 1.0;
    bool routeInObs = false;
    uint32_t simSeed = 1;

    CommandLine cmd(__FILE__);
    cmd.AddValue("c2n", "Refractive index structure parameter [m^-2/3]", c2n);
    cmd.AddValue("episodeSteps", "Decision steps per episode", episodeSteps);
    cmd.AddValue("stepTime", "Agent decision interval [s]", stepTime);
    cmd.AddValue("updateIntervalMs", "Fading refresh period [ms]", updateIntervalMs);
    cmd.AddValue("coherenceLarge",
                 "Large-scale fading coherence time, e.g. 100ms (0 = i.i.d.)",
                 coherenceLarge);
    cmd.AddValue("coherenceSmall",
                 "Small-scale fading coherence time, e.g. 10ms (0 = i.i.d.)",
                 coherenceSmall);
    cmd.AddValue("txPowerDbm", "Transmit optical power [dBm]", txPowerDbm);
    cmd.AddValue("noiseDbm", "Receiver noise-equivalent power [dBm]", noiseDbm);
    cmd.AddValue("wavelength", "Optical wavelength [m]", wavelength);
    cmd.AddValue("extinction", "Beer-Lambert extinction coefficient [1/m]", extinction);
    cmd.AddValue("meshRadius", "Pentagon circumradius [m]", meshRadius);
    cmd.AddValue("topology", "Mesh layout: pentagon | disjoint", topology);
    cmd.AddValue("disjointDirectM", "Disjoint topology: 0-3 direct link length [m]",
                 disjointDirectM);
    cmd.AddValue("disjointRelayM", "Disjoint topology: per-hop relay link length [m]",
                 disjointRelayM);
    cmd.AddValue("dataRate", "FSO link data rate", dataRate);
    cmd.AddValue("trafficProtocol", "Transport of the 0->3 flow: udp | tcp", trafficProtocol);
    cmd.AddValue("trafficRate", "Offered load of the 0->3 flow", trafficRate);
    cmd.AddValue("packetSize", "Application payload size [bytes]", packetSize);
    cmd.AddValue("dropWeight", "Reward weight per dropped packet (udp)", dropWeight);
    cmd.AddValue("delayWeight", "Reward weight per ms mean delay", delayWeight);
    cmd.AddValue("flapPenalty", "Reward penalty for switching routes", flapPenalty);
    cmd.AddValue("energyWeight", "Reward weight per packet-hop sent", energyWeight);
    cmd.AddValue("goodputWeight", "Reward weight per undelivered packet (tcp)", goodputWeight);
    cmd.AddValue("routeInObs",
                 "Append a one-hot of the current route to the observation",
                 routeInObs);
    cmd.AddValue("simSeed", "Run number for the RNG", simSeed);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(topology != "pentagon" && topology != "disjoint",
                    "unknown topology '" << topology << "' (pentagon | disjoint)");
    NS_ABORT_MSG_IF(trafficProtocol != "udp" && trafficProtocol != "tcp",
                    "unknown trafficProtocol '" << trafficProtocol << "' (udp | tcp)");
    const bool tcp = trafficProtocol == "tcp";

    // Must exist before any Python interaction
    Ptr<OpenGymInterface> openGymInterface = OpenGymInterface::Get();

    SeedManager::SetSeed(1);
    SeedManager::SetRun(simSeed);

    NodeContainer nodes;
    nodes.Create(5);

    auto positions = CreateObject<ListPositionAllocator>();
    if (topology == "pentagon")
    {
        for (uint32_t i = 0; i < 5; i++)
        {
            double angle = 2.0 * M_PI * i / 5.0;
            positions->Add(
                Vector(meshRadius * std::cos(angle), meshRadius * std::sin(angle), 0.0));
        }
    }
    else
    {
        // Disjoint geometry (see the header comment): 0 and 3 are the flow
        // endpoints D metres apart; relays 1/2/4 sit halfway across at offset
        // h so each relay hop is exactly disjointRelayM long. The direct link
        // is longer than a relay hop on purpose — d^(11/6) Rytov scaling makes
        // it fade deeper, so the 2-hop relays genuinely compete with it.
        NS_ABORT_MSG_IF(disjointRelayM <= disjointDirectM / 2.0,
                        "disjointRelayM must exceed disjointDirectM/2");
        double half = disjointDirectM / 2.0;
        double h = std::sqrt(disjointRelayM * disjointRelayM - half * half);
        positions->Add(Vector(0.0, 0.0, 0.0));              // 0: source
        positions->Add(Vector(half, h, 0.0));               // 1: relay
        positions->Add(Vector(half, -h, 0.0));              // 2: relay
        positions->Add(Vector(disjointDirectM, 0.0, 0.0));  // 3: destination
        positions->Add(Vector(half, 0.0, h));               // 4: elevated relay
    }
    MobilityHelper mobility;
    mobility.SetPositionAllocator(positions);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    // Routing is fully agent-controlled: static host routes only
    InternetStackHelper internet;
    Ipv4StaticRoutingHelper staticRouting;
    internet.SetRoutingHelper(staticRouting);
    internet.Install(nodes);

    FsoTopologyHelper fso;
    fso.SetDeviceAttribute("DataRate", StringValue(dataRate));
    fso.SetChannelAttribute("Delay", StringValue("5us"));
    fso.SetLossModelAttribute("C2n", DoubleValue(c2n));
    fso.SetLossModelAttribute("Wavelength", DoubleValue(wavelength));
    fso.SetLossModelAttribute("ExtinctionCoefficient", DoubleValue(extinction));
    fso.SetLossModelAttribute("CoherenceTimeLargeScale", TimeValue(coherenceLarge));
    fso.SetLossModelAttribute("CoherenceTimeSmallScale", TimeValue(coherenceSmall));
    fso.SetLinkAttribute("TxPowerDbm", DoubleValue(txPowerDbm));
    fso.SetLinkAttribute("NoiseDbm", DoubleValue(noiseDbm));
    fso.SetLinkAttribute("UpdateInterval", TimeValue(MilliSeconds(updateIntervalMs)));
    fso.SetLinkAttribute("PacketSize", UintegerValue(packetSize));

    // Shadow loss model for deterministic observation terms (margin, SI)
    auto shadowLoss = CreateObject<GammaGammaFsoLossModel>();
    shadowLoss->SetAttribute("C2n", DoubleValue(c2n));
    shadowLoss->SetAttribute("Wavelength", DoubleValue(wavelength));
    shadowLoss->SetAttribute("ExtinctionCoefficient", DoubleValue(extinction));

    // pentagon: ring plus two cross links; disjoint: the direct 0-3 link and
    // the three relay 2-hop paths' links (7 links either way)
    const std::vector<std::pair<uint32_t, uint32_t>> linkPairs =
        topology == "pentagon"
            ? std::vector<std::pair<uint32_t, uint32_t>>{{0, 1},
                                                         {1, 2},
                                                         {2, 3},
                                                         {3, 4},
                                                         {4, 0},
                                                         {0, 2},
                                                         {1, 3}}
            : std::vector<std::pair<uint32_t, uint32_t>>{{0, 3},
                                                         {0, 1},
                                                         {1, 3},
                                                         {0, 2},
                                                         {2, 3},
                                                         {0, 4},
                                                         {4, 3}};

    std::vector<FsoLinkRecord> links;
    links.reserve(linkPairs.size());
    Ipv4AddressHelper addresses;
    uint32_t subnet = 1;
    for (auto [i, j] : linkPairs)
    {
        NetDeviceContainer devices = fso.Install(nodes.Get(i), nodes.Get(j));
        std::ostringstream base;
        base << "10.1." << subnet++ << ".0";
        addresses.SetBase(base.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer ifaces = addresses.Assign(devices);

        FsoLinkRecord link;
        link.nodeA = i;
        link.nodeB = j;
        link.devA = DynamicCast<PointToPointNetDevice>(devices.Get(0));
        link.devB = DynamicCast<PointToPointNetDevice>(devices.Get(1));
        PointerValue errorModel;
        link.devA->GetAttribute("ReceiveErrorModel", errorModel);
        link.errA = errorModel.Get<RateErrorModel>();
        link.devB->GetAttribute("ReceiveErrorModel", errorModel);
        link.errB = errorModel.Get<RateErrorModel>();
        NS_ABORT_MSG_IF(!link.errA || !link.errB, "missing link error model");
        link.addrA = ifaces.GetAddress(0);
        link.addrB = ifaces.GetAddress(1);
        link.ifA = ifaces.Get(0).second;
        link.ifB = ifaces.Get(1).second;

        double distance = nodes.Get(i)->GetObject<MobilityModel>()->GetDistanceFrom(
            nodes.Get(j)->GetObject<MobilityModel>());
        link.snrMarginDb = txPowerDbm - shadowLoss->GetExtinctionLossDb(distance) - noiseDbm;
        auto [alpha, beta] = shadowLoss->GetAlphaBeta(distance);
        link.scintIndex = 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta);
        links.push_back(link);
    }
    fso.AssignStreams(100);

    // Counters must attach after the vector stops reallocating
    for (auto& link : links)
    {
        for (auto dev : {link.devA, link.devB})
        {
            dev->TraceConnectWithoutContext("PhyRxEnd",
                                            MakeBoundCallback(&LinkRxOkTrace, &link));
            dev->TraceConnectWithoutContext("PhyRxDrop",
                                            MakeBoundCallback(&LinkRxDropTrace, &link));
        }
    }

    // Candidate routes for the 0->3 flow, as node-id paths. The disjoint set
    // is pairwise link-disjoint (routes share only the endpoints), so a fade
    // epoch on one route leaves the alternatives genuinely clean.
    std::vector<std::vector<uint32_t>> routes =
        topology == "pentagon"
            ? std::vector<std::vector<uint32_t>>{{0, 2, 3}, {0, 1, 3}, {0, 4, 3}, {0, 1, 2, 3}}
            : std::vector<std::vector<uint32_t>>{{0, 3}, {0, 1, 3}, {0, 2, 3}, {0, 4, 3}};

    // Flow destination: a node-3 address (weak end-system model accepts it
    // on any interface). Pentagon keeps the 2-3 link address for backward
    // compatibility; disjoint uses the direct 0-3 link address.
    Ipv4Address dstAddr = topology == "pentagon" ? links[2].addrB : links[0].addrB;

    // Flow source address: node 0's address on the initial route's first
    // link — the address TCP binds to when connecting, and therefore the
    // destination of the reverse (ACK) host routes. Unused for UDP.
    Ipv4Address srcAddr;
    for (const auto& link : links)
    {
        if ((link.nodeA == routes[0][0] && link.nodeB == routes[0][1]) ||
            (link.nodeA == routes[0][1] && link.nodeB == routes[0][0]))
        {
            srcAddr = (link.nodeA == routes[0][0]) ? link.addrA : link.addrB;
            break;
        }
    }
    NS_ABORT_MSG_IF(tcp && srcAddr == Ipv4Address(), "no link on the initial route's first hop");

    const uint16_t port = 9000;
    const double episodeEnd = stepTime * episodeSteps;

    if (tcp)
    {
        // One TCP segment carries one application packet, keeping the
        // packet-mode error model's drops-per-byte comparable with UDP
        Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(packetSize));
    }
    const std::string socketFactory = tcp ? "ns3::TcpSocketFactory" : "ns3::UdpSocketFactory";

    OnOffHelper onOff(socketFactory, InetSocketAddress(dstAddr, port));
    onOff.SetConstantRate(DataRate(trafficRate), packetSize);
    ApplicationContainer apps = onOff.Install(nodes.Get(0));

    PacketSinkHelper sink(socketFactory, InetSocketAddress(Ipv4Address::GetAny(), port));
    apps.Add(sink.Install(nodes.Get(3)));
    apps.Start(Seconds(0.0));
    apps.Stop(Seconds(episodeEnd));
    Ptr<PacketSink> sinkApp = DynamicCast<PacketSink>(apps.Get(1));

    TcpRetxCounter retxCounter;
    if (tcp)
    {
        // The OnOff socket only exists after StartApplication (t = 0)
        Simulator::Schedule(MilliSeconds(1),
                            &ConnectTcpRetxTrace,
                            DynamicCast<OnOffApplication>(apps.Get(0)),
                            &retxCounter);
    }

    FlowMonitorHelper flowMonitorHelper;
    Ptr<FlowMonitor> flowMonitor = flowMonitorHelper.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowMonitorHelper.GetClassifier());

    FsoRlEnvConfig config;
    config.stepTime = Seconds(stepTime);
    config.episodeSteps = episodeSteps;
    config.dropWeight = dropWeight;
    config.delayWeight = delayWeight;
    config.flapPenalty = flapPenalty;
    config.energyWeight = energyWeight;
    config.tcp = tcp;
    config.goodputWeight = goodputWeight;
    config.offeredPktsPerStep =
        DataRate(trafficRate).GetBitRate() * stepTime / (8.0 * packetSize);
    config.packetSize = packetSize;
    config.flowPort = port;
    config.routeInObs = routeInObs;

    Ptr<FsoRlEnv> env = CreateObject<FsoRlEnv>();
    env->Setup(nodes,
               &links,
               routes,
               dstAddr,
               tcp ? srcAddr : Ipv4Address(),
               flowMonitor,
               classifier,
               sinkApp,
               tcp ? &retxCounter : nullptr,
               config);
    env->SetOpenGymInterface(openGymInterface);
    env->Start();

    Simulator::Stop(Seconds(episodeEnd) + MilliSeconds(1));
    Simulator::Run();

    openGymInterface->NotifySimulationEnd();
    Simulator::Destroy();
    return 0;
}
