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
// Topology: 5 nodes on an 800 m radius pentagon, 7 FSO links (ring plus the
// 0-2 and 1-3 cross links), built with FsoTopologyHelper so every link's
// packet error rate tracks Gamma-Gamma fading at the configured C2n.
//
// One UDP flow runs from node 0 to node 3. Every stepTime the environment
// publishes a per-link observation to the Python agent and applies the
// agent's route choice via Ipv4StaticRouting host routes.
//
// Observation (Box, double, shape [numLinks * 4] = [28]), for link i:
//   [4i+0] snrMarginDb  mean SNR margin, TxPowerDbm - extinctionDb(d) - NoiseDbm
//   [4i+1] dropRate     PhyRxDrop / (PhyRxDrop + PhyRxEnd) over the last step,
//                       both directions combined (1.0 if nothing was received)
//   [4i+2] scintIndex   1/alpha + 1/beta + 1/(alpha*beta) at (C2n, d)
//   [4i+3] queuePkts    packets queued in the two device TX queues
//
// Action (Discrete(4)): route for the 0->3 flow,
//   0: 0-2-3    1: 0-1-3    2: 0-4-3    3: 0-1-2-3
//
// Reward per step:
//   - dropWeight   * phyDrops          (packets lost to fading, all links)
//   - delayWeight  * meanDelayMs       (mean e2e delay of packets delivered
//                                       this step; 0 if none delivered)
//   - flapPenalty  * routeChanged      (1 if last action switched routes)
//   - energyWeight * hops * txPackets  (energy proxy: every packet sent on an
//                                       h-hop route costs ~h laser transmissions)

#include "ns3/ai-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/fso-topology-helper.h"
#include "ns3/gamma-gamma-fso-loss-model.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/queue.h"

#include <cmath>
#include <sstream>
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
 * \brief Scalar knobs shared between main() and the environment.
 */
struct FsoRlEnvConfig
{
    Time stepTime;         //!< Agent decision interval
    uint32_t episodeSteps; //!< Number of decision steps per episode
    double dropWeight;     //!< Reward weight per dropped packet
    double delayWeight;    //!< Reward weight per ms of mean delivery delay
    double flapPenalty;    //!< Reward penalty for switching routes
    double energyWeight;   //!< Reward weight per packet-hop transmitted
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
     * \param monitor flow monitor tracking the src->dst flow
     * \param config scalar configuration
     */
    void Setup(NodeContainer nodes,
               std::vector<FsoLinkRecord>* links,
               std::vector<std::vector<uint32_t>> routes,
               Ipv4Address dstAddr,
               Ptr<FlowMonitor> monitor,
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
     * \param routeIndex index into the candidate route set
     */
    void ApplyRoute(uint32_t routeIndex);

    NodeContainer m_nodes;                        //!< Mesh nodes
    std::vector<FsoLinkRecord>* m_links{nullptr}; //!< Link records (owned by main)
    std::vector<std::vector<uint32_t>> m_routes;  //!< Candidate routes
    Ipv4Address m_dstAddr;                        //!< Flow destination
    Ptr<FlowMonitor> m_monitor;                   //!< Flow statistics source
    FsoRlEnvConfig m_config;                      //!< Scalar configuration

    uint32_t m_stepCount{0};      //!< Steps taken so far
    uint32_t m_currentRoute{0};   //!< Active route index
    bool m_flapped{false};        //!< Route switched since last reward
    uint64_t m_prevFlowTx{0};     //!< Flow tx packets at previous step
    uint64_t m_prevFlowRx{0};     //!< Flow rx packets at previous step
    Time m_prevDelaySum;          //!< Flow delay sum at previous step
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
                Ptr<FlowMonitor> monitor,
                const FsoRlEnvConfig& config)
{
    m_nodes = nodes;
    m_links = links;
    m_routes = std::move(routes);
    m_dstAddr = dstAddr;
    m_monitor = monitor;
    m_config = config;
    m_obs.assign(m_links->size() * 4, 0.0);
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
    std::vector<uint32_t> shape = {static_cast<uint32_t>(m_links->size() * 4)};
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
        uint64_t ok = link.rxOk - link.prevRxOk;
        uint64_t drop = link.rxDrop - link.prevRxDrop;
        link.prevRxOk = link.rxOk;
        link.prevRxDrop = link.rxDrop;
        dropDelta += drop;

        double dropRate = (ok + drop) > 0 ? double(drop) / double(ok + drop) : 1.0;
        uint32_t queued =
            link.devA->GetQueue()->GetNPackets() + link.devB->GetQueue()->GetNPackets();

        m_obs[4 * i + 0] = link.snrMarginDb;
        m_obs[4 * i + 1] = dropRate;
        m_obs[4 * i + 2] = link.scintIndex;
        m_obs[4 * i + 3] = queued;
    }

    uint64_t flowTx = 0;
    uint64_t flowRx = 0;
    Time delaySum;
    for (const auto& [flowId, stats] : m_monitor->GetFlowStats())
    {
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

    double hops = static_cast<double>(m_routes[m_currentRoute].size() - 1);
    m_reward = static_cast<float>(-m_config.dropWeight * double(dropDelta) -
                                  m_config.delayWeight * meanDelayMs -
                                  m_config.flapPenalty * (m_flapped ? 1.0 : 0.0) -
                                  m_config.energyWeight * hops * double(txDelta));
    m_flapped = false;

    std::ostringstream info;
    info << "step=" << m_stepCount << " route=" << m_currentRoute << " drops=" << dropDelta
         << " txPkts=" << txDelta << " rxPkts=" << rxDelta << " meanDelayMs=" << meanDelayMs;
    m_info = info.str();
    NS_LOG_INFO(m_info << " reward=" << m_reward);
}

void
FsoRlEnv::ApplyRoute(uint32_t routeIndex)
{
    Ipv4StaticRoutingHelper helper;

    // Drop every stale host route to the flow destination
    for (uint32_t n = 0; n < m_nodes.GetN(); n++)
    {
        Ptr<Ipv4StaticRouting> routing =
            helper.GetStaticRouting(m_nodes.Get(n)->GetObject<Ipv4>());
        for (int32_t r = static_cast<int32_t>(routing->GetNRoutes()) - 1; r >= 0; r--)
        {
            if (routing->GetRoute(r).GetDest() == m_dstAddr)
            {
                routing->RemoveRoute(r);
            }
        }
    }

    // Chain host routes hop by hop along the chosen path
    const std::vector<uint32_t>& path = m_routes[routeIndex];
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
        routing->AddHostRouteTo(m_dstAddr, nextHop, ifIndex);
    }
    NS_LOG_INFO("route " << routeIndex << " installed");
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

int
main(int argc, char* argv[])
{
    double c2n = 1e-15;
    uint32_t episodeSteps = 100;
    double stepTime = 0.1;
    double updateIntervalMs = 1.0;
    double txPowerDbm = 10.0;
    double noiseDbm = -8.0;
    double wavelength = 1550e-9;
    double extinction = 1e-5;
    double meshRadius = 800.0;
    std::string dataRate = "100Mbps";
    std::string trafficRate = "2Mbps";
    uint32_t packetSize = 1024;
    double dropWeight = 1.0;
    double delayWeight = 0.1;
    double flapPenalty = 5.0;
    double energyWeight = 0.01;
    uint32_t simSeed = 1;

    CommandLine cmd(__FILE__);
    cmd.AddValue("c2n", "Refractive index structure parameter [m^-2/3]", c2n);
    cmd.AddValue("episodeSteps", "Decision steps per episode", episodeSteps);
    cmd.AddValue("stepTime", "Agent decision interval [s]", stepTime);
    cmd.AddValue("updateIntervalMs", "Fading refresh period [ms]", updateIntervalMs);
    cmd.AddValue("txPowerDbm", "Transmit optical power [dBm]", txPowerDbm);
    cmd.AddValue("noiseDbm", "Receiver noise-equivalent power [dBm]", noiseDbm);
    cmd.AddValue("wavelength", "Optical wavelength [m]", wavelength);
    cmd.AddValue("extinction", "Beer-Lambert extinction coefficient [1/m]", extinction);
    cmd.AddValue("meshRadius", "Pentagon circumradius [m]", meshRadius);
    cmd.AddValue("dataRate", "FSO link data rate", dataRate);
    cmd.AddValue("trafficRate", "Offered UDP load of the 0->3 flow", trafficRate);
    cmd.AddValue("packetSize", "UDP payload size [bytes]", packetSize);
    cmd.AddValue("dropWeight", "Reward weight per dropped packet", dropWeight);
    cmd.AddValue("delayWeight", "Reward weight per ms mean delay", delayWeight);
    cmd.AddValue("flapPenalty", "Reward penalty for switching routes", flapPenalty);
    cmd.AddValue("energyWeight", "Reward weight per packet-hop sent", energyWeight);
    cmd.AddValue("simSeed", "Run number for the RNG", simSeed);
    cmd.Parse(argc, argv);

    // Must exist before any Python interaction
    Ptr<OpenGymInterface> openGymInterface = OpenGymInterface::Get();

    SeedManager::SetSeed(1);
    SeedManager::SetRun(simSeed);

    NodeContainer nodes;
    nodes.Create(5);

    auto positions = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < 5; i++)
    {
        double angle = 2.0 * M_PI * i / 5.0;
        positions->Add(Vector(meshRadius * std::cos(angle), meshRadius * std::sin(angle), 0.0));
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
    fso.SetLinkAttribute("TxPowerDbm", DoubleValue(txPowerDbm));
    fso.SetLinkAttribute("NoiseDbm", DoubleValue(noiseDbm));
    fso.SetLinkAttribute("UpdateInterval", TimeValue(MilliSeconds(updateIntervalMs)));
    fso.SetLinkAttribute("PacketSize", UintegerValue(packetSize));

    // Shadow loss model for deterministic observation terms (margin, SI)
    auto shadowLoss = CreateObject<GammaGammaFsoLossModel>();
    shadowLoss->SetAttribute("C2n", DoubleValue(c2n));
    shadowLoss->SetAttribute("Wavelength", DoubleValue(wavelength));
    shadowLoss->SetAttribute("ExtinctionCoefficient", DoubleValue(extinction));

    // Ring plus two cross links
    const std::pair<uint32_t, uint32_t> linkPairs[] =
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 2}, {1, 3}};

    std::vector<FsoLinkRecord> links;
    links.reserve(std::size(linkPairs));
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

    // Candidate routes for the 0->3 flow, as node-id paths
    std::vector<std::vector<uint32_t>> routes = {{0, 2, 3}, {0, 1, 3}, {0, 4, 3}, {0, 1, 2, 3}};

    // Flow destination: node 3's address on the 2-3 link (weak end-system
    // model accepts it on any interface)
    Ipv4Address dstAddr = links[2].addrB;
    const uint16_t port = 9000;
    const double episodeEnd = stepTime * episodeSteps;

    OnOffHelper onOff("ns3::UdpSocketFactory", InetSocketAddress(dstAddr, port));
    onOff.SetConstantRate(DataRate(trafficRate), packetSize);
    ApplicationContainer apps = onOff.Install(nodes.Get(0));

    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    apps.Add(sink.Install(nodes.Get(3)));
    apps.Start(Seconds(0.0));
    apps.Stop(Seconds(episodeEnd));

    FlowMonitorHelper flowMonitorHelper;
    Ptr<FlowMonitor> flowMonitor = flowMonitorHelper.InstallAll();

    FsoRlEnvConfig config;
    config.stepTime = Seconds(stepTime);
    config.episodeSteps = episodeSteps;
    config.dropWeight = dropWeight;
    config.delayWeight = delayWeight;
    config.flapPenalty = flapPenalty;
    config.energyWeight = energyWeight;

    Ptr<FsoRlEnv> env = CreateObject<FsoRlEnv>();
    env->Setup(nodes, &links, routes, dstAddr, flowMonitor, config);
    env->SetOpenGymInterface(openGymInterface);
    env->Start();

    Simulator::Stop(Seconds(episodeEnd) + MilliSeconds(1));
    Simulator::Run();

    openGymInterface->NotifySimulationEnd();
    Simulator::Destroy();
    return 0;
}
