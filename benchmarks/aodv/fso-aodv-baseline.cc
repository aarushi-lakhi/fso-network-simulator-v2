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

// AODV baseline over the Phase 3 FSO mesh — the classical-routing benchmark
// for Phase 5. Builds the exact topology, link budget, and traffic of
// fso-rl-env.cc (5 nodes on an 800 m pentagon, 7 Gamma-Gamma faded links,
// one 2 Mbps UDP flow 0->3) but routes with ns-3's AODV instead of
// agent-controlled static host routes.
//
// Fading realisations match fso-rl-env for the same simSeed because the
// helper's random streams are pinned with the same AssignStreams offset.
//
// Every stepTime the program prints one machine-readable line:
//
//   FSO-BENCH step=<n> drops=<d> txPkts=<t> rxPkts=<r> meanDelayMs=<ms>
//             txTotal=<l> reward=<w>
//
// where drops counts PhyRxDrop on all links (fading losses, control packets
// included), txPkts/rxPkts/meanDelayMs are FlowMonitor deltas of the UDP
// flow only (port 9000), and txTotal counts every link transmission
// (PhyTxEnd on all devices — data forwarding plus AODV control overhead).
// The reward mirrors fso-rl-env's shaping so episode totals are comparable:
//
//   r = - dropWeight * drops - delayWeight * meanDelayMs
//       - energyWeight * txTotal
//
// with two honest differences: no flap penalty (AODV has no agent-visible
// route switch), and the energy proxy charges *actual* laser transmissions
// (fso-rl-env charges hops * txPackets, which for its loss-free static
// forwarding is the same quantity; here it also bills AODV's RREQ/RREP/
// RERR/hello traffic).

#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/fso-topology-helper.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-net-device.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FsoAodvBaseline");

/**
 * \brief Cumulative PHY counters updated by trace sinks.
 */
struct PhyCounters
{
    uint64_t rxDrop{0};  //!< PhyRxDrop over all devices
    uint64_t txTotal{0}; //!< PhyTxEnd over all devices
};

/**
 * \brief Trace sink counting fading-corrupted receptions.
 * \param counters the shared counters
 * \param packet the dropped packet (unused)
 */
static void
PhyRxDropTrace(PhyCounters* counters, Ptr<const Packet> packet)
{
    counters->rxDrop++;
}

/**
 * \brief Trace sink counting completed link transmissions.
 * \param counters the shared counters
 * \param packet the transmitted packet (unused)
 */
static void
PhyTxEndTrace(PhyCounters* counters, Ptr<const Packet> packet)
{
    counters->txTotal++;
}

/**
 * \brief Periodic probe printing per-step metrics in fso-rl-env's format.
 */
class StepProbe
{
  public:
    /**
     * \brief Construct the probe.
     *
     * \param counters shared PHY counters
     * \param monitor flow monitor covering the mesh
     * \param classifier flow classifier used to isolate the UDP flow
     * \param flowPort destination port identifying the measured flow
     * \param stepTime probe period
     * \param episodeSteps number of steps to run
     * \param dropWeight reward weight per dropped packet
     * \param delayWeight reward weight per ms mean delay
     * \param energyWeight reward weight per link transmission
     */
    StepProbe(PhyCounters* counters,
              Ptr<FlowMonitor> monitor,
              Ptr<Ipv4FlowClassifier> classifier,
              uint16_t flowPort,
              Time stepTime,
              uint32_t episodeSteps,
              double dropWeight,
              double delayWeight,
              double energyWeight)
        : m_counters(counters),
          m_monitor(monitor),
          m_classifier(classifier),
          m_flowPort(flowPort),
          m_stepTime(stepTime),
          m_episodeSteps(episodeSteps),
          m_dropWeight(dropWeight),
          m_delayWeight(delayWeight),
          m_energyWeight(energyWeight)
    {
    }

    /**
     * \brief Schedule the first step.
     */
    void Start()
    {
        Simulator::Schedule(m_stepTime, &StepProbe::Step, this);
    }

  private:
    /**
     * \brief Snapshot counters, print the step line, reschedule.
     */
    void Step()
    {
        m_stepCount++;

        uint64_t dropDelta = m_counters->rxDrop - m_prevRxDrop;
        uint64_t txTotalDelta = m_counters->txTotal - m_prevTxTotal;
        m_prevRxDrop = m_counters->rxDrop;
        m_prevTxTotal = m_counters->txTotal;

        uint64_t flowTx = 0;
        uint64_t flowRx = 0;
        Time delaySum;
        for (const auto& [flowId, stats] : m_monitor->GetFlowStats())
        {
            Ipv4FlowClassifier::FiveTuple tuple = m_classifier->FindFlow(flowId);
            if (tuple.destinationPort != m_flowPort)
            {
                continue; // AODV control traffic
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

        double reward = -m_dropWeight * double(dropDelta) - m_delayWeight * meanDelayMs -
                        m_energyWeight * double(txTotalDelta);

        std::cout << "FSO-BENCH step=" << m_stepCount << " drops=" << dropDelta
                  << " txPkts=" << txDelta << " rxPkts=" << rxDelta
                  << " meanDelayMs=" << meanDelayMs << " txTotal=" << txTotalDelta
                  << " reward=" << reward << std::endl;

        if (m_stepCount < m_episodeSteps)
        {
            Simulator::Schedule(m_stepTime, &StepProbe::Step, this);
        }
    }

    PhyCounters* m_counters;               //!< Shared PHY counters
    Ptr<FlowMonitor> m_monitor;             //!< Flow statistics source
    Ptr<Ipv4FlowClassifier> m_classifier;   //!< Flow tuple lookup
    uint16_t m_flowPort;                    //!< Measured flow's port
    Time m_stepTime;                        //!< Probe period
    uint32_t m_episodeSteps;                //!< Steps per episode
    double m_dropWeight;                    //!< Reward weight per drop
    double m_delayWeight;                   //!< Reward weight per ms delay
    double m_energyWeight;                  //!< Reward weight per transmission

    uint32_t m_stepCount{0};   //!< Steps taken so far
    uint64_t m_prevRxDrop{0};  //!< rxDrop at the previous step
    uint64_t m_prevTxTotal{0}; //!< txTotal at the previous step
    uint64_t m_prevFlowTx{0};  //!< Flow tx packets at the previous step
    uint64_t m_prevFlowRx{0};  //!< Flow rx packets at the previous step
    Time m_prevDelaySum;       //!< Flow delay sum at the previous step
};

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
    std::string dataRate = "100Mbps";
    std::string trafficRate = "2Mbps";
    uint32_t packetSize = 1024;
    double dropWeight = 1.0;
    double delayWeight = 0.1;
    double energyWeight = 0.01;
    uint32_t simSeed = 1;

    CommandLine cmd(__FILE__);
    cmd.AddValue("c2n", "Refractive index structure parameter [m^-2/3]", c2n);
    cmd.AddValue("episodeSteps", "Decision steps per episode", episodeSteps);
    cmd.AddValue("stepTime", "Metrics interval [s]", stepTime);
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
    cmd.AddValue("dataRate", "FSO link data rate", dataRate);
    cmd.AddValue("trafficRate", "Offered UDP load of the 0->3 flow", trafficRate);
    cmd.AddValue("packetSize", "UDP payload size [bytes]", packetSize);
    cmd.AddValue("dropWeight", "Reward weight per dropped packet", dropWeight);
    cmd.AddValue("delayWeight", "Reward weight per ms mean delay", delayWeight);
    cmd.AddValue("energyWeight", "Reward weight per link transmission", energyWeight);
    cmd.AddValue("simSeed", "Run number for the RNG", simSeed);
    cmd.Parse(argc, argv);

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

    AodvHelper aodv;
    InternetStackHelper internet;
    internet.SetRoutingHelper(aodv);
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

    // Same links, install order, and addressing as fso-rl-env.cc
    const std::pair<uint32_t, uint32_t> linkPairs[] =
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 2}, {1, 3}};

    PhyCounters counters;
    Ipv4Address dstAddr;
    Ipv4AddressHelper addresses;
    uint32_t subnet = 1;
    for (auto [i, j] : linkPairs)
    {
        NetDeviceContainer devices = fso.Install(nodes.Get(i), nodes.Get(j));
        std::ostringstream base;
        base << "10.1." << subnet++ << ".0";
        addresses.SetBase(base.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer ifaces = addresses.Assign(devices);
        if (i == 2 && j == 3)
        {
            // Flow destination: node 3's address on the 2-3 link
            dstAddr = ifaces.GetAddress(1);
        }
        for (uint32_t d = 0; d < devices.GetN(); d++)
        {
            devices.Get(d)->TraceConnectWithoutContext(
                "PhyRxDrop", MakeBoundCallback(&PhyRxDropTrace, &counters));
            devices.Get(d)->TraceConnectWithoutContext(
                "PhyTxEnd", MakeBoundCallback(&PhyTxEndTrace, &counters));
        }
    }
    fso.AssignStreams(100);

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
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowMonitorHelper.GetClassifier());

    StepProbe probe(&counters,
                    flowMonitor,
                    classifier,
                    port,
                    Seconds(stepTime),
                    episodeSteps,
                    dropWeight,
                    delayWeight,
                    energyWeight);
    probe.Start();

    Simulator::Stop(Seconds(episodeEnd) + MilliSeconds(1));
    Simulator::Run();
    Simulator::Destroy();

    std::cout << "FSO-BENCH done" << std::endl;
    return 0;
}
