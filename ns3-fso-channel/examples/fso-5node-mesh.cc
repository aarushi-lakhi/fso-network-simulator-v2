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

// Five FSO terminals on a ~800 m radius pentagon, connected by point-to-point
// laser links (ring plus two cross links). UDP flows cross the mesh while the
// Gamma-Gamma fading bridge modulates each link's packet error rate. Per-flow
// packet delivery ratio is reported via FlowMonitor.
//
// Usage:
//   ns3 run 'fso-5node-mesh --regime=weak'      (C2n = 1e-17)
//   ns3 run 'fso-5node-mesh --regime=moderate'  (C2n = 1e-15)
//   ns3 run 'fso-5node-mesh --regime=strong'    (C2n = 1e-13)
//
// Temporally correlated fading (default 0 = i.i.d. block fading):
//   ns3 run 'fso-5node-mesh --coherenceLarge=100ms --coherenceSmall=10ms'

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/fso-topology-helper.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Fso5NodeMesh");

int
main(int argc, char* argv[])
{
    std::string regime = "moderate";
    double duration = 10.0;
    Time coherenceLarge = Seconds(0);
    Time coherenceSmall = Seconds(0);

    CommandLine cmd(__FILE__);
    cmd.AddValue("regime", "Turbulence regime: weak | moderate | strong", regime);
    cmd.AddValue("duration", "Simulation duration [s]", duration);
    cmd.AddValue("coherenceLarge",
                 "Large-scale fading coherence time, e.g. 100ms (0 = i.i.d.)",
                 coherenceLarge);
    cmd.AddValue("coherenceSmall",
                 "Small-scale fading coherence time, e.g. 10ms (0 = i.i.d.)",
                 coherenceSmall);
    cmd.Parse(argc, argv);

    double c2n;
    if (regime == "weak")
    {
        c2n = 1e-17;
    }
    else if (regime == "moderate")
    {
        c2n = 1e-15;
    }
    else if (regime == "strong")
    {
        c2n = 1e-13;
    }
    else
    {
        std::cerr << "Unknown regime '" << regime << "'" << std::endl;
        return 1;
    }

    NodeContainer nodes;
    nodes.Create(5);

    // Pentagon, radius 800 m (side ~940 m, cross links ~1522 m)
    auto positions = CreateObject<ListPositionAllocator>();
    const double radius = 800.0;
    for (uint32_t i = 0; i < 5; i++)
    {
        double angle = 2.0 * M_PI * i / 5.0;
        positions->Add(Vector(radius * std::cos(angle), radius * std::sin(angle), 0.0));
    }
    MobilityHelper mobility;
    mobility.SetPositionAllocator(positions);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    FsoTopologyHelper fso;
    fso.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    fso.SetChannelAttribute("Delay", StringValue("5us"));
    fso.SetLossModelAttribute("C2n", DoubleValue(c2n));
    fso.SetLossModelAttribute("Wavelength", DoubleValue(1550e-9));
    fso.SetLossModelAttribute("ExtinctionCoefficient", DoubleValue(1e-5));
    fso.SetLossModelAttribute("CoherenceTimeLargeScale", TimeValue(coherenceLarge));
    fso.SetLossModelAttribute("CoherenceTimeSmallScale", TimeValue(coherenceSmall));
    fso.SetLinkAttribute("TxPowerDbm", DoubleValue(10.0));
    fso.SetLinkAttribute("NoiseDbm", DoubleValue(-8.0));
    fso.SetLinkAttribute("UpdateInterval", TimeValue(MilliSeconds(1)));
    fso.SetLinkAttribute("PacketSize", UintegerValue(1024));

    // Ring plus two cross links
    const std::pair<uint32_t, uint32_t> linkPairs[] =
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 2}, {1, 3}};

    InternetStackHelper internet;
    internet.Install(nodes);

    Ipv4AddressHelper addresses;
    uint32_t subnet = 1;
    for (auto [i, j] : linkPairs)
    {
        NetDeviceContainer devices = fso.Install(nodes.Get(i), nodes.Get(j));
        std::ostringstream base;
        base << "10.1." << subnet++ << ".0";
        addresses.SetBase(base.str().c_str(), "255.255.255.0");
        addresses.Assign(devices);
    }
    fso.AssignStreams(100);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Three UDP flows crossing the mesh
    const std::pair<uint32_t, uint32_t> flowPairs[] = {{0, 3}, {1, 4}, {2, 0}};
    const uint16_t port = 9000;
    const uint32_t packetSize = 1024;

    ApplicationContainer apps;
    for (auto [src, dst] : flowPairs)
    {
        Ptr<Ipv4> ipv4 = nodes.Get(dst)->GetObject<Ipv4>();
        Ipv4Address dstAddress = ipv4->GetAddress(1, 0).GetLocal();

        OnOffHelper onOff("ns3::UdpSocketFactory", InetSocketAddress(dstAddress, port));
        onOff.SetConstantRate(DataRate("2Mbps"), packetSize);
        apps.Add(onOff.Install(nodes.Get(src)));

        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        apps.Add(sink.Install(nodes.Get(dst)));
    }
    apps.Start(Seconds(1.0));
    apps.Stop(Seconds(1.0 + duration));

    FlowMonitorHelper flowMonitorHelper;
    Ptr<FlowMonitor> flowMonitor = flowMonitorHelper.InstallAll();

    Simulator::Stop(Seconds(2.0 + duration));
    Simulator::Run();

    auto classifier = DynamicCast<Ipv4FlowClassifier>(flowMonitorHelper.GetClassifier());
    std::cout << "FSO 5-node mesh, regime=" << regime << " (C2n=" << c2n << " m^-2/3), "
              << duration << " s of traffic, coherence large/small = "
              << coherenceLarge.As(Time::MS) << "/" << coherenceSmall.As(Time::MS) << std::endl;
    std::cout << std::left << std::setw(24) << "flow" << std::setw(10) << "txPkts"
              << std::setw(10) << "rxPkts" << "PDR" << std::endl;

    double totalTx = 0;
    double totalRx = 0;
    for (const auto& [flowId, stats] : flowMonitor->GetFlowStats())
    {
        Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow(flowId);
        std::ostringstream label;
        label << tuple.sourceAddress << " -> " << tuple.destinationAddress;
        double pdr = stats.txPackets > 0 ? double(stats.rxPackets) / stats.txPackets : 0.0;
        totalTx += stats.txPackets;
        totalRx += stats.rxPackets;
        std::cout << std::left << std::setw(24) << label.str() << std::setw(10)
                  << stats.txPackets << std::setw(10) << stats.rxPackets << std::fixed
                  << std::setprecision(4) << pdr << std::endl;
    }
    std::cout << "overall PDR: " << std::fixed << std::setprecision(4)
              << (totalTx > 0 ? totalRx / totalTx : 0.0) << std::endl;

    Simulator::Destroy();
    return 0;
}
