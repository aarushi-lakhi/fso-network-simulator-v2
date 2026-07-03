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

#include "fso-topology-helper.h"

#include "ns3/abort.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/node.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"

#include <cmath>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("FsoTopologyHelper");

NS_OBJECT_ENSURE_REGISTERED(FsoLinkFadingModel);

TypeId
FsoLinkFadingModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::FsoLinkFadingModel")
            .SetParent<Object>()
            .SetGroupName("FsoChannel")
            .AddConstructor<FsoLinkFadingModel>()
            .AddAttribute("TxPowerDbm",
                          "Transmit optical power [dBm].",
                          DoubleValue(10.0),
                          MakeDoubleAccessor(&FsoLinkFadingModel::m_txPowerDbm),
                          MakeDoubleChecker<double>())
            .AddAttribute("NoiseDbm",
                          "Receiver noise-equivalent power [dBm]; the mean electrical "
                          "SNR is TxPowerDbm - extinction loss - NoiseDbm.",
                          DoubleValue(-8.0),
                          MakeDoubleAccessor(&FsoLinkFadingModel::m_noiseDbm),
                          MakeDoubleChecker<double>())
            .AddAttribute("UpdateInterval",
                          "Fading state (block fading) refresh period, on the order "
                          "of the turbulence coherence time.",
                          TimeValue(MilliSeconds(1)),
                          MakeTimeAccessor(&FsoLinkFadingModel::m_updateInterval),
                          MakeTimeChecker())
            .AddAttribute("PacketSize",
                          "Nominal packet size used for the BER to PER conversion [bytes].",
                          UintegerValue(1024),
                          MakeUintegerAccessor(&FsoLinkFadingModel::m_packetSize),
                          MakeUintegerChecker<uint32_t>(1));
    return tid;
}

FsoLinkFadingModel::FsoLinkFadingModel()
{
    NS_LOG_FUNCTION(this);
}

FsoLinkFadingModel::~FsoLinkFadingModel()
{
    NS_LOG_FUNCTION(this);
}

void
FsoLinkFadingModel::Setup(Ptr<GammaGammaFsoLossModel> lossModel,
                          Ptr<MobilityModel> mobilityA,
                          Ptr<MobilityModel> mobilityB,
                          Ptr<RateErrorModel> errorModelAtB,
                          Ptr<RateErrorModel> errorModelAtA)
{
    NS_LOG_FUNCTION(this);
    m_lossModel = lossModel;
    m_mobilityA = mobilityA;
    m_mobilityB = mobilityB;
    m_errorModelAtB = errorModelAtB;
    m_errorModelAtA = errorModelAtA;
}

void
FsoLinkFadingModel::Start()
{
    NS_ABORT_MSG_IF(!m_lossModel, "Setup() must be called before Start()");
    m_updateEvent = Simulator::ScheduleNow(&FsoLinkFadingModel::Update, this);
}

int64_t
FsoLinkFadingModel::AssignStreams(int64_t stream)
{
    int64_t currentStream = stream;
    currentStream += m_lossModel->AssignStreams(currentStream);
    currentStream += m_errorModelAtB->AssignStreams(currentStream);
    currentStream += m_errorModelAtA->AssignStreams(currentStream);
    return currentStream - stream;
}

void
FsoLinkFadingModel::DoDispose()
{
    m_updateEvent.Cancel();
    m_lossModel = nullptr;
    m_mobilityA = nullptr;
    m_mobilityB = nullptr;
    m_errorModelAtB = nullptr;
    m_errorModelAtA = nullptr;
    Object::DoDispose();
}

double
FsoLinkFadingModel::CalcPer(double snr, double irradiance) const
{
    double ber = 0.5 * std::erfc(std::sqrt(snr * irradiance / 2.0));
    double per = 1.0 - std::pow(1.0 - ber, 8.0 * m_packetSize);
    return per;
}

void
FsoLinkFadingModel::Update()
{
    double distance = m_mobilityA->GetDistanceFrom(m_mobilityB);
    double meanRxDbm = m_txPowerDbm - m_lossModel->GetExtinctionLossDb(distance);
    double snr = std::pow(10.0, (meanRxDbm - m_noiseDbm) / 10.0);

    // Independent fading realisation per direction
    double perAtoB = CalcPer(snr, m_lossModel->GetFadingSample(distance));
    double perBtoA = CalcPer(snr, m_lossModel->GetFadingSample(distance));
    m_errorModelAtB->SetRate(perAtoB);
    m_errorModelAtA->SetRate(perBtoA);

    NS_LOG_DEBUG("d=" << distance << " m, PER A->B=" << perAtoB << ", PER B->A=" << perBtoA);

    m_updateEvent = Simulator::Schedule(m_updateInterval, &FsoLinkFadingModel::Update, this);
}

FsoTopologyHelper::FsoTopologyHelper()
{
    m_lossModelFactory.SetTypeId("ns3::GammaGammaFsoLossModel");
    m_linkFactory.SetTypeId("ns3::FsoLinkFadingModel");
}

void
FsoTopologyHelper::SetDeviceAttribute(std::string name, const AttributeValue& value)
{
    m_p2p.SetDeviceAttribute(name, value);
}

void
FsoTopologyHelper::SetChannelAttribute(std::string name, const AttributeValue& value)
{
    m_p2p.SetChannelAttribute(name, value);
}

void
FsoTopologyHelper::SetLossModelAttribute(std::string name, const AttributeValue& value)
{
    m_lossModelFactory.Set(name, value);
}

void
FsoTopologyHelper::SetLinkAttribute(std::string name, const AttributeValue& value)
{
    m_linkFactory.Set(name, value);
}

NetDeviceContainer
FsoTopologyHelper::Install(Ptr<Node> a, Ptr<Node> b)
{
    Ptr<MobilityModel> mobilityA = a->GetObject<MobilityModel>();
    Ptr<MobilityModel> mobilityB = b->GetObject<MobilityModel>();
    NS_ABORT_MSG_IF(!mobilityA || !mobilityB,
                    "FsoTopologyHelper::Install: both nodes need a MobilityModel");

    NetDeviceContainer devices = m_p2p.Install(a, b);

    auto errorModelAtA = CreateObject<RateErrorModel>();
    auto errorModelAtB = CreateObject<RateErrorModel>();
    for (auto errorModel : {errorModelAtA, errorModelAtB})
    {
        errorModel->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
        errorModel->SetRate(0.0);
    }
    devices.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(errorModelAtA));
    devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(errorModelAtB));

    auto lossModel = m_lossModelFactory.Create<GammaGammaFsoLossModel>();
    auto link = m_linkFactory.Create<FsoLinkFadingModel>();
    link->Setup(lossModel, mobilityA, mobilityB, errorModelAtB, errorModelAtA);
    link->Start();
    m_links.push_back(link);

    return devices;
}

Ptr<FsoLinkFadingModel>
FsoTopologyHelper::GetLink(std::size_t i) const
{
    NS_ABORT_MSG_IF(i >= m_links.size(), "link index out of range");
    return m_links[i];
}

int64_t
FsoTopologyHelper::AssignStreams(int64_t stream)
{
    int64_t currentStream = stream;
    for (auto& link : m_links)
    {
        currentStream += link->AssignStreams(currentStream);
    }
    return currentStream - stream;
}

} // namespace ns3
