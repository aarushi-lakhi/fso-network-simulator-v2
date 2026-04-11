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

#ifndef FSO_TOPOLOGY_HELPER_H
#define FSO_TOPOLOGY_HELPER_H

#include "ns3/error-model.h"
#include "ns3/event-id.h"
#include "ns3/gamma-gamma-fso-loss-model.h"
#include "ns3/net-device-container.h"
#include "ns3/nstime.h"
#include "ns3/object-factory.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/ptr.h"

#include <string>
#include <vector>

namespace ns3
{

class MobilityModel;
class Node;

/**
 * \ingroup fso-channel
 *
 * \brief Per-link bridge from Gamma-Gamma fading to a p2p packet error rate.
 *
 * Architecture note: the FSO topology uses PointToPointChannel (an FSO link
 * is a laser beam, not broadcast RF), but PointToPointChannel does not
 * consume a PropagationLossModel. This object bridges the two abstractions:
 * every UpdateInterval it draws a fresh irradiance sample I from the link's
 * GammaGammaFsoLossModel (one independent sample per direction), computes the
 * instantaneous electrical SNR from a simple link budget,
 *
 *   snr = 10^((TxPowerDbm - extinctionLossDb(d) - NoiseDbm) / 10)
 *
 * maps it to an OOK-IM/DD bit error rate
 *
 *   BER = 0.5 erfc(sqrt(snr * I / 2))
 *
 * converts BER to a packet error rate for the nominal PacketSize,
 *
 *   PER = 1 - (1 - BER)^(8 * PacketSize)
 *
 * and installs the PER on the receiving device's RateErrorModel
 * (ERROR_UNIT_PACKET). The result is block fading: the channel state is held
 * for UpdateInterval (~turbulence coherence time), then redrawn.
 *
 * Instances are created by FsoTopologyHelper, one per FSO link.
 */
class FsoLinkFadingModel : public Object
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    FsoLinkFadingModel();
    ~FsoLinkFadingModel() override;

    /**
     * \brief Attach the link endpoints and the models to drive.
     *
     * \param lossModel the FSO loss model providing fading samples
     * \param mobilityA mobility model of the first endpoint
     * \param mobilityB mobility model of the second endpoint
     * \param errorModelAtB error model on B's device (corrupts A->B traffic)
     * \param errorModelAtA error model on A's device (corrupts B->A traffic)
     */
    void Setup(Ptr<GammaGammaFsoLossModel> lossModel,
               Ptr<MobilityModel> mobilityA,
               Ptr<MobilityModel> mobilityB,
               Ptr<RateErrorModel> errorModelAtB,
               Ptr<RateErrorModel> errorModelAtA);

    /**
     * \brief Schedule the first channel-state update at time 0.
     */
    void Start();

    /**
     * \brief Assign fixed stream numbers to the underlying random variables.
     *
     * Covers the loss model's fading RNG and both error model variates.
     *
     * \param stream first stream index to use
     * \return the number of stream indices assigned
     */
    int64_t AssignStreams(int64_t stream);

  protected:
    void DoDispose() override;

  private:
    /**
     * \brief Redraw the fading state and update both directions' PER.
     */
    void Update();

    /**
     * \brief Compute the packet error rate for one fading realisation.
     * \param snr mean electrical SNR (linear)
     * \param irradiance normalised irradiance sample I
     * \return packet error rate in [0, 1]
     */
    double CalcPer(double snr, double irradiance) const;

    double m_txPowerDbm;     //!< Transmit optical power [dBm]
    double m_noiseDbm;       //!< Receiver noise-equivalent power [dBm]
    Time m_updateInterval;   //!< Channel coherence / update interval
    uint32_t m_packetSize;   //!< Nominal packet size for BER->PER [bytes]

    Ptr<GammaGammaFsoLossModel> m_lossModel; //!< Fading source
    Ptr<MobilityModel> m_mobilityA;          //!< Endpoint A position
    Ptr<MobilityModel> m_mobilityB;          //!< Endpoint B position
    Ptr<RateErrorModel> m_errorModelAtB;     //!< PER sink for A->B
    Ptr<RateErrorModel> m_errorModelAtA;     //!< PER sink for B->A
    EventId m_updateEvent;                   //!< Pending update event
};

/**
 * \ingroup fso-channel
 *
 * \brief Builds point-to-point FSO links with turbulence-driven error rates.
 *
 * For each Install(a, b) call the helper creates a PointToPointChannel and
 * devices, a GammaGammaFsoLossModel parameterised from SetLossModelAttribute,
 * a packet-mode RateErrorModel on each device's receive path, and an
 * FsoLinkFadingModel (see its documentation for the bridging design) that
 * periodically maps the current fading state onto the error models.
 *
 * Both nodes must have a MobilityModel installed before Install() so the
 * link distance can be derived.
 */
class FsoTopologyHelper
{
  public:
    FsoTopologyHelper();

    /**
     * \brief Set an attribute on the created PointToPointNetDevices.
     * \param name the attribute name (e.g. "DataRate")
     * \param value the attribute value
     */
    void SetDeviceAttribute(std::string name, const AttributeValue& value);

    /**
     * \brief Set an attribute on the created PointToPointChannels.
     * \param name the attribute name (e.g. "Delay")
     * \param value the attribute value
     */
    void SetChannelAttribute(std::string name, const AttributeValue& value);

    /**
     * \brief Set an attribute on the created GammaGammaFsoLossModels.
     * \param name the attribute name (e.g. "C2n")
     * \param value the attribute value
     */
    void SetLossModelAttribute(std::string name, const AttributeValue& value);

    /**
     * \brief Set an attribute on the created FsoLinkFadingModels.
     * \param name the attribute name (e.g. "TxPowerDbm", "UpdateInterval")
     * \param value the attribute value
     */
    void SetLinkAttribute(std::string name, const AttributeValue& value);

    /**
     * \brief Create an FSO link between two nodes.
     *
     * \param a first node
     * \param b second node
     * \return container with the two created devices (a's first)
     */
    NetDeviceContainer Install(Ptr<Node> a, Ptr<Node> b);

    /**
     * \brief Get the fading model of the i-th installed link.
     * \param i link index, in order of Install() calls
     * \return the link's FsoLinkFadingModel
     */
    Ptr<FsoLinkFadingModel> GetLink(std::size_t i) const;

    /**
     * \brief Assign fixed stream numbers to all links' random variables.
     * \param stream first stream index to use
     * \return the number of stream indices assigned
     */
    int64_t AssignStreams(int64_t stream);

  private:
    PointToPointHelper m_p2p;                      //!< Underlying p2p helper
    ObjectFactory m_lossModelFactory;              //!< Per-link loss models
    ObjectFactory m_linkFactory;                   //!< Per-link fading bridges
    std::vector<Ptr<FsoLinkFadingModel>> m_links;  //!< Installed links
};

} // namespace ns3

#endif /* FSO_TOPOLOGY_HELPER_H */
