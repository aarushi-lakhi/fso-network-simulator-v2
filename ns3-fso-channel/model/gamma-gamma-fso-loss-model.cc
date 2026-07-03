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

#include "gamma-gamma-fso-loss-model.h"

#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("GammaGammaFsoLossModel");

NS_OBJECT_ENSURE_REGISTERED(GammaGammaFsoLossModel);

TypeId
GammaGammaFsoLossModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::GammaGammaFsoLossModel")
            .SetParent<PropagationLossModel>()
            .SetGroupName("FsoChannel")
            .AddConstructor<GammaGammaFsoLossModel>()
            .AddAttribute("C2n",
                          "Refractive index structure parameter C_n^2 [m^(-2/3)]. "
                          "Typical: 1e-17 (weak), 1e-15 (moderate), 1e-13 (strong).",
                          DoubleValue(1e-15),
                          MakeDoubleAccessor(&GammaGammaFsoLossModel::m_c2n),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("Wavelength",
                          "Optical carrier wavelength [m].",
                          DoubleValue(1550e-9),
                          MakeDoubleAccessor(&GammaGammaFsoLossModel::m_wavelength),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("ExtinctionCoefficient",
                          "Beer-Lambert atmospheric extinction coefficient sigma_ext [1/m]. "
                          "Typical: 1e-5 (clear air), 1e-4 (haze/fog).",
                          DoubleValue(1e-5),
                          MakeDoubleAccessor(&GammaGammaFsoLossModel::m_extinction),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("TurbulenceEnabled",
                          "Whether the random Gamma-Gamma fading term is applied.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&GammaGammaFsoLossModel::m_turbulence),
                          MakeBooleanChecker());
    return tid;
}

GammaGammaFsoLossModel::GammaGammaFsoLossModel()
    : PropagationLossModel()
{
    NS_LOG_FUNCTION(this);
    m_rng = CreateObject<GammaRandomVariable>();
}

GammaGammaFsoLossModel::~GammaGammaFsoLossModel()
{
    NS_LOG_FUNCTION(this);
}

double
GammaGammaFsoLossModel::GetRytovVariance(double distance) const
{
    NS_ABORT_MSG_IF(distance <= 0.0, "distance must be positive");
    return 0.0;
}

std::pair<double, double>
GammaGammaFsoLossModel::GetAlphaBeta(double distance) const
{
    NS_ABORT_MSG_IF(distance <= 0.0, "distance must be positive");
    return {1.0, 1.0};
}

double
GammaGammaFsoLossModel::GetFadingSample(double distance)
{
    NS_ABORT_MSG_IF(distance <= 0.0, "distance must be positive");
    return 1.0;
}

double
GammaGammaFsoLossModel::GetExtinctionLossDb(double distance) const
{
    return 0.0;
}

double
GammaGammaFsoLossModel::DoCalcRxPower(double txPowerDbm,
                                      Ptr<MobilityModel> a,
                                      Ptr<MobilityModel> b) const
{
    return txPowerDbm;
}

int64_t
GammaGammaFsoLossModel::DoAssignStreams(int64_t stream)
{
    m_rng->SetStream(stream);
    return 1;
}

} // namespace ns3
