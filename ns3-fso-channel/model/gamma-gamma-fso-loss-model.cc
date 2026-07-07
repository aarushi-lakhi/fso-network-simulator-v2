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

#include "ns3/abort.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"

#include <cmath>

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
                          MakeBooleanChecker())
            .AddAttribute("CoherenceTimeLargeScale",
                          "Coherence time of the large-scale (alpha) fading component. "
                          "Zero (default) keeps the historical i.i.d. per-call draws.",
                          TimeValue(Seconds(0)),
                          MakeTimeAccessor(&GammaGammaFsoLossModel::SetCoherenceTimeLargeScale,
                                           &GammaGammaFsoLossModel::GetCoherenceTimeLargeScale),
                          MakeTimeChecker())
            .AddAttribute("CoherenceTimeSmallScale",
                          "Coherence time of the small-scale (beta) fading component. "
                          "Zero (default) keeps the historical i.i.d. per-call draws.",
                          TimeValue(Seconds(0)),
                          MakeTimeAccessor(&GammaGammaFsoLossModel::SetCoherenceTimeSmallScale,
                                           &GammaGammaFsoLossModel::GetCoherenceTimeSmallScale),
                          MakeTimeChecker());
    return tid;
}

GammaGammaFsoLossModel::GammaGammaFsoLossModel()
    : PropagationLossModel()
{
    NS_LOG_FUNCTION(this);
    m_rng = CreateObject<GammaRandomVariable>();
    m_correlatedFading = CreateObject<CorrelatedGammaGammaFading>();
}

GammaGammaFsoLossModel::~GammaGammaFsoLossModel()
{
    NS_LOG_FUNCTION(this);
}

void
GammaGammaFsoLossModel::SetCoherenceTimeLargeScale(Time tau)
{
    m_tauLargeScale = tau;
    m_correlatedFading->SetAttribute("CoherenceTimeLargeScale", TimeValue(tau));
}

Time
GammaGammaFsoLossModel::GetCoherenceTimeLargeScale() const
{
    return m_tauLargeScale;
}

void
GammaGammaFsoLossModel::SetCoherenceTimeSmallScale(Time tau)
{
    m_tauSmallScale = tau;
    m_correlatedFading->SetAttribute("CoherenceTimeSmallScale", TimeValue(tau));
}

Time
GammaGammaFsoLossModel::GetCoherenceTimeSmallScale() const
{
    return m_tauSmallScale;
}

bool
GammaGammaFsoLossModel::IsTurbulenceEnabled() const
{
    return m_turbulence;
}

double
GammaGammaFsoLossModel::GetRytovVariance(double distance) const
{
    NS_ABORT_MSG_IF(distance <= 0.0, "distance must be positive");
    double k = 2.0 * M_PI / m_wavelength;
    return 1.23 * m_c2n * std::pow(k, 7.0 / 6.0) * std::pow(distance, 11.0 / 6.0);
}

std::pair<double, double>
GammaGammaFsoLossModel::GetAlphaBeta(double distance) const
{
    double sigma2R = GetRytovVariance(distance);
    double sigmaR = std::sqrt(sigma2R);

    double expArgAlpha = 0.49 * sigma2R / std::pow(1.0 + 1.11 * std::pow(sigmaR, 12.0 / 5.0), 7.0 / 6.0);
    double expArgBeta = 0.51 * sigma2R / std::pow(1.0 + 0.69 * std::pow(sigmaR, 12.0 / 5.0), 5.0 / 6.0);

    // expm1 keeps alpha/beta finite and accurate as sigma2R -> 0
    double alpha = 1.0 / std::expm1(expArgAlpha);
    double beta = 1.0 / std::expm1(expArgBeta);

    return {alpha, beta};
}

double
GammaGammaFsoLossModel::GetFadingSample(double distance)
{
    if (!m_turbulence)
    {
        return 1.0;
    }
    auto [alpha, beta] = GetAlphaBeta(distance);
    if (m_tauLargeScale.IsStrictlyPositive() || m_tauSmallScale.IsStrictlyPositive())
    {
        return m_correlatedFading->GetSample(alpha, beta);
    }
    // Historical i.i.d. path, kept bit-identical for zero coherence times
    double largeScale = m_rng->GetValue(alpha, 1.0 / alpha);
    double smallScale = m_rng->GetValue(beta, 1.0 / beta);
    return largeScale * smallScale;
}

double
GammaGammaFsoLossModel::GetExtinctionLossDb(double distance) const
{
    // -10 log10(exp(-sigma_ext * d)) = 10 sigma_ext d / ln(10)
    return 10.0 * m_extinction * distance / std::log(10.0);
}

double
GammaGammaFsoLossModel::DoCalcRxPower(double txPowerDbm,
                                      Ptr<MobilityModel> a,
                                      Ptr<MobilityModel> b) const
{
    double distance = a->GetDistanceFrom(b);
    if (distance <= 0.0)
    {
        return txPowerDbm;
    }

    double rxPowerDbm = txPowerDbm - GetExtinctionLossDb(distance);
    // GetFadingSample() only mutates the RNG stream; propagation models are
    // conceptually const per ns-3 convention (cf. RandomPropagationLossModel)
    double irradiance = const_cast<GammaGammaFsoLossModel*>(this)->GetFadingSample(distance);
    rxPowerDbm += 10.0 * std::log10(irradiance);

    NS_LOG_DEBUG("d=" << distance << " m, I=" << irradiance << ", rx=" << rxPowerDbm << " dBm");
    return rxPowerDbm;
}

int64_t
GammaGammaFsoLossModel::DoAssignStreams(int64_t stream)
{
    m_rng->SetStream(stream);
    return 1 + m_correlatedFading->AssignStreams(stream + 1);
}

} // namespace ns3
