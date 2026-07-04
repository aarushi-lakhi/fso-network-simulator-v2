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

#include "correlated-gamma-gamma-fading.h"

#include "ns3/abort.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("CorrelatedGammaGammaFading");

NS_OBJECT_ENSURE_REGISTERED(CorrelatedGammaGammaFading);

TypeId
CorrelatedGammaGammaFading::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::CorrelatedGammaGammaFading")
            .SetParent<Object>()
            .SetGroupName("FsoChannel")
            .AddConstructor<CorrelatedGammaGammaFading>()
            .AddAttribute("CoherenceTimeLargeScale",
                          "Coherence time tau of the large-scale (alpha) Gamma component; "
                          "the latent AR(1) coefficient is exp(-dt/tau). "
                          "Zero means i.i.d. draws.",
                          TimeValue(Seconds(0)),
                          MakeTimeAccessor(&CorrelatedGammaGammaFading::m_tauLargeScale),
                          MakeTimeChecker())
            .AddAttribute("CoherenceTimeSmallScale",
                          "Coherence time tau of the small-scale (beta) Gamma component; "
                          "the latent AR(1) coefficient is exp(-dt/tau). "
                          "Zero means i.i.d. draws.",
                          TimeValue(Seconds(0)),
                          MakeTimeAccessor(&CorrelatedGammaGammaFading::m_tauSmallScale),
                          MakeTimeChecker());
    return tid;
}

CorrelatedGammaGammaFading::CorrelatedGammaGammaFading()
{
    NS_LOG_FUNCTION(this);
    m_rng = CreateObject<NormalRandomVariable>();
}

CorrelatedGammaGammaFading::~CorrelatedGammaGammaFading()
{
    NS_LOG_FUNCTION(this);
}

double
CorrelatedGammaGammaFading::SampleComponent(ComponentState& state, double shape, Time tau)
{
    NS_ABORT_MSG_IF(shape <= 0.0, "Gamma shape parameter must be positive");

    double epsilon = m_rng->GetValue();
    double rho = 0.0;
    if (state.initialized && tau.IsStrictlyPositive())
    {
        Time now = Simulator::Now();
        double dt = (now - state.lastSampleTime).GetSeconds();
        rho = std::exp(-dt / tau.GetSeconds());
    }
    state.g = rho * state.g + std::sqrt(1.0 - rho * rho) * epsilon;
    state.initialized = true;
    state.lastSampleTime = Simulator::Now();

    // Probability integral transform: u = Phi(g), then the Gamma quantile.
    // gamma_p_inv gives the unit-scale quantile; divide by the shape for
    // scale 1/shape so the component has unit mean. Clamp u away from the
    // endpoints where the quantile is 0 or infinite.
    double u = 0.5 * std::erfc(-state.g * M_SQRT1_2);
    u = std::clamp(u, 1e-16, 1.0 - 1e-16);
    return boost::math::gamma_p_inv(shape, u) / shape;
}

double
CorrelatedGammaGammaFading::GetSample(double alpha, double beta)
{
    double largeScale = SampleComponent(m_largeScale, alpha, m_tauLargeScale);
    double smallScale = SampleComponent(m_smallScale, beta, m_tauSmallScale);
    NS_LOG_DEBUG("t=" << Simulator::Now().GetSeconds() << " s, X=" << largeScale
                      << ", Y=" << smallScale);
    return largeScale * smallScale;
}

int64_t
CorrelatedGammaGammaFading::AssignStreams(int64_t stream)
{
    m_rng->SetStream(stream);
    return 1;
}

} // namespace ns3
