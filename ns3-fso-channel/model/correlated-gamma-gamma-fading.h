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

#ifndef CORRELATED_GAMMA_GAMMA_FADING_H
#define CORRELATED_GAMMA_GAMMA_FADING_H

#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/random-variable-stream.h"

namespace ns3
{

/**
 * \ingroup fso-channel
 *
 * \brief Temporally correlated Gamma-Gamma fading process for one FSO path.
 *
 * Generates a stationary irradiance process \f$ I(t) = X(t) Y(t) \f$ whose
 * marginal distribution is exactly the unit-mean Gamma-Gamma law
 * \f$ X \sim \Gamma(\alpha, 1/\alpha),\, Y \sim \Gamma(\beta, 1/\beta) \f$
 * while temporal coherence is tunable per component. Each Gamma component
 * with shape \f$ k \f$ and coherence time \f$ \tau \f$ evolves through a
 * latent standard-normal AR(1) process (Gaussian copula):
 *
 *   \f$ g_{t+\Delta t} = \rho\, g_t + \sqrt{1 - \rho^2}\, \epsilon,\quad
 *       \rho = e^{-\Delta t / \tau},\quad \epsilon \sim N(0, 1) \f$
 *
 * mapped through the probability integral transform
 *
 *   \f$ u = \Phi(g),\quad c = P^{-1}(k, u) / k \f$
 *
 * where \f$ \Phi \f$ is the standard normal CDF and \f$ P^{-1} \f$ the
 * inverse regularized lower incomplete gamma function. Because \f$ g_t \f$
 * is standard normal at every instant, \f$ c \f$ is exactly
 * \f$ \Gamma(k, 1/k) \f$ at every instant, so all Gamma-Gamma moment
 * identities (E[I] = 1, scintillation index) hold for any coherence time.
 *
 * A coherence time of zero (the default) makes the component i.i.d. across
 * calls. \f$ \Delta t \f$ is measured on the simulator clock between
 * successive GetSample() calls; two calls at the same simulation time with
 * \f$ \tau > 0 \f$ return the same component values (the process is a
 * function of time). Use one instance per statistically independent path
 * (e.g. one per link direction).
 *
 * The shape parameters are passed per call, so they may track link geometry;
 * the latent Gaussian state is shape-independent and survives such changes.
 */
class CorrelatedGammaGammaFading : public Object
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    CorrelatedGammaGammaFading();
    ~CorrelatedGammaGammaFading() override;

    /**
     * \brief Draw the irradiance value at the current simulation time.
     *
     * Advances both latent AR(1) states by the simulator time elapsed since
     * the previous call and returns \f$ I = X Y \f$ with exact unit-mean
     * Gamma-Gamma marginal.
     *
     * \param alpha large-scale Gamma shape parameter, must be > 0
     * \param beta small-scale Gamma shape parameter, must be > 0
     * \return normalised irradiance sample (> 0)
     */
    double GetSample(double alpha, double beta);

    /**
     * \brief Assign a fixed stream number to the latent Gaussian variate.
     *
     * Same stream (and same call pattern) implies the same fading sequence.
     *
     * \param stream first stream index to use
     * \return the number of stream indices assigned (1)
     */
    int64_t AssignStreams(int64_t stream);

  private:
    /// Latent AR(1) state of one Gamma component.
    struct ComponentState
    {
        double g{0.0};          //!< Latent standard-normal value
        Time lastSampleTime{};  //!< Simulation time of the previous sample
        bool initialized{false}; //!< Whether g holds a drawn value
    };

    /**
     * \brief Evolve one component's latent state and return its Gamma value.
     * \param state the component's latent AR(1) state
     * \param shape Gamma shape parameter k, must be > 0
     * \param tau coherence time (zero means i.i.d.)
     * \return sample from Gamma(shape, 1/shape)
     */
    double SampleComponent(ComponentState& state, double shape, Time tau);

    Time m_tauLargeScale;         //!< Coherence time of the alpha component
    Time m_tauSmallScale;         //!< Coherence time of the beta component
    ComponentState m_largeScale;  //!< Latent state of the alpha component
    ComponentState m_smallScale;  //!< Latent state of the beta component
    Ptr<NormalRandomVariable> m_rng; //!< Latent AR(1) innovation source
};

} // namespace ns3

#endif /* CORRELATED_GAMMA_GAMMA_FADING_H */
