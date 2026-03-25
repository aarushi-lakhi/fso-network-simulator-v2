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

#ifndef GAMMA_GAMMA_FSO_LOSS_MODEL_H
#define GAMMA_GAMMA_FSO_LOSS_MODEL_H

#include "ns3/propagation-loss-model.h"
#include "ns3/random-variable-stream.h"

#include <utility>

namespace ns3
{

/**
 * \defgroup fso-channel FSO Channel Models
 *
 * Free-Space Optical (FSO) channel models: atmospheric turbulence
 * (Gamma-Gamma scintillation) and Beer-Lambert extinction.
 */

/**
 * \ingroup fso-channel
 *
 * \brief Propagation loss model for FSO links under Gamma-Gamma turbulence.
 *
 * Models two effects along an optical line-of-sight path of length \f$d\f$:
 *
 * 1. Deterministic Beer-Lambert atmospheric extinction:
 *    \f$ L_{atm}(d) = e^{-\sigma_{ext} d} \f$
 *
 * 2. Random irradiance scintillation \f$I\f$ drawn from the Gamma-Gamma
 *    distribution (Andrews & Phillips, 2005), realised as the product of two
 *    independent Gamma variates with unit mean:
 *    \f$ I = X Y,\; X \sim \Gamma(\alpha, 1/\alpha),\;
 *        Y \sim \Gamma(\beta, 1/\beta),\; E[I] = 1 \f$
 *
 * The shape parameters \f$\alpha, \beta\f$ follow from the plane-wave Rytov
 * variance \f$ \sigma_R^2 = 1.23\, C_n^2\, k^{7/6} d^{11/6} \f$ evaluated at
 * the current node separation, so the fading statistics adapt automatically
 * to link distance and turbulence strength \f$C_n^2\f$.
 *
 * The received power is
 * \f$ P_{rx} = P_{tx} + 10\log_{10} e^{-\sigma_{ext} d} + 10\log_{10} I \f$ [dBm].
 *
 * Each call to CalcRxPower() draws a fresh fading sample, i.e. the model has
 * no temporal correlation; block-fading dynamics are provided by callers that
 * sample the model periodically (see FsoTopologyHelper).
 */
class GammaGammaFsoLossModel : public PropagationLossModel
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    GammaGammaFsoLossModel();
    ~GammaGammaFsoLossModel() override;

    // Delete copy constructor and assignment operator to avoid misuse
    GammaGammaFsoLossModel(const GammaGammaFsoLossModel&) = delete;
    GammaGammaFsoLossModel& operator=(const GammaGammaFsoLossModel&) = delete;

    /**
     * \brief Compute the plane-wave Rytov variance for a given path length.
     *
     * \f$ \sigma_R^2 = 1.23\, C_n^2\, k^{7/6} d^{11/6} \f$ with
     * \f$ k = 2\pi/\lambda \f$.
     *
     * \param distance path length [m], must be > 0
     * \return the Rytov variance (dimensionless)
     */
    double GetRytovVariance(double distance) const;

    /**
     * \brief Compute the Gamma-Gamma shape parameters for a given path length.
     *
     * Plane-wave closed forms (Andrews & Phillips, 2005, Eqs. 8.16-8.17):
     * \f$ \alpha = [e^{A} - 1]^{-1},\;
     *     A = 0.49\sigma_R^2 (1 + 1.11\sigma_R^{12/5})^{-7/6} \f$ and
     * \f$ \beta = [e^{B} - 1]^{-1},\;
     *     B = 0.51\sigma_R^2 (1 + 0.69\sigma_R^{12/5})^{-5/6} \f$.
     *
     * \param distance path length [m], must be > 0
     * \return pair (alpha, beta), both > 0
     */
    std::pair<double, double> GetAlphaBeta(double distance) const;

    /**
     * \brief Draw one normalised irradiance sample for a given path length.
     *
     * Returns \f$ I = X Y \f$ with \f$ X \sim \Gamma(\alpha, 1/\alpha) \f$,
     * \f$ Y \sim \Gamma(\beta, 1/\beta) \f$, so that \f$ E[I] = 1 \f$.
     * Returns exactly 1.0 when turbulence is disabled.
     *
     * \param distance path length [m], must be > 0
     * \return normalised irradiance sample (> 0)
     */
    double GetFadingSample(double distance);

    /**
     * \brief Compute the deterministic Beer-Lambert extinction loss.
     *
     * \param distance path length [m]
     * \return extinction loss \f$ -10\log_{10} e^{-\sigma_{ext} d} \f$ [dB], >= 0
     */
    double GetExtinctionLossDb(double distance) const;

  private:
    double DoCalcRxPower(double txPowerDbm,
                         Ptr<MobilityModel> a,
                         Ptr<MobilityModel> b) const override;
    int64_t DoAssignStreams(int64_t stream) override;

    double m_c2n;           //!< Refractive index structure parameter [m^(-2/3)]
    double m_wavelength;    //!< Optical wavelength [m]
    double m_extinction;    //!< Beer-Lambert extinction coefficient [1/m]
    bool m_turbulence;      //!< Whether Gamma-Gamma fading is applied
    Ptr<GammaRandomVariable> m_rng; //!< RNG for the two Gamma variates
};

} // namespace ns3

#endif /* GAMMA_GAMMA_FSO_LOSS_MODEL_H */
