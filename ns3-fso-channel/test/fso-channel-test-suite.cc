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

#include "ns3/boolean.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/double.h"
#include "ns3/gamma-gamma-fso-loss-model.h"
#include "ns3/test.h"

#include <cmath>

using namespace ns3;

/**
 * \ingroup fso-channel
 * \defgroup fso-channel-tests FSO channel module tests
 */

namespace
{

/**
 * \ingroup fso-channel-tests
 * \brief Create a loss model with fixed positions at the given separation.
 *
 * \param distance node separation [m]
 * \param mobilityA output: first endpoint
 * \param mobilityB output: second endpoint
 * \return the loss model
 */
Ptr<GammaGammaFsoLossModel>
MakeModel(double distance,
          Ptr<MobilityModel>& mobilityA,
          Ptr<MobilityModel>& mobilityB)
{
    mobilityA = CreateObject<ConstantPositionMobilityModel>();
    mobilityB = CreateObject<ConstantPositionMobilityModel>();
    mobilityA->SetPosition(Vector(0, 0, 0));
    mobilityB->SetPosition(Vector(distance, 0, 0));
    auto model = CreateObject<GammaGammaFsoLossModel>();
    model->AssignStreams(1);
    return model;
}

} // namespace

/**
 * \ingroup fso-channel-tests
 * \brief With turbulence disabled the loss must equal Beer-Lambert extinction.
 */
class FsoBeerLambertTestCase : public TestCase
{
  public:
    FsoBeerLambertTestCase()
        : TestCase("Deterministic loss matches Beer-Lambert extinction in dB")
    {
    }

  private:
    void DoRun() override
    {
        const double distance = 1200.0;
        const double sigmaExt = 5e-4;
        const double txPowerDbm = 10.0;

        Ptr<MobilityModel> a;
        Ptr<MobilityModel> b;
        auto model = MakeModel(distance, a, b);
        model->SetAttribute("TurbulenceEnabled", BooleanValue(false));
        model->SetAttribute("ExtinctionCoefficient", DoubleValue(sigmaExt));

        double rxPowerDbm = model->CalcRxPower(txPowerDbm, a, b);
        double expectedDbm = txPowerDbm + 10.0 * std::log10(std::exp(-sigmaExt * distance));

        NS_TEST_ASSERT_MSG_EQ_TOL(rxPowerDbm,
                                  expectedDbm,
                                  1e-9,
                                  "Beer-Lambert extinction mismatch");
    }
};

/**
 * \ingroup fso-channel-tests
 * \brief The linear fading gain must have unit mean (E[I] = 1).
 */
class FsoUnitMeanFadingTestCase : public TestCase
{
  public:
    FsoUnitMeanFadingTestCase()
        : TestCase("Mean linear fading gain is 1 over many draws")
    {
    }

  private:
    void DoRun() override
    {
        const double distance = 1000.0;
        const uint32_t nSamples = 100000;

        Ptr<MobilityModel> a;
        Ptr<MobilityModel> b;
        auto model = MakeModel(distance, a, b);
        model->SetAttribute("C2n", DoubleValue(1e-13)); // strong: worst-case spread

        double sum = 0.0;
        for (uint32_t i = 0; i < nSamples; i++)
        {
            sum += model->GetFadingSample(distance);
        }
        double mean = sum / nSamples;

        NS_TEST_ASSERT_MSG_EQ_TOL(mean, 1.0, 0.02, "E[I] != 1");
    }
};

/**
 * \ingroup fso-channel-tests
 * \brief Empirical scintillation index must match 1/a + 1/b + 1/(ab).
 */
class FsoScintillationIndexTestCase : public TestCase
{
  public:
    FsoScintillationIndexTestCase()
        : TestCase("Empirical SI matches theoretical 1/a + 1/b + 1/(ab)")
    {
    }

  private:
    void DoRun() override
    {
        const double distance = 1000.0;
        const uint32_t nSamples = 200000;

        Ptr<MobilityModel> a;
        Ptr<MobilityModel> b;
        auto model = MakeModel(distance, a, b);
        model->SetAttribute("C2n", DoubleValue(1e-13));

        auto [alpha, beta] = model->GetAlphaBeta(distance);
        double theoreticalSi = 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta);

        double sum = 0.0;
        double sumSq = 0.0;
        for (uint32_t i = 0; i < nSamples; i++)
        {
            double irradiance = model->GetFadingSample(distance);
            sum += irradiance;
            sumSq += irradiance * irradiance;
        }
        double mean = sum / nSamples;
        double meanSq = sumSq / nSamples;
        double empiricalSi = meanSq / (mean * mean) - 1.0;

        NS_TEST_ASSERT_MSG_EQ_TOL(empiricalSi,
                                  theoreticalSi,
                                  0.05 * theoreticalSi,
                                  "scintillation index mismatch");
    }
};

/**
 * \ingroup fso-channel-tests
 * \brief Path loss and turbulence strength must grow with distance.
 */
class FsoDistanceMonotonicTestCase : public TestCase
{
  public:
    FsoDistanceMonotonicTestCase()
        : TestCase("Loss and Rytov variance increase with distance")
    {
    }

  private:
    void DoRun() override
    {
        const double txPowerDbm = 10.0;

        Ptr<MobilityModel> a;
        Ptr<MobilityModel> b;
        auto model = MakeModel(500.0, a, b);
        model->SetAttribute("TurbulenceEnabled", BooleanValue(false));
        model->SetAttribute("ExtinctionCoefficient", DoubleValue(1e-4));

        double previousRxDbm = txPowerDbm;
        double previousRytov = 0.0;
        for (double distance : {500.0, 1000.0, 2000.0, 4000.0})
        {
            b->SetPosition(Vector(distance, 0, 0));
            double rxPowerDbm = model->CalcRxPower(txPowerDbm, a, b);
            double rytov = model->GetRytovVariance(distance);

            NS_TEST_ASSERT_MSG_LT(rxPowerDbm,
                                  previousRxDbm,
                                  "rx power must strictly decrease with distance");
            NS_TEST_ASSERT_MSG_GT(rytov,
                                  previousRytov,
                                  "Rytov variance must strictly increase with distance");
            previousRxDbm = rxPowerDbm;
            previousRytov = rytov;
        }
    }
};

/**
 * \ingroup fso-channel-tests
 * \brief FSO channel model test suite.
 */
class FsoChannelTestSuite : public TestSuite
{
  public:
    FsoChannelTestSuite()
        : TestSuite("fso-channel", UNIT)
    {
        AddTestCase(new FsoBeerLambertTestCase, TestCase::QUICK);
        AddTestCase(new FsoUnitMeanFadingTestCase, TestCase::QUICK);
        AddTestCase(new FsoScintillationIndexTestCase, TestCase::QUICK);
        AddTestCase(new FsoDistanceMonotonicTestCase, TestCase::QUICK);
    }
};

/// Static test suite instance
static FsoChannelTestSuite g_fsoChannelTestSuite;
