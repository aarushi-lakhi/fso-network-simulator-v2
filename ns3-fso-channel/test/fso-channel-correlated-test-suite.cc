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

#include "ns3/constant-position-mobility-model.h"
#include "ns3/correlated-gamma-gamma-fading.h"
#include "ns3/double.h"
#include "ns3/gamma-gamma-fso-loss-model.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"
#include "ns3/test.h"

#include <cmath>
#include <vector>

using namespace ns3;

/**
 * \ingroup fso-channel-tests
 * \defgroup fso-channel-correlated-tests Correlated fading tests
 */

namespace
{

/// Sampling period of the generated fading series.
const Time g_dt = MilliSeconds(1);

/**
 * \ingroup fso-channel-correlated-tests
 * \brief Append one fader sample to a series and reschedule until done.
 *
 * \param fader the fading process to sample
 * \param alpha large-scale Gamma shape
 * \param beta small-scale Gamma shape
 * \param remaining samples still to draw, including this one
 * \param out output series
 */
void
SampleFaderSeries(Ptr<CorrelatedGammaGammaFading> fader,
                  double alpha,
                  double beta,
                  uint32_t remaining,
                  std::vector<double>* out)
{
    out->push_back(fader->GetSample(alpha, beta));
    if (remaining > 1)
    {
        Simulator::Schedule(g_dt, &SampleFaderSeries, fader, alpha, beta, remaining - 1, out);
    }
}

/**
 * \ingroup fso-channel-correlated-tests
 * \brief Append one loss model fading sample to a series and reschedule.
 *
 * \param model the loss model to sample
 * \param distance path length [m]
 * \param remaining samples still to draw, including this one
 * \param out output series
 */
void
SampleModelSeries(Ptr<GammaGammaFsoLossModel> model,
                  double distance,
                  uint32_t remaining,
                  std::vector<double>* out)
{
    out->push_back(model->GetFadingSample(distance));
    if (remaining > 1)
    {
        Simulator::Schedule(g_dt, &SampleModelSeries, model, distance, remaining - 1, out);
    }
}

/**
 * \brief Sample mean of a series.
 * \param series the series
 * \return the mean
 */
double
Mean(const std::vector<double>& series)
{
    double sum = 0.0;
    for (double v : series)
    {
        sum += v;
    }
    return sum / series.size();
}

/**
 * \brief Empirical scintillation index E[I^2]/E[I]^2 - 1 of a series.
 * \param series the series
 * \return the scintillation index
 */
double
ScintillationIndex(const std::vector<double>& series)
{
    double sum = 0.0;
    double sumSq = 0.0;
    for (double v : series)
    {
        sum += v;
        sumSq += v * v;
    }
    double mean = sum / series.size();
    double meanSq = sumSq / series.size();
    return meanSq / (mean * mean) - 1.0;
}

/**
 * \brief Lag-1 autocorrelation coefficient of a series.
 * \param series the series
 * \return the lag-1 autocorrelation in [-1, 1]
 */
double
Lag1Autocorrelation(const std::vector<double>& series)
{
    double mean = Mean(series);
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < series.size(); i++)
    {
        double centered = series[i] - mean;
        denominator += centered * centered;
        if (i + 1 < series.size())
        {
            numerator += centered * (series[i + 1] - mean);
        }
    }
    return numerator / denominator;
}

/**
 * \brief Create a fading process with the given coherence times.
 * \param tauLarge large-scale coherence time
 * \param tauSmall small-scale coherence time
 * \param stream RNG stream to assign
 * \return the process
 */
Ptr<CorrelatedGammaGammaFading>
MakeFader(Time tauLarge, Time tauSmall, int64_t stream)
{
    auto fader = CreateObject<CorrelatedGammaGammaFading>();
    fader->SetAttribute("CoherenceTimeLargeScale", TimeValue(tauLarge));
    fader->SetAttribute("CoherenceTimeSmallScale", TimeValue(tauSmall));
    fader->AssignStreams(stream);
    return fader;
}

} // namespace

/**
 * \ingroup fso-channel-correlated-tests
 * \brief With zero coherence times the marginal must match i.i.d. Gamma-Gamma.
 */
class FsoCorrelatedIidMarginalTestCase : public TestCase
{
  public:
    FsoCorrelatedIidMarginalTestCase()
        : TestCase("tau = 0: mean 1 and SI match the i.i.d. closed form")
    {
    }

  private:
    void DoRun() override
    {
        const double alpha = 4.0;
        const double beta = 2.0;
        const uint32_t nSamples = 200000;
        const double theoreticalSi = 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta);

        auto fader = MakeFader(Seconds(0), Seconds(0), 1);
        // tau = 0 means i.i.d. draws, no simulator clock needed
        std::vector<double> series;
        series.reserve(nSamples);
        for (uint32_t i = 0; i < nSamples; i++)
        {
            series.push_back(fader->GetSample(alpha, beta));
        }
        Simulator::Destroy();

        NS_TEST_ASSERT_MSG_EQ_TOL(Mean(series), 1.0, 0.02, "E[I] != 1 for tau = 0");
        NS_TEST_ASSERT_MSG_EQ_TOL(ScintillationIndex(series),
                                  theoreticalSi,
                                  0.05 * theoreticalSi,
                                  "scintillation index mismatch for tau = 0");
    }
};

/**
 * \ingroup fso-channel-correlated-tests
 * \brief Positive coherence times must preserve the Gamma-Gamma marginal.
 */
class FsoCorrelatedMarginalTestCase : public TestCase
{
  public:
    FsoCorrelatedMarginalTestCase()
        : TestCase("tau > 0: long series keeps mean 1 and the SI identity")
    {
    }

  private:
    void DoRun() override
    {
        const double alpha = 4.0;
        const double beta = 2.0;
        const uint32_t nSamples = 400000;
        const double theoreticalSi = 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta);

        auto fader = MakeFader(MilliSeconds(20), MilliSeconds(20), 1);
        std::vector<double> series;
        series.reserve(nSamples);
        Simulator::Schedule(Seconds(0), &SampleFaderSeries, fader, alpha, beta, nSamples, &series);
        Simulator::Run();
        Simulator::Destroy();

        // Correlation shrinks the effective sample size, hence the looser
        // tolerances compared to the i.i.d. case
        NS_TEST_ASSERT_MSG_EQ_TOL(Mean(series), 1.0, 0.05, "E[I] != 1 for tau > 0");
        NS_TEST_ASSERT_MSG_EQ_TOL(ScintillationIndex(series),
                                  theoreticalSi,
                                  0.15 * theoreticalSi,
                                  "scintillation index mismatch for tau > 0");
    }
};

/**
 * \ingroup fso-channel-correlated-tests
 * \brief Lag-1 autocorrelation: ~0 for tau = 0, positive and increasing in tau.
 */
class FsoCorrelatedAutocorrelationTestCase : public TestCase
{
  public:
    FsoCorrelatedAutocorrelationTestCase()
        : TestCase("Lag-1 autocorrelation is ~0 for tau = 0 and grows with tau")
    {
    }

  private:
    void DoRun() override
    {
        const double alpha = 4.0;
        const double beta = 2.0;
        const uint32_t nSamples = 100000;

        auto faderIid = MakeFader(Seconds(0), Seconds(0), 1);
        auto faderShort = MakeFader(MilliSeconds(10), MilliSeconds(10), 2);
        auto faderLong = MakeFader(MilliSeconds(100), MilliSeconds(100), 3);

        std::vector<double> seriesIid;
        std::vector<double> seriesShort;
        std::vector<double> seriesLong;
        for (auto* series : {&seriesIid, &seriesShort, &seriesLong})
        {
            series->reserve(nSamples);
        }
        Simulator::Schedule(Seconds(0), &SampleFaderSeries, faderIid, alpha, beta, nSamples,
                            &seriesIid);
        Simulator::Schedule(Seconds(0), &SampleFaderSeries, faderShort, alpha, beta, nSamples,
                            &seriesShort);
        Simulator::Schedule(Seconds(0), &SampleFaderSeries, faderLong, alpha, beta, nSamples,
                            &seriesLong);
        Simulator::Run();
        Simulator::Destroy();

        double r1Iid = Lag1Autocorrelation(seriesIid);
        double r1Short = Lag1Autocorrelation(seriesShort);
        double r1Long = Lag1Autocorrelation(seriesLong);

        // Qualitative assertions only: the copula transform distorts the
        // exact exponential decay of the latent AR(1)
        NS_TEST_ASSERT_MSG_LT(std::abs(r1Iid), 0.05, "tau = 0 series is not uncorrelated");
        NS_TEST_ASSERT_MSG_GT(r1Short, 0.2, "tau = 10 dt series is not clearly correlated");
        NS_TEST_ASSERT_MSG_GT(r1Long,
                              r1Short,
                              "autocorrelation must increase with coherence time");
    }
};

/**
 * \ingroup fso-channel-correlated-tests
 * \brief The loss model with tau > 0 must keep the Gamma-Gamma marginal.
 */
class FsoLossModelCorrelatedMarginalTestCase : public TestCase
{
  public:
    FsoLossModelCorrelatedMarginalTestCase()
        : TestCase("Loss model with tau > 0 keeps mean 1 and the SI identity")
    {
    }

  private:
    void DoRun() override
    {
        const double distance = 1000.0;
        const uint32_t nSamples = 300000;

        auto model = CreateObject<GammaGammaFsoLossModel>();
        model->SetAttribute("C2n", DoubleValue(1e-13));
        model->SetAttribute("CoherenceTimeLargeScale", TimeValue(MilliSeconds(20)));
        model->SetAttribute("CoherenceTimeSmallScale", TimeValue(MilliSeconds(20)));
        model->AssignStreams(1);

        auto [alpha, beta] = model->GetAlphaBeta(distance);
        double theoreticalSi = 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta);

        std::vector<double> series;
        series.reserve(nSamples);
        Simulator::Schedule(Seconds(0), &SampleModelSeries, model, distance, nSamples, &series);
        Simulator::Run();
        Simulator::Destroy();

        NS_TEST_ASSERT_MSG_EQ_TOL(Mean(series), 1.0, 0.05, "E[I] != 1");
        NS_TEST_ASSERT_MSG_EQ_TOL(ScintillationIndex(series),
                                  theoreticalSi,
                                  0.15 * theoreticalSi,
                                  "scintillation index mismatch");
    }
};

/**
 * \ingroup fso-channel-correlated-tests
 * \brief Same streams and seed must reproduce the exact fading sequence.
 */
class FsoCorrelatedStreamDeterminismTestCase : public TestCase
{
  public:
    FsoCorrelatedStreamDeterminismTestCase()
        : TestCase("AssignStreams + same seed reproduces the fading sequence")
    {
    }

  private:
    /**
     * \brief Generate one fading series from freshly created objects.
     * \return the series
     */
    std::vector<double> GenerateSeries() const
    {
        const uint32_t nSamples = 2000;

        auto fader = MakeFader(MilliSeconds(50), MilliSeconds(5), 42);
        auto model = CreateObject<GammaGammaFsoLossModel>();
        model->SetAttribute("C2n", DoubleValue(1e-13));
        model->SetAttribute("CoherenceTimeLargeScale", TimeValue(MilliSeconds(50)));
        model->SetAttribute("CoherenceTimeSmallScale", TimeValue(MilliSeconds(5)));
        model->AssignStreams(7);

        std::vector<double> faderSeries;
        std::vector<double> modelSeries;
        faderSeries.reserve(nSamples);
        modelSeries.reserve(nSamples);
        Simulator::Schedule(Seconds(0), &SampleFaderSeries, fader, 4.0, 2.0, nSamples,
                            &faderSeries);
        Simulator::Schedule(Seconds(0), &SampleModelSeries, model, 1000.0, nSamples,
                            &modelSeries);
        Simulator::Run();
        Simulator::Destroy();

        faderSeries.insert(faderSeries.end(), modelSeries.begin(), modelSeries.end());
        return faderSeries;
    }

    void DoRun() override
    {
        std::vector<double> first = GenerateSeries();
        std::vector<double> second = GenerateSeries();

        NS_TEST_ASSERT_MSG_EQ(first.size(), second.size(), "series length mismatch");
        for (std::size_t i = 0; i < first.size(); i++)
        {
            NS_TEST_ASSERT_MSG_EQ(first[i],
                                  second[i],
                                  "sequence diverged at sample " << i);
        }
    }
};

/**
 * \ingroup fso-channel-correlated-tests
 * \brief Correlated FSO fading test suite.
 */
class FsoChannelCorrelatedTestSuite : public TestSuite
{
  public:
    FsoChannelCorrelatedTestSuite()
        : TestSuite("fso-channel-correlated", UNIT)
    {
        AddTestCase(new FsoCorrelatedIidMarginalTestCase, TestCase::QUICK);
        AddTestCase(new FsoCorrelatedMarginalTestCase, TestCase::QUICK);
        AddTestCase(new FsoCorrelatedAutocorrelationTestCase, TestCase::QUICK);
        AddTestCase(new FsoLossModelCorrelatedMarginalTestCase, TestCase::QUICK);
        AddTestCase(new FsoCorrelatedStreamDeterminismTestCase, TestCase::QUICK);
    }
};

/// Static test suite instance
static FsoChannelCorrelatedTestSuite g_fsoChannelCorrelatedTestSuite;
