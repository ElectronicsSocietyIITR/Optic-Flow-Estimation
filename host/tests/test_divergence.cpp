// Parity: fcw divergence vs collision_avoidance/fcw/divergence.py (golden file divergence.yml.gz).
#include <gtest/gtest.h>

#include "fcw/divergence.hpp"
#include "golden.hpp"

TEST(DivergenceParity, EveryCase) {
  golden::File g("divergence");
  const double dt = g.num("dt");
  const cv::Mat ppm = g.mat("pp"), bm = g.mat("box"), win = g.mat("window");
  const cv::Point2d pp(ppm.at<double>(0, 0), ppm.at<double>(0, 1));
  const fcw::Box box{bm.at<double>(0, 0), bm.at<double>(0, 1), bm.at<double>(0, 2), bm.at<double>(0, 3)};
  const cv::Rect window(cv::Point(static_cast<int>(win.at<double>(0, 0)), static_cast<int>(win.at<double>(0, 1))),
                        cv::Point(static_cast<int>(win.at<double>(0, 2)), static_cast<int>(win.at<double>(0, 3))));
  for (int i = 0; i < static_cast<int>(g.num("cases")); ++i) {
    const std::string k = "c" + std::to_string(i);
    SCOPED_TRACE(g.str(k + "_name"));
    cv::Mat flow = cv::Mat::zeros(384, 512, CV_32FC2);
    g.mat(k + "_flow").copyTo(flow(window));
    const double rot = g.num(k + "_rotdiv");

    const auto m = fcw::flowMoments(flow, box, pp);
    ASSERT_TRUE(m.has_value());
    const cv::Mat gm = g.mat(k + "_moments");
    for (int j = 0; j < 12; ++j) EXPECT_NEAR((*m)[j], gm.at<double>(0, j), 1e-9 * (1 + std::abs(gm.at<double>(0, j)))) << "term " << j;
    const auto plain = fcw::divergenceTtc(m, box, rot, 1, dt);
    ASSERT_TRUE(plain.has_value());
    EXPECT_NEAR(plain->eta, g.num(k + "_plain_eta"), 1e-9);
    EXPECT_NEAR(plain->var, g.num(k + "_plain_var"), 1e-12);

    const cv::Mat s = g.mat(k + "_samples");
    std::vector<std::array<int, 3>> samples;
    for (int r = 0; r < s.rows; ++r) samples.push_back({static_cast<int>(s.at<double>(r, 0)), static_cast<int>(s.at<double>(r, 1)), static_cast<int>(s.at<double>(r, 2))});
    const cv::Mat mask = fcw::robustMask(flow, box, pp, {}, samples);
    cv::Mat gmask = g.mat(k + "_mask"), mine;
    mask(window).convertTo(mine, CV_8U);
    EXPECT_EQ(cv::countNonZero(mine != gmask), 0);
    const auto robust = fcw::divergenceTtc(fcw::flowMoments(flow, box, pp, mask), box, rot, 1, dt);
    ASSERT_TRUE(robust.has_value());
    EXPECT_NEAR(robust->eta, g.num(k + "_robust_eta"), 1e-9);
  }
}

TEST(DivergenceParity, SharedLcgMatchesPython) {
  const auto s = fcw::ransacSamples(1000, 4);
  EXPECT_EQ(s[0], (std::array<int, 3>{502, 397, 989}));  // pinned in tests/fcw/test_divergence.py
}

TEST(DivergenceParity, ProductionRansacRejectsTheBackgroundLeak) {
  // Production path draws its own samples from the shared LCG: identical to Python, so exact.
  golden::File g("divergence");
  const cv::Mat ppm = g.mat("pp"), bm = g.mat("box"), win = g.mat("window");
  const cv::Point2d pp(ppm.at<double>(0, 0), ppm.at<double>(0, 1));
  const fcw::Box box{bm.at<double>(0, 0), bm.at<double>(0, 1), bm.at<double>(0, 2), bm.at<double>(0, 3)};
  const cv::Rect window(cv::Point(static_cast<int>(win.at<double>(0, 0)), static_cast<int>(win.at<double>(0, 1))),
                        cv::Point(static_cast<int>(win.at<double>(0, 2)), static_cast<int>(win.at<double>(0, 3))));
  cv::Mat flow = cv::Mat::zeros(384, 512, CV_32FC2);
  g.mat("c2_flow").copyTo(flow(window));  // case 2: background leak
  const auto m = fcw::divergenceTtc(fcw::flowMoments(flow, box, pp, fcw::robustMask(flow, box, pp)), box, 0.0, 1, g.num("dt"));
  ASSERT_TRUE(m.has_value());
  EXPECT_NEAR(m->eta, g.num("c2_robust_eta"), 1e-6);
}

TEST(DivergenceParity, EmptyBoxHasNoMoments) {
  cv::Mat flow = cv::Mat::zeros(384, 512, CV_32FC2);
  EXPECT_FALSE(fcw::flowMoments(flow, {600, 10, 700, 50}, {256, 192}).has_value());
  EXPECT_FALSE(fcw::divergenceTtc(std::nullopt, {0, 0, 10, 10}, 0.0, 1, 1 / 30.0).has_value());
}

TEST(DivergenceParity, PredictedMask) {
  golden::File g("divergence");
  const cv::Mat ppm = g.mat("pp"), bm = g.mat("box"), win = g.mat("window");
  const cv::Point2d pp(ppm.at<double>(0, 0), ppm.at<double>(0, 1));
  const fcw::Box box{bm.at<double>(0, 0), bm.at<double>(0, 1), bm.at<double>(0, 2), bm.at<double>(0, 3)};
  const cv::Rect window(cv::Point(static_cast<int>(win.at<double>(0, 0)), static_cast<int>(win.at<double>(0, 1))),
                        cv::Point(static_cast<int>(win.at<double>(0, 2)), static_cast<int>(win.at<double>(0, 3))));
  for (int i = 0; i < static_cast<int>(g.num("cases")); ++i) {
    const std::string k = "c" + std::to_string(i);
    cv::Mat flow(384, 512, CV_32FC2, cv::Scalar(0, 0));
    g.mat(k + "_flow").copyTo(flow(window));
    const cv::Mat flow_q = fcw::quantiseFlow(flow);
    cv::Mat qwin;
    flow_q(window).convertTo(qwin, CV_64F);
    EXPECT_EQ(cv::norm(qwin, g.mat(k + "_flow_q"), cv::NORM_INF), 0.0) << k << " quantised flow";
    const cv::Mat fm = g.mat(k + "_fit"), fq = g.mat(k + "_fit_q");
    const fcw::FitQ fit = fcw::fitToFixed({fm.at<double>(0, 0), fm.at<double>(0, 1), fm.at<double>(0, 2)},
                                          {fm.at<double>(0, 3), fm.at<double>(0, 4), fm.at<double>(0, 5)});
    const int64_t mine[6] = {fit.a0, fit.a1, fit.a2, fit.b0, fit.b1, fit.b2};
    for (int j = 0; j < 6; ++j) EXPECT_EQ(mine[j], static_cast<int64_t>(fq.at<double>(0, j))) << k << " fit " << j;
    const auto rect = fcw::flowRect(box, 0.10, 512, 384);
    ASSERT_TRUE(rect.has_value());
    const cv::Mat pred = fcw::predictedMaskFixed(flow_q, *rect, cv::Point(256, 192), fit, 64);
    EXPECT_EQ(cv::norm(pred, g.mat(k + "_pred_mask"), cv::NORM_INF), 0.0) << k << " predicted mask";
    const auto [mask, used] = fcw::divergenceMask(flow, flow_q, box, pp, std::make_pair(cv::Vec3d(fm.at<double>(0, 0), fm.at<double>(0, 1), fm.at<double>(0, 2)),
                                                                                    cv::Vec3d(fm.at<double>(0, 3), fm.at<double>(0, 4), fm.at<double>(0, 5))), 1, {});
    EXPECT_EQ(used, g.num(k + "_pred_used") != 0) << k;
    const auto moments = fcw::flowMomentsFixed(flow_q, box, pp, 0.10, mask);
    ASSERT_TRUE(moments.has_value());
    const cv::Mat ref = g.mat(k + "_fixed_moments");
    for (int j = 0; j < 12; ++j) EXPECT_EQ((*moments)[j], static_cast<int64_t>(ref.at<double>(0, j))) << k << " moment " << j;
    const auto stale = fcw::divergenceMask(flow, flow_q, box, pp, std::make_pair(cv::Vec3d(fm.at<double>(0, 0), fm.at<double>(0, 1), fm.at<double>(0, 2)),
                                                                               cv::Vec3d(fm.at<double>(0, 3), fm.at<double>(0, 4), fm.at<double>(0, 5))), 2, {});
    EXPECT_EQ(stale.second, g.num(k + "_stale_used") != 0) << k << " stale";
  }
}

// quantise_flow in divergence.py: np.round (half to even) of flow * 64, clipped to int16.
TEST(QuantiseFlow, RoundsHalfToEvenAndSaturatesLikeNumpy) {
  const float px[] = {0.5f / 64, 1.5f / 64, 2.5f / 64, -0.5f / 64, -1.5f / 64, 0.3f / 64, 511.99f, 600.0f, -600.0f, -512.0f};
  const short want_u[] = {0, 2, 2, 0, -2, 0, 32767, 32767, -32768, -32768};
  const short want_v[] = {0, -2, -2, 0, 2, 0, -32767, -32768, 32767, 32767};  // v = -u
  cv::Mat flow(1, 10, CV_32FC2);
  for (int i = 0; i < 10; ++i) flow.at<cv::Vec2f>(0, i) = {px[i], -px[i]};
  const cv::Mat q = fcw::quantiseFlow(flow);
  ASSERT_EQ(q.type(), CV_16SC2);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(q.at<cv::Vec2s>(0, i)[0], want_u[i]) << px[i];
    EXPECT_EQ(q.at<cv::Vec2s>(0, i)[1], want_v[i]) << -px[i];
  }
}

TEST(QuantiseFlow, MatchesTheScalarDefinitionOnAFullFrame) {
  cv::Mat flow(288, 512, CV_32FC2);
  cv::RNG rng(7);
  rng.fill(flow, cv::RNG::UNIFORM, -40.0, 40.0);
  for (int i = 0; i < 2000; ++i)  // exact ties: k/128 px
    flow.at<cv::Vec2f>(rng.uniform(0, 288), rng.uniform(0, 512)) = {rng.uniform(-2000, 2000) / 128.0f, rng.uniform(-2000, 2000) / 128.0f};
  const cv::Mat q = fcw::quantiseFlow(flow);
  for (int y = 0; y < flow.rows; ++y)
    for (int x = 0; x < flow.cols; ++x)
      for (int c = 0; c < 2; ++c) {
        const float f = flow.at<cv::Vec2f>(y, x)[c];
        const short want = static_cast<short>(std::clamp(std::nearbyint(f * 64.0f), -32768.0f, 32767.0f));
        ASSERT_EQ(q.at<cv::Vec2s>(y, x)[c], want) << y << "," << x << "," << c << " flow " << f;
      }
}

// The Vitis LK kernel's flow word: u in the upper 16 bits, v in the lower 16, both 1/64 px.
TEST(UnpackFlowWords, SplitsTheVitisWordIntoUAndV) {
  const short u[] = {0, 64, -64, 32767, -32768, 1, -1, 1000};
  const short v[] = {0, -32, 96, -32768, 32767, -1, 1, -1000};
  std::vector<uint32_t> words;
  for (int i = 0; i < 8; ++i) words.push_back(static_cast<uint32_t>(static_cast<uint16_t>(u[i])) << 16 | static_cast<uint16_t>(v[i]));
  const cv::Mat q = fcw::unpackFlowWords(words.data(), cv::Size(4, 2));
  ASSERT_EQ(q.type(), CV_16SC2);
  ASSERT_EQ(q.size(), cv::Size(4, 2));
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(q.at<cv::Vec2s>(i / 4, i % 4)[0], u[i]) << i;
    EXPECT_EQ(q.at<cv::Vec2s>(i / 4, i % 4)[1], v[i]) << i;
  }
  cv::Mat px;
  q.convertTo(px, CV_32FC2, 1.0 / fcw::kFlowScale);  // and back: exact (k / 64 is exact in float)
  EXPECT_EQ(cv::norm(fcw::quantiseFlow(px), q, cv::NORM_INF), 0.0);
}
