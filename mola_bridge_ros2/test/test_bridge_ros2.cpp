// Copyright 2022 Zeal Robotics

// What BridgeROS2 hands its two sides: the stamp of a ROS IMU reading as it
// reaches MOLA, and the /tf and registration messages a localization update
// becomes in ROS. The bridge runs as a MOLA system would run it, beside one
// localization source it discovers, and a node on the same graph plays the
// rest of the ROS system.

#include <gtest/gtest.h>
#include <mola_bridge_ros2/BridgeROS2.h>
#include <mola_kernel/MinimalModuleContainer.h>
#include <mola_kernel/interfaces/ExecutableBase.h>
#include <mola_kernel/interfaces/LocalizationSourceBase.h>
#include <mola_kernel/interfaces/RawDataConsumer.h>
#include <mrpt/containers/yaml.h>
#include <mrpt/math/CMatrixFixed.h>
#include <mrpt/math/CQuaternion.h>
#include <mrpt/obs/CObservationIMU.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/ros2bridge/time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <string>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/qos.hpp>
#include <thread>
#include <vector>

namespace mola
{
/** A localization source whose updates the test decides. */
class TestLocalizationSource : public ExecutableBase, public LocalizationSourceBase
{
  DEFINE_MRPT_OBJECT(TestLocalizationSource, mola)

 public:
  void initialize(const mrpt::containers::yaml&) override {}
  void spinOnce() override {}

  void emit(const LocalizationUpdate& l) { advertiseUpdatedLocalization(l); }
};
}  // namespace mola
IMPLEMENTS_MRPT_OBJECT(TestLocalizationSource, mola::ExecutableBase, mola)

namespace
{
using namespace std::chrono_literals;
using Update = mola::LocalizationSourceBase::LocalizationUpdate;

constexpr double TRANSFORM_TOLERANCE = 0.1;
constexpr double REBROADCAST_PERIOD  = 0.05;
const char*      IMU_TOPIC           = "/bridge_test/imu";

/** The observations the bridge forwards into MOLA. */
class ObservationSink : public mola::RawDataConsumer
{
 public:
  void onNewObservation(const mola::CObservation::ConstPtr& o) override
  {
    std::lock_guard<std::mutex> lck(mtx_);
    received_.push_back(o);
  }

  std::vector<mola::CObservation::ConstPtr> received() const
  {
    std::lock_guard<std::mutex> lck(mtx_);
    return received_;
  }

 private:
  mutable std::mutex                        mtx_;
  std::vector<mola::CObservation::ConstPtr> received_;
};

/** Every message a topic delivered, in arrival order, each with when it arrived.
 * The callback owns what it writes, so it may still run on the executor's
 * threads after the recorder is gone. */
template <class Msg>
class Recorder
{
 public:
  struct Entry
  {
    Msg          msg;
    rclcpp::Time arrival;
  };

  Recorder(const rclcpp::Node::SharedPtr& node, const std::string& topic, const rclcpp::QoS& qos)
  {
    sub_ = node->create_subscription<Msg>(
        topic, qos,
        [log = log_, clock = node->get_clock()](const Msg& m)
        {
          std::lock_guard<std::mutex> lck(log->mtx);
          log->entries.push_back({m, clock->now()});
        });
  }

  std::vector<Entry> entries() const
  {
    std::lock_guard<std::mutex> lck(log_->mtx);
    return log_->entries;
  }

 private:
  struct Log
  {
    std::mutex         mtx;
    std::vector<Entry> entries;
  };
  std::shared_ptr<Log>                          log_ = std::make_shared<Log>();
  typename rclcpp::Subscription<Msg>::SharedPtr sub_;
};

/** Repeats `act` until `done` holds or `timeout` passes, and says which. */
bool repeatUntil(
    const std::function<bool()>& done, const std::function<void()>& act,
    std::chrono::milliseconds timeout = 10s)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline)
  {
    if (done()) return true;
    act();
    std::this_thread::sleep_for(20ms);
  }
  return done();
}

bool waitFor(const std::function<bool()>& done, std::chrono::milliseconds timeout = 10s)
{
  return repeatUntil(
      done, [] {}, timeout);
}

mrpt::Clock::time_point simTime(double seconds) { return mrpt::Clock::fromDouble(seconds); }

builtin_interfaces::msg::Time rosTime(double seconds)
{
  return mrpt::ros2bridge::toROS(simTime(seconds));
}

double toSeconds(const builtin_interfaces::msg::Time& t) { return rclcpp::Time(t).seconds(); }

mrpt::poses::CPose3D poseOf(const geometry_msgs::msg::Transform& t)
{
  return mrpt::poses::CPose3D(
      mrpt::math::CQuaternionDouble(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z),
      t.translation.x, t.translation.y, t.translation.z);
}

geometry_msgs::msg::TransformStamped transformStamped(
    const std::string& parent, const std::string& child, double stamp,
    const mrpt::poses::CPose3D& p)
{
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = parent;
  tf.header.stamp    = rosTime(stamp);
  tf.child_frame_id  = child;
  mrpt::math::CQuaternionDouble q;
  p.getAsQuaternion(q);
  tf.transform.translation.x = p.x();
  tf.transform.translation.y = p.y();
  tf.transform.translation.z = p.z();
  tf.transform.rotation.w    = q.r();
  tf.transform.rotation.x    = q.x();
  tf.transform.rotation.y    = q.y();
  tf.transform.rotation.z    = q.z();
  return tf;
}

double poseDistance(const mrpt::poses::CPose3D& a, const mrpt::poses::CPose3D& b)
{
  using M       = mrpt::math::CMatrixDouble44;
  const auto ha = a.getHomogeneousMatrixVal<M>();
  const auto hb = b.getHomogeneousMatrixVal<M>();
  double     d  = 0.0;
  for (int r = 0; r < 4; r++)
  {
    for (int c = 0; c < 4; c++) d = std::max(d, std::abs(ha(r, c) - hb(r, c)));
  }
  return d;
}

void expectSamePose(const mrpt::poses::CPose3D& actual, const mrpt::poses::CPose3D& expected)
{
  EXPECT_LT(poseDistance(actual, expected), 1e-6)
      << "actual " << actual << " expected " << expected;
}

Update update(
    const std::string& method, const std::string& reference, const std::string& child, double stamp,
    const mrpt::poses::CPose3D& pose, double quality = 1.0)
{
  Update l;
  l.method          = method;
  l.reference_frame = reference;
  l.child_frame     = child;
  l.timestamp       = simTime(stamp);
  l.pose            = pose.asTPose();
  l.quality         = quality;
  return l;
}

class BridgeROS2Test : public ::testing::Test
{
 protected:
  void SetUp() override
  {
    tester_ = std::make_shared<rclcpp::Node>("bridge_test_peer");
    executor_.add_node(tester_);
    spinner_ = std::thread([this] { executor_.spin(); });
  }

  void TearDown() override
  {
    container_.reset();
    bridge_.reset();
    executor_.cancel();
    spinner_.join();
  }

  /** Starts the bridge as the vehicle runs it, and returns once its node is up. */
  void startBridge()
  {
    const auto cfg = mrpt::containers::yaml::FromText(
        "params:\n"
        "  base_link_frame: base_link\n"
        "  odom_frame: odom\n"
        "  reference_frame: map\n"
        "  base_footprint_frame: ''\n"
        "  publish_localization_following_rep105: true\n"
        "  publish_in_sim_time: true\n"
        "  publish_odometry_msgs_from_slam: true\n"
        "  publish_registration_msgs_from_slam_source: lidar_odometry\n"
        "  transform_tolerance: " +
        std::to_string(TRANSFORM_TOLERANCE) +
        "\n"
        "  transform_publish_period: " +
        std::to_string(REBROADCAST_PERIOD) +
        "\n"
        "  subscribe:\n"
        "    - topic: " +
        std::string(IMU_TOPIC) +
        "\n"
        "      msg_type: Imu\n"
        "      output_sensor_label: imu\n"
        "      fixed_sensor_pose: '0 0 0 0 0 0'\n");

    source_ = std::make_shared<mola::TestLocalizationSource>();
    bridge_ = std::make_shared<mola::BridgeROS2>();
    bridge_->setModuleInstanceName("ros2_bridge");
    container_ = std::make_unique<mola::MinimalModuleContainer>(
        std::vector<mola::ExecutableBase::Ptr>{bridge_, source_});
    bridge_->attachToDataConsumer(sink_);
    bridge_->initialize(cfg);

    // The bridge subscribes to its topics once its node and broadcasters exist.
    ASSERT_TRUE(waitFor([this] { return tester_->count_subscribers(IMU_TOPIC) > 0; }));
    // Its first spin discovers the localization source.
    bridge_->spinOnce();
  }

  rclcpp::Node::SharedPtr                       tester_;
  rclcpp::executors::MultiThreadedExecutor      executor_;
  std::thread                                   spinner_;
  ObservationSink                               sink_;
  std::shared_ptr<mola::TestLocalizationSource> source_;
  std::shared_ptr<mola::BridgeROS2>             bridge_;
  std::unique_ptr<mola::MinimalModuleContainer> container_;
};

std::vector<geometry_msgs::msg::TransformStamped> transformsBetween(
    const std::vector<Recorder<tf2_msgs::msg::TFMessage>::Entry>& entries,
    const std::string& parent, const std::string& child,
    std::vector<rclcpp::Time>* arrivals = nullptr)
{
  std::vector<geometry_msgs::msg::TransformStamped> out;
  for (const auto& e : entries)
  {
    for (const auto& t : e.msg.transforms)
    {
      if (t.header.frame_id == parent && t.child_frame_id == child)
      {
        out.push_back(t);
        if (arrivals) arrivals->push_back(e.arrival);
      }
    }
  }
  return out;
}

// The front end pairs IMU readings with the scans they cover by stamp, so a
// reading reaches MOLA at the instant its header carries, not when it arrived.
TEST_F(BridgeROS2Test, ImuObservationCarriesTheMessageHeaderStamp)
{
  ASSERT_NO_FATAL_FAILURE(startBridge());

  auto pub = tester_->create_publisher<sensor_msgs::msg::Imu>(IMU_TOPIC, rclcpp::SensorDataQoS());
  sensor_msgs::msg::Imu imu;
  imu.header.frame_id = "imu_link";
  imu.header.stamp    = rosTime(1000.25);
  imu.orientation.w   = 1.0;

  ASSERT_TRUE(
      repeatUntil([this] { return !sink_.received().empty(); }, [&] { pub->publish(imu); }));

  const auto obs =
      std::dynamic_pointer_cast<const mrpt::obs::CObservationIMU>(sink_.received().front());
  ASSERT_TRUE(obs);
  EXPECT_EQ(obs->sensorLabel, "imu");
  EXPECT_EQ(obs->timestamp.time_since_epoch().count(), simTime(1000.25).time_since_epoch().count());
}

// REP-105: map -> odom is the localized pose composed with the odometry at the
// instant the localizer measured, not the latest odometry, which has moved on.
TEST_F(BridgeROS2Test, MapToOdomComposesTheOdometryAtTheLocalizedInstant)
{
  ASSERT_NO_FATAL_FAILURE(startBridge());
  Recorder<tf2_msgs::msg::TFMessage> tf(tester_, "/tf", tf2_ros::DynamicBroadcasterQoS());

  const auto odomAtT1 = mrpt::poses::CPose3D::FromXYZYawPitchRoll(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  const auto odomAtT2 =
      mrpt::poses::CPose3D::FromXYZYawPitchRoll(5.0, 2.0, 0.0, mrpt::DEG2RAD(90.0), 0.0, 0.0);
  const auto mapToBase =
      mrpt::poses::CPose3D::FromXYZYawPitchRoll(10.0, -3.0, 0.0, mrpt::DEG2RAD(30.0), 0.0, 0.0);
  const double t1 = 2000.0;
  const double t2 = 2001.0;

  auto odomPub =
      tester_->create_publisher<tf2_msgs::msg::TFMessage>("/tf", tf2_ros::DynamicBroadcasterQoS());
  tf2_msgs::msg::TFMessage odom;
  odom.transforms = {
      transformStamped("odom", "base_link", t1, odomAtT1),
      transformStamped("odom", "base_link", t2, odomAtT2)};

  // Whether a map -> odom published after the first `skip` ones is `expected`.
  const auto mapToOdomIs = [&](const mrpt::poses::CPose3D& expected, size_t skip = 0)
  {
    const auto published = transformsBetween(tf.entries(), "map", "odom");
    for (size_t i = skip; i < published.size(); i++)
    {
      if (poseDistance(poseOf(published[i].transform), expected) < 1e-6) return true;
    }
    return false;
  };

  // Once a localization at t2 composes with the odometry at t2, the bridge
  // holds both odometry samples.
  ASSERT_TRUE(repeatUntil(
      [&] { return mapToOdomIs(mapToBase + (-odomAtT2)); },
      [&]
      {
        odomPub->publish(odom);
        source_->emit(update("lidar_odometry", "map", "base_link", t2, mapToBase));
      }));

  // Before it held both, it may have composed against t1's as the latest.
  const size_t probed = transformsBetween(tf.entries(), "map", "odom").size();
  EXPECT_TRUE(repeatUntil(
      [&] { return mapToOdomIs(mapToBase + (-odomAtT1), probed); },
      [&] { source_->emit(update("lidar_odometry", "map", "base_link", t1, mapToBase)); }, 3s))
      << "a localization at t1 did not compose with the odometry at t1";
}

// map -> odom is a slowly varying correction, so it is stamped ahead of now by
// the transform tolerance and rebroadcast between updates: a lookup of
// map -> anything at a fresh sensor stamp then never extrapolates past it.
// That holds both for the bridge's own REP-105 composition and for a state
// estimator that publishes map -> odom itself.
void expectStampedAheadAndRebroadcast(
    const std::function<void()>& emitOnce,
    const std::function<
        std::vector<geometry_msgs::msg::TransformStamped>(std::vector<rclcpp::Time>*)>& mapToOdom)
{
  ASSERT_TRUE(repeatUntil([&] { return !mapToOdom(nullptr).empty(); }, emitOnce));

  const size_t before = mapToOdom(nullptr).size();
  std::this_thread::sleep_for(std::chrono::duration<double>(10 * REBROADCAST_PERIOD));

  std::vector<rclcpp::Time> arrivals;
  const auto                all = mapToOdom(&arrivals);
  EXPECT_GE(all.size(), before + 3) << "map -> odom was not rebroadcast between updates";

  // Stamped at now + tolerance, each arrives the tolerance less its latency
  // ahead of the receiver's clock. The largest lead is the least delayed one's.
  double maxLead = -1e300;
  for (size_t i = 0; i < all.size(); i++)
  {
    const double lead = toSeconds(all[i].header.stamp) - arrivals[i].seconds();
    EXPECT_LE(lead, TRANSFORM_TOLERANCE) << "map -> odom #" << i << " is stamped too far ahead";
    maxLead = std::max(maxLead, lead);
  }
  EXPECT_GT(maxLead, 0.0) << "map -> odom is not stamped ahead of now";
}

TEST_F(BridgeROS2Test, ComposedMapToOdomIsStampedAheadAndRebroadcast)
{
  ASSERT_NO_FATAL_FAILURE(startBridge());
  Recorder<tf2_msgs::msg::TFMessage> tf(tester_, "/tf", tf2_ros::DynamicBroadcasterQoS());
  auto                               odomPub = tester_->create_publisher<tf2_msgs::msg::TFMessage>(
      "/tf_static", tf2_ros::StaticBroadcasterQoS());
  tf2_msgs::msg::TFMessage odom;
  odom.transforms = {transformStamped(
      "odom", "base_link", 0.0, mrpt::poses::CPose3D::FromXYZYawPitchRoll(1, 2, 0, 0, 0, 0))};
  odomPub->publish(odom);

  expectStampedAheadAndRebroadcast(
      [&] {
        source_->emit(update("lidar_odometry", "map", "base_link", 3000.0, mrpt::poses::CPose3D()));
      },
      [&](std::vector<rclcpp::Time>* arrivals)
      { return transformsBetween(tf.entries(), "map", "odom", arrivals); });
}

TEST_F(BridgeROS2Test, EstimatorMapToOdomIsStampedAheadAndRebroadcast)
{
  ASSERT_NO_FATAL_FAILURE(startBridge());
  Recorder<tf2_msgs::msg::TFMessage> tf(tester_, "/tf", tf2_ros::DynamicBroadcasterQoS());

  const auto correction =
      mrpt::poses::CPose3D::FromXYZYawPitchRoll(0.5, -0.25, 0.0, mrpt::DEG2RAD(5.0), 0.0, 0.0);

  expectStampedAheadAndRebroadcast(
      [&]
      { source_->emit(update("state_estimation/map_odom", "map", "odom", 3000.0, correction)); },
      [&](std::vector<rclcpp::Time>* arrivals)
      { return transformsBetween(tf.entries(), "map", "odom", arrivals); });

  const auto published = transformsBetween(tf.entries(), "map", "odom");
  ASSERT_FALSE(published.empty());
  expectSamePose(poseOf(published.back().transform), correction);
}

// The named source's accepted registrations, and nothing else, reach
// "<source>/registration", at the instant and in the frames they were measured.
TEST_F(BridgeROS2Test, AcceptedRegistrationIsPublishedAsMeasured)
{
  ASSERT_NO_FATAL_FAILURE(startBridge());
  Recorder<nav_msgs::msg::Odometry> registration(
      tester_, "lidar_odometry/registration", rclcpp::SystemDefaultsQoS());

  const auto pose =
      mrpt::poses::CPose3D::FromXYZYawPitchRoll(4.0, 1.5, 0.2, mrpt::DEG2RAD(-40.0), 0.0, 0.0);
  auto                        l = update("lidar_odometry", "map", "base_link", 4000.5, pose, 0.8);
  mrpt::math::CMatrixDouble66 cov;
  cov.setDiagonal(6, 0.01);
  l.cov = cov;

  ASSERT_TRUE(
      repeatUntil([&] { return !registration.entries().empty(); }, [&] { source_->emit(l); }));

  const auto msg = registration.entries().front().msg;
  EXPECT_EQ(msg.header.frame_id, "map");
  EXPECT_EQ(msg.child_frame_id, "base_link");
  EXPECT_DOUBLE_EQ(toSeconds(msg.header.stamp), 4000.5);
  const auto& p = msg.pose.pose;
  expectSamePose(
      mrpt::poses::CPose3D(
          mrpt::math::CQuaternionDouble(
              p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z),
          p.position.x, p.position.y, p.position.z),
      pose);
  EXPECT_DOUBLE_EQ(msg.pose.covariance[0], 0.01);
  EXPECT_DOUBLE_EQ(msg.pose.covariance[7], 0.01);
}

// A prediction (quality 0) and another source's update are each published on
// their source's pose topic, which is the control that they were delivered,
// and on no registration topic.
TEST_F(BridgeROS2Test, PredictionsAndOtherSourcesPublishNoRegistration)
{
  ASSERT_NO_FATAL_FAILURE(startBridge());

  struct Case
  {
    Update      l;
    std::string what;
  };
  const std::vector<Case> cases = {
      {update("lidar_odometry", "map", "base_link", 5000.0, mrpt::poses::CPose3D(), 0.0),
       "a prediction of the named source"},
      {update("state_estimation", "map", "base_link", 5000.0, mrpt::poses::CPose3D(), 1.0),
       "an update of another source"}};

  for (const auto& c : cases)
  {
    Recorder<nav_msgs::msg::Odometry> pose(
        tester_, c.l.method + "/pose", rclcpp::SystemDefaultsQoS());
    Recorder<nav_msgs::msg::Odometry> registration(
        tester_, c.l.method + "/registration", rclcpp::SystemDefaultsQoS());

    // Both topics' publishers would be created by the same update, so once the
    // pose has arrived several times, a registration would have too.
    ASSERT_TRUE(
        repeatUntil([&] { return pose.entries().size() >= 5; }, [&] { source_->emit(c.l); }))
        << c.what;
    EXPECT_TRUE(registration.entries().empty()) << c.what << " was published as a registration";
  }
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
