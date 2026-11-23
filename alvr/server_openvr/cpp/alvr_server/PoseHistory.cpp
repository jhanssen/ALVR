#include "PoseHistory.h"
#include "Logger.h"
#include "Utils.h"
#include "include/openvr_math.h"
#include <chrono>
#include <mutex>
#include <optional>

void PoseHistory::OnPoseUpdated(uint64_t targetTimestampNs, FfiDeviceMotion motion) {
    // Put pose history buffer
    TrackingHistoryFrame history;
    history.targetTimestampNs = targetTimestampNs;
    history.motion = motion;

    HmdMatrix_QuatToMat(
        motion.orientation.w,
        motion.orientation.x,
        motion.orientation.y,
        motion.orientation.z,
        &history.rotationMatrix
    );

    std::unique_lock<std::mutex> lock(m_mutex);
    if (!m_transformIdentity) {
        vr::HmdMatrix34_t rotation = vrmath::matMul33(m_transform, history.rotationMatrix);
        history.rotationMatrix = rotation;
    }

    if (m_poseBuffer.size() == 0) {
        m_poseBuffer.push_back(history);
    } else {
        if (m_poseBuffer.back().targetTimestampNs != targetTimestampNs) {
            // New track info
            m_poseBuffer.push_back(history);
        }
    }
    // The value should match with the client's MAXIMUM_TRACKING_FRAMES in ovr_context.cpp
    if (m_poseBuffer.size() > 120 * 3) {
        m_poseBuffer.pop_front();
    }
}

std::optional<PoseHistory::TrackingHistoryFrame>
PoseHistory::GetBestPoseMatch(const vr::HmdMatrix34_t& pose) const {
    std::unique_lock<std::mutex> lock(m_mutex);

    // The Linux capture layer reports an all-zero pose when it can't locate the compositor's
    // frame pose; matching against that is meaningless, so use the newest pose instead.
    float norm = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            norm += pose.m[j][i] * pose.m[j][i];
        }
    }
    if (norm < 0.5f) {
        if (m_poseBuffer.empty()) {
            return {};
        }
        static uint64_t fb_n = 0, fb_last_ns = 0;
        fb_n++;
        auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch()
        )
                          .count();
        if (now_ns - fb_last_ns > 1000000000ULL) {
            Info("PoseMatch: zero pose from layer, using newest (n=%llu)\n", (unsigned long long)fb_n);
            fb_last_ns = now_ns;
            fb_n = 0;
        }
        return m_poseBuffer.back();
    }

    float minDiff = 100000;
    auto minIt = m_poseBuffer.begin();
    for (auto it = m_poseBuffer.begin(); it != m_poseBuffer.end(); ++it) {
        float distance = 0;
        // Rotation matrix composes a part of ViewMatrix of TrackingInfo.
        // Be carefull of transpose.
        // And bottom side and right side of matrix should not be compared, because pPose does not
        // contain that part of matrix.
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                distance += pow(it->rotationMatrix.m[j][i] - pose.m[j][i], 2);
            }
        }
        if (minDiff > distance) {
            minIt = it;
            minDiff = distance;
        }
    }
    if (minIt != m_poseBuffer.end()) {
        static uint64_t dbg_n = 0, dbg_last_ns = 0, dbg_max_age = 0, dbg_sum_age = 0;
        static float dbg_max_dist = 0;
        uint64_t age = m_poseBuffer.back().targetTimestampNs - minIt->targetTimestampNs;
        dbg_n++;
        dbg_sum_age += age;
        if (age > dbg_max_age) dbg_max_age = age;
        if (minDiff > dbg_max_dist) dbg_max_dist = minDiff;
        auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch()
        )
                          .count();
        if (now_ns - dbg_last_ns > 1000000000ULL) {
            Info(
                "PoseMatch: n=%llu buf=%zu avg_age_ms=%.1f max_age_ms=%.1f max_dist=%g\n",
                (unsigned long long)dbg_n, m_poseBuffer.size(),
                dbg_n ? dbg_sum_age / 1e6 / dbg_n : 0.0, dbg_max_age / 1e6, dbg_max_dist
            );
            dbg_last_ns = now_ns;
            dbg_n = dbg_sum_age = dbg_max_age = 0;
            dbg_max_dist = 0;
        }
        return *minIt;
    }

    Debug("PoseHistory::GetBestPoseMatch: No pose matched.");
    return {};
}

std::optional<PoseHistory::TrackingHistoryFrame> PoseHistory::GetPoseAt(uint64_t timestampNs
) const {
    std::unique_lock<std::mutex> lock(m_mutex);
    for (auto it = m_poseBuffer.rbegin(), end = m_poseBuffer.rend(); it != end; ++it) {
        if (it->targetTimestampNs == timestampNs)
            return *it;
    }

    Debug("PoseHistory::GetPoseAt: No pose matched.");
    return {};
}

void PoseHistory::SetTransform(const vr::HmdMatrix34_t& transform) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_transform = transform;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (transform.m[i][j] != (i == j ? 1 : 0)) {
                m_transformIdentity = false;
                return;
            }
        }
    }
    m_transformIdentity = true;
}
