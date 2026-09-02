from ppe_matcher import MatcherConfig, PPEMatcher


def test_matcher_classifies_worker_statuses():
    matcher = PPEMatcher(MatcherConfig(temporal_filter_frames=1))

    detections = [
        {"class": "person", "confidence": 0.95, "bbox": [50, 80, 190, 430]},
        {"class": "helmet", "confidence": 0.91, "bbox": [90, 90, 150, 150]},
        {"class": "vest", "confidence": 0.88, "bbox": [75, 180, 170, 300]},

        {"class": "person", "confidence": 0.93, "bbox": [300, 90, 440, 440]},
        {"class": "vest", "confidence": 0.86, "bbox": [325, 190, 420, 310]},
    ]

    results = matcher.match(detections)

    assert len(results) == 2
    assert results[0]["status"] == "SAFE"
    assert results[1]["status"] == "NO_HELMET"
    assert results[1]["missing_ppe"] == ["helmet"]


def test_matcher_requires_four_consecutive_violation_frames():
    matcher = PPEMatcher()
    detections = [
        {"class": "person", "confidence": 0.95, "bbox": [50, 80, 190, 430], "track_id": 42},
        {"class": "vest", "confidence": 0.88, "bbox": [75, 180, 170, 300]},
    ]

    statuses = [matcher.match(detections)[0]["status"] for _ in range(4)]

    assert statuses[:3] == ["UNKNOWN", "UNKNOWN", "UNKNOWN"]
    assert statuses[3] == "NO_HELMET"
    assert matcher.match(detections)[0]["person_id"] == 42


def test_matcher_resets_filter_after_safe_frame():
    matcher = PPEMatcher(MatcherConfig(stability_frames=0))
    violating = [
        {"class": "person", "confidence": 0.95, "bbox": [50, 80, 190, 430], "track_id": 7},
        {"class": "vest", "confidence": 0.88, "bbox": [75, 180, 170, 300]},
    ]
    safe = violating + [
        {"class": "helmet", "confidence": 0.91, "bbox": [90, 90, 150, 150]},
    ]

    for _ in range(4):
        matcher.match(violating)
    assert matcher.match(safe)[0]["status"] == "SAFE"
    assert matcher.match(violating)[0]["status"] == "UNKNOWN"
