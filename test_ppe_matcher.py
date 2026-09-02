from ppe_matcher import PPEMatcher


def test_matcher_classifies_worker_statuses():
    matcher = PPEMatcher()

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
