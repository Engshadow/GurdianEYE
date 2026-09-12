from dataclasses import dataclass


@dataclass
class TimingConfig:

    violation_seconds: float = 1.0
    filter_seconds: float = 0.5
    stability_seconds: float = 0.6
    source_fps: float = 30.0
    playback_slowdown: float = 1.0

    @property
    def temporal_filter_frames(self):
        return max(2, round(self.filter_seconds * self.source_fps))

    @property
    def stability_frames(self):
        return max(1, round(self.stability_seconds * self.source_fps))

    @property
    def violation_duration_wall(self):
        return self.violation_seconds * max(0.01, self.playback_slowdown)
