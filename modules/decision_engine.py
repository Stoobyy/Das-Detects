"""
Decision Engine — Frame Smoothing & Notification Logic
=======================================================
Buffers ensemble scores across multiple audio frames,
computes a smoothed average, classifies the result,
and triggers one-shot notifications on state transitions.

Designed for real-time streaming inference — uses a deque
for O(1) appends with automatic eviction of oldest frames.
"""

from collections import deque
from typing import NamedTuple, Optional


class FrameResult(NamedTuple):
    """Result returned after each frame update."""
    ensemble_score: float       # raw ensemble score for this frame (0–1)
    average_score: float        # smoothed average across the buffer (0–1)
    label: str                  # "Human", "Suspicious", or "AI Generated"
    should_notify: bool         # True only on first AI detection after buffer fill
    buffer_full: bool           # True when buffer has reached capacity
    frames_buffered: int        # current number of scores in the buffer


class DecisionEngine:
    """
    Smooths per-frame CNN + GMM predictions using a rolling buffer.

    Parameters
    ----------
    buffer_size : int
        Number of recent ensemble scores to keep (default 5).
    cnn_weight : float
        Weight for the CNN probability in the ensemble (default 0.7).
    gmm_weight : float
        Weight for the GMM probability in the ensemble (default 0.3).
    human_threshold : float
        Scores below this are classified "Human" (default 0.40).
    ai_threshold : float
        Scores at or above this are classified "AI Generated" (default 0.70).
    """

    # ─── Classification labels ────────────────────────────
    LABEL_HUMAN = "Human"
    LABEL_SUSPICIOUS = "Suspicious"
    LABEL_AI = "AI Generated"

    def __init__(
        self,
        buffer_size: int = 5,
        cnn_weight: float = 0.7,
        gmm_weight: float = 0.3,
        human_threshold: float = 0.40,
        ai_threshold: float = 0.70,
    ):
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if not (0 <= cnn_weight <= 1 and 0 <= gmm_weight <= 1):
            raise ValueError("Weights must be in [0, 1]")

        self._buffer_size = buffer_size
        self._cnn_weight = cnn_weight
        self._gmm_weight = gmm_weight
        self._human_threshold = human_threshold
        self._ai_threshold = ai_threshold

        # Rolling buffer of ensemble scores
        self._buffer: deque[float] = deque(maxlen=buffer_size)

        # State tracking for notification logic
        self._previous_label: str = self.LABEL_HUMAN

    # ─── Public API ───────────────────────────────────────

    def update(
        self,
        cnn_probability: Optional[float] = None,
        gmm_probability: Optional[float] = None,
    ) -> FrameResult:
        """
        Process one frame's model outputs.

        Parameters
        ----------
        cnn_probability : float or None
            CNN's AI-voice probability (0–1).
        gmm_probability : float or None
            GMM's AI-voice probability (0–1).

        Returns
        -------
        FrameResult
            Named tuple with ensemble score, smoothed average,
            classification label, and notification flag.
        """
        ensemble = self._compute_ensemble(cnn_probability, gmm_probability)
        self._buffer.append(ensemble)

        avg = self._compute_average()
        label = self._classify(avg)

        # Notification: only on first transition to AI after buffer fill
        should_notify = (
            self.buffer_full
            and label == self.LABEL_AI
            and self._previous_label != self.LABEL_AI
        )

        # Only update state tracking once buffer is full,
        # so partial fills don't prematurely set "AI Generated"
        if self.buffer_full:
            self._previous_label = label

        return FrameResult(
            ensemble_score=ensemble,
            average_score=avg,
            label=label,
            should_notify=should_notify,
            buffer_full=self.buffer_full,
            frames_buffered=len(self._buffer),
        )

    def reset(self) -> None:
        """Clear the buffer and reset state (call on stop / call-end)."""
        self._buffer.clear()
        self._previous_label = self.LABEL_HUMAN

    # ─── Properties ───────────────────────────────────────

    @property
    def buffer_full(self) -> bool:
        """True when the buffer has reached its configured capacity."""
        return len(self._buffer) >= self._buffer_size

    @property
    def average_score(self) -> float:
        """Current smoothed average across buffered scores."""
        return self._compute_average()

    @property
    def label(self) -> str:
        """Current classification label based on the smoothed average."""
        return self._classify(self._compute_average())

    @property
    def buffer_size(self) -> int:
        """Configured buffer capacity."""
        return self._buffer_size

    @property
    def frames_buffered(self) -> int:
        """Number of scores currently in the buffer."""
        return len(self._buffer)

    @property
    def buffer_scores(self) -> list[float]:
        """Copy of all ensemble scores currently in the buffer (oldest first)."""
        return list(self._buffer)

    def classify_score(self, score: float) -> str:
        """Classify a single score — exposes threshold logic for per-frame coloring."""
        return self._classify(score)

    # ─── Internal helpers ─────────────────────────────────

    def _compute_ensemble(
        self,
        cnn_prob: Optional[float],
        gmm_prob: Optional[float],
    ) -> float:
        """Weighted combination of CNN and GMM probabilities."""
        if cnn_prob is not None and gmm_prob is not None:
            return (self._cnn_weight * cnn_prob) + (self._gmm_weight * gmm_prob)
        elif cnn_prob is not None:
            return cnn_prob
        elif gmm_prob is not None:
            return gmm_prob
        else:
            return 0.0  # no data — treat as human

    def _compute_average(self) -> float:
        """Mean of all scores in the buffer."""
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)

    def _classify(self, score: float) -> str:
        """Map a score to a classification label."""
        if score < self._human_threshold:
            return self.LABEL_HUMAN
        elif score < self._ai_threshold:
            return self.LABEL_SUSPICIOUS
        else:
            return self.LABEL_AI
