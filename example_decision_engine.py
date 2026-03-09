"""
Example: DecisionEngine Usage
==============================
Simulates 12 audio frames transitioning from human speech
to AI-generated voice, demonstrating frame smoothing,
classification, and notification behaviour.
"""

from modules.decision_engine import DecisionEngine


def main():
    engine = DecisionEngine(buffer_size=5, cnn_weight=0.7, gmm_weight=0.3)

    # Simulated frames: (cnn_probability, gmm_probability)
    # Gradually transitions from clearly human → AI generated
    frames = [
        (0.10, 0.15),   # Frame 1  — clearly human
        (0.20, 0.25),   # Frame 2  — still human
        (0.35, 0.40),   # Frame 3  — borderline human
        (0.50, 0.55),   # Frame 4  — suspicious
        (0.60, 0.65),   # Frame 5  — suspicious (buffer fills here)
        (0.72, 0.68),   # Frame 6  — entering AI territory
        (0.85, 0.80),   # Frame 7  — strong AI signal
        (0.90, 0.88),   # Frame 8  — very strong AI
        (0.92, 0.91),   # Frame 9  — sustained AI
        (0.88, 0.85),   # Frame 10 — slight dip, still AI
        (0.40, 0.35),   # Frame 11 — drops to suspicious
        (0.15, 0.10),   # Frame 12 — back to human
    ]

    print("=" * 70)
    print("DecisionEngine Demo — Frame-by-Frame Smoothing")
    print("=" * 70)
    print(f"Buffer size: {engine.buffer_size} | "
          f"Weights: CNN={0.7}, GMM={0.3}")
    print(f"Thresholds: Human < 0.40 | Suspicious 0.40–0.70 | AI ≥ 0.70")
    print("-" * 70)

    for i, (cnn, gmm) in enumerate(frames, 1):
        result = engine.update(cnn_probability=cnn, gmm_probability=gmm)

        notify_flag = " 🔔 NOTIFY!" if result.should_notify else ""
        buffer_indicator = "●" if result.buffer_full else "○"

        print(
            f"Frame {i:2d} | "
            f"CNN={cnn:.2f} GMM={gmm:.2f} | "
            f"Ensemble={result.ensemble_score:.3f} | "
            f"Avg={result.average_score:.3f} | "
            f"{buffer_indicator} [{result.frames_buffered}/{engine.buffer_size}] | "
            f"{result.label:<12s}{notify_flag}"
        )

    print("-" * 70)
    print(f"Final state: {engine.label} ({engine.average_score:.1%})")
    print()

    # Demonstrate reset
    engine.reset()
    print("After reset:")
    print(f"  Buffer: {engine.frames_buffered}/{engine.buffer_size}")
    print(f"  Label:  {engine.label}")
    print(f"  Score:  {engine.average_score:.3f}")


if __name__ == "__main__":
    main()
