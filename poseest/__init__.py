"""Vision-based precision pose estimation for autonomous formation flying.

Re-implementation of the pipeline described in R. Punnoose, "Vision-Based Precision
Pose Estimation For Autonomous Formation Flying" (Stanford CS229, Spring 2020):
  detected aircraft features -> ANN pose-label classifier -> particle filter.
We add one extension suggested in the paper's future work: a Gauss-Newton
(bundle-adjustment style) refinement of the coarse pose.
"""
