# Vision-Based Precision Pose Estimation for Autonomous Formation Flying

UE24CS352A Machine Learning mini-project · PES University · Section J · Group 16 · Problem statement **#9**

| Team member | SRN |
|---|---|
| Neha K S | PES1UG24CS581 |
| Mohith N | PES1UG24CS580 |

Re-implementation of R. Punnoose, *"Vision-Based Precision Pose Estimation For Autonomous Formation Flying"*
(Stanford CS229, Spring 2020) with one extension from the paper's own future-work section.

## Problem
A *follower* aircraft carries a camera that always looks at a *leader* aircraft. From the pixel locations of
14 known structural features of the leader (some of which are self-occluded), estimate the relative pose
`[x, y, z, roll]` accurately enough for tight formation flight.

## Pipeline
```
 simulated detector ─► 42-d feature vector ─► ANN (42-100-100-4800) ─► coarse pose label ─┐
   (14 x [u, v, occluded])                                                                │
                                                                                          ▼
              particle filter (dynamics + reprojection likelihood, resampled from ANN label) ─► pose
              Gauss-Newton / LM reprojection refinement (ours, from ANN top-3 labels)        ─► pose
```

## Layout
```
poseest/        config, aircraft+camera model, discretisation, dataset, ANN, dynamics, particle_filter, refine, pipeline
scripts/        train_ann.py  run_experiment.py  noise_sweep.py  demo.py
tests/          test_data.py  test_filters.py
artifacts/      trained ANN weights, validation error pool, ANN metrics
results/        figures, json results, training log, demo gif
docs/           write-up (PDF) and presentation slides
```

## Reference
R. Punnoose, "Vision-Based Precision Pose Estimation For Autonomous Formation Flying", CS229 Final Report, Stanford, 2020.
