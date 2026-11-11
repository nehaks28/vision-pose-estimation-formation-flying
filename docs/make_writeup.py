import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = ParagraphStyle("b", fontName="Helvetica", fontSize=8.6, leading=10.9, spaceAfter=3, alignment=4)
H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10.2, leading=12, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1b4b6b"))
T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14, leading=17, spaceAfter=1)
S = ParagraphStyle("s", fontName="Helvetica", fontSize=8.6, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=3)
C = ParagraphStyle("c", fontName="Helvetica", fontSize=7.6, leading=9.2, textColor=colors.HexColor("#444444"), alignment=1)
doc = SimpleDocTemplate(f"{ROOT}/docs/Writeup_Problem9_Group16.pdf", pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=13*mm, bottomMargin=12*mm,
                        title="Vision-Based Precision Pose Estimation for Autonomous Formation Flying", author="Neha K S, Mohith N")
s = []
s.append(Paragraph("Vision-Based Precision Pose Estimation for Autonomous Formation Flying", T))
s.append(Paragraph("UE24CS352A Machine Learning Mini-Project &nbsp;|&nbsp; Section J, Group 16, Problem #9 &nbsp;|&nbsp; Neha K S (PES1UG24CS581), Mohith N (PES1UG24CS580)", S))

s.append(Paragraph("1. Problem statement", H))
s.append(Paragraph("A follower aircraft F carries a camera that is gimballed to always point at a leader aircraft L. From a single image we must "
  "estimate the relative pose of the leader (position and roll about the line of sight) precisely enough for <i>tight</i> formation flight. "
  "Our assigned problem is the reproduction of Punnoose (Stanford CS229, 2020): an ANN classifies detected aircraft features into a coarse "
  "discrete pose label, and a particle filter (PF) informed by that label refines it. GPS/inertial sensors are not precise enough at these "
  "separations, and a pure end-to-end regressor gives no uncertainty and ignores dynamics, which motivates the hybrid design.", B))

s.append(Paragraph("2. Dataset", H))
s.append(Paragraph("The paper releases no data, so we generate it synthetically. A 14-feature aircraft model (nose, canopy, wing/stabiliser tips, nozzles, intakes, ...) "
  "is projected through a pinhole camera (1280x960 px, f = 1600 px). Each feature has an outward normal; features facing away from the camera are "
  "<i>occluded</i> (a normal-based stand-in for ray tracing). The detector adds 0.5 px Gaussian noise and misses 2 % of visible features. "
  "A sample is the 42-vector [u, v, occluded] x 14, and its label is one of <b>4800</b> pose cells (x: 10 bins over [-100, 50] m, y: 10 over [-50, 50] m, "
  "z: 8 over [-20, 20] m, roll: 6 over [-45, 45] deg, i.e. Table I of the paper). We draw 300 000 training, 30 000 validation and 30 000 test poses "
  "uniformly in that box (range &ge; 15 m). Evaluation uses 10 random 10 s approach trajectories (10 Hz) that the follower flies from ~90 m to ~30 m behind the leader "
  "while the leader manoeuvres.", B))

s.append(Paragraph("3. Approach", H))
s.append(Paragraph("<b>(a) ANN pose-label classifier</b>: 42 &rarr; 100 &rarr; 100 &rarr; 4800, ReLU, softmax, Adam (as in the paper). "
  "<b>(b) Particle filter</b>: particles are initialised from the ANN label plus a (true &minus; label) error pool measured on validation data. Each step, particles "
  "are propagated through the relative dynamics, where the leader's unknown speed, pitch and roll appear as a uniform disturbance; each particle's 14 features are "
  "projected and compared with the detected ones (occluded ones ignored); particles are weighted, and the resampling draws floor(&alpha;N) from the weighted set and the rest from "
  "the ANN label + error pool. We implement the paper's inverse-MSE likelihood (two configurations: N=1000, &alpha;=0.9 every 10 cycles; N=5000, &alpha;=0.9 every cycle) "
  "and a Gaussian likelihood with roughening (N=2000, &alpha;=0.5). "
  "<b>(c) Our extension</b> (the paper's own future-work suggestion): a Levenberg-Marquardt minimisation of the reprojection error over the 4-DoF pose, started from the ANN's top-3 classes "
  "and keeping the lowest-cost solution (single-view bundle adjustment).", B))

s.append(Paragraph("4. Implementation overview", H))
s.append(Paragraph("Python + numpy/scipy/matplotlib only (no DL framework: forward/backward passes of the ANN are written by hand). Package <font face='Courier'>poseest/</font>: "
  "<font face='Courier'>aircraft.py</font> (model, projection, occlusion, detector), <font face='Courier'>discretize.py</font>, <font face='Courier'>dataset.py</font>, <font face='Courier'>ann.py</font>, "
  "<font face='Courier'>dynamics.py</font>, <font face='Courier'>particle_filter.py</font>, <font face='Courier'>refine.py</font>, <font face='Courier'>pipeline.py</font>. "
  "Scripts: <font face='Courier'>train_ann.py</font> (~9 min on one CPU core), <font face='Courier'>run_experiment.py</font>, <font face='Courier'>noise_sweep.py</font>, <font face='Courier'>demo.py</font> (live demo + GIF). "
  "Unit tests cover label round-trips, projection scaling, occlusion, the refinement and trajectory validity.", B))

s.append(Paragraph("5. Results", H))
rows = [["Method", "Position error (m)", "Roll error (deg)"],
        ["ANN only (cell centre)", "5.32 \u00b1 0.31", "4.51 \u00b1 0.58"],
        ["ANN + PF, paper cfg 1 (N=1000)", "0.96 \u00b1 0.25", "1.09 \u00b1 0.20"],
        ["ANN + PF, paper cfg 2 (N=5000)", "0.70 \u00b1 0.09", "0.80 \u00b1 0.14"],
        ["ANN + PF, Gaussian likelihood (N=2000)", "0.62 \u00b1 0.09", "0.96 \u00b1 0.09"],
        ["ANN + Gauss-Newton refinement (ours)", "0.25 \u00b1 0.04", "0.18 \u00b1 0.03"]]
t = Table(rows, colWidths=[78*mm, 40*mm, 40*mm])
t.setStyle(TableStyle([("FONT", (0,0), (-1,-1), "Helvetica", 8.2), ("FONT", (0,0), (-1,0), "Helvetica-Bold", 8.2),
  ("FONT", (0,-1), (-1,-1), "Helvetica-Bold", 8.2), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dfeaf2")),
  ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#e3f3ea")), ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#9fb3c2")),
  ("ALIGN", (1,0), (-1,-1), "CENTER"), ("TOPPADDING", (0,0), (-1,-1), 1.8), ("BOTTOMPADDING", (0,0), (-1,-1), 1.8)]))
s.append(t)
s.append(Paragraph("Mean &plusmn; std over 10 trajectories. ANN test accuracy: top-1 69.6 % (paper: 54 %), top-5 96.0 %, 98.3 % within one neighbouring cell; "
  "mean cell-centre error 5.6 m / 4.3&deg;, so almost all error is discretisation, as the paper observed.", C))
s.append(Spacer(1, 3))
imgs = Table([[Image(f"{ROOT}/results/error_vs_time.png", width=100*mm, height=100*mm*3.6/10), Image(f"{ROOT}/results/noise_sweep.png", width=64*mm, height=64*mm*3.3/4.6)]],
             colWidths=[104*mm, 70*mm])
imgs.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 0)]))
s.append(imgs)
s.append(Paragraph("Left: mean error over time (log scale). Right: robustness to detector pixel noise (ANN not retrained).", C))

s.append(Paragraph("6. Conclusions", H))
s.append(Paragraph("The ANN gives a reliable <i>coarse</i> label (70 % exact, 98 % within one cell) but its precision is capped by the 15 m x 10 m x 5 m x 15&deg; cell size. "
  "Unlike the paper, where the PF changed little, our ANN-informed PF cuts the error about 7x (5.3 m &rarr; 0.6-0.7 m); we attribute this to our cleaner simulator "
  "(exact camera model, Gaussian noise, no outliers, 4-DoF state), so absolute numbers are <b>not</b> comparable to the paper. "
  "The reprojection-based refinement is clearly best at low noise (0.25 m, 0.18&deg;), confirming the paper's suggestion that the ANN is best used as a robust initialiser. "
  "It is per-frame, though: beyond ~2 px of detector noise the temporal Gaussian PF overtakes it, so the natural next step is to fuse both "
  "(refinement as the PF measurement). Limitations: synthetic data only, simplified occlusion and dynamics, and the CNN feature detector is assumed rather than built.", B))
doc.build(s)
