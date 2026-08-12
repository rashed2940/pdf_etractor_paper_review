"""Generate a richer synthetic PDF for Phase 1 acceptance test.

Produces a ~4-page cybersecurity paper with multiple sections so the
chunker exercises section-aware splitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

OUT = ROOT / "data" / "_test_pdfs" / "sample_paper.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

c = canvas.Canvas(str(OUT), pagesize=letter)
W, H = letter
PAGE_TEXT_MARGIN = 72

# ---- Page 1: title + abstract + intro ----
y = H - PAGE_TEXT_MARGIN
lines_p1 = [
    "Deep Learning for Intrusion Detection in IoT Networks",
    "Alice Smith, Bob Jones, Carol Lee",
    "Proceedings of the 2024 IEEE Symposium on Security and Privacy",
    "DOI: 10.1109/SP.2024.000123",
    "",
    "Abstract",
    "We propose a novel deep learning approach for detecting network intrusions in IoT",
    "environments. Our method combines convolutional and recurrent layers to capture",
    "both spatial and temporal patterns in network traffic. We evaluate on CICIDS2017",
    "and a custom IoT dataset, achieving 99.2% accuracy and outperforming baselines.",
    "",
    "1. Introduction",
    "Intrusion detection is critical for cybersecurity in modern IoT networks.",
    "Existing approaches struggle with the heterogeneity of IoT devices and the",
    "evolving nature of attacks. We motivate our work by reviewing prior art and",
    "identifying gaps in handling encrypted traffic.",
    "",
    "2. Related Work",
    "Prior work falls into three categories: signature-based, anomaly-based, and",
    "machine learning approaches. Recent advances leverage deep neural networks.",
]
for ln in lines_p1:
    c.drawString(PAGE_TEXT_MARGIN, y, ln)
    y -= 16
c.showPage()

# ---- Page 2: Methodology + Architecture ----
y = H - PAGE_TEXT_MARGIN
lines_p2 = [
    "2",
    "Proceedings of the 2024 IEEE Symposium on Security and Privacy",
    "",
    "3. Methodology",
    "We designed a hybrid CNN-LSTM architecture implemented in PyTorch. The CNN",
    "extracts spatial features from packet headers; the LSTM models temporal",
    "dependencies across sequences of packets.",
    "",
    "4. Architecture",
    "Our model has three convolutional blocks with 64, 128, and 256 filters,",
    "followed by a two-layer LSTM with 128 hidden units. We use dropout 0.3 and",
    "the Adam optimizer with learning rate 1e-3.",
    "",
    "5. Experiments",
    "We trained on the CICIDS2017 dataset (80/20 split) and tested on an IoT-specific",
    "dataset collected from a smart-home testbed. All experiments used an NVIDIA A100 GPU.",
]
for ln in lines_p2:
    c.drawString(PAGE_TEXT_MARGIN, y, ln)
    y -= 16
c.showPage()

# ---- Page 3: Results + Discussion ----
y = H - PAGE_TEXT_MARGIN
lines_p3 = [
    "3",
    "Proceedings of the 2024 IEEE Symposium on Security and Privacy",
    "",
    "6. Results",
    "Our model achieves 99.2% accuracy on CICIDS2017 and 96.8% on the IoT testbed.",
    "We outperform Random Forest and XGBoost baselines by 3.7 and 2.1 percentage",
    "points respectively. False positive rate is 0.4%.",
    "",
    "7. Discussion",
    "The improvement is most pronounced on encrypted traffic where header-only features",
    "are available. Training time is 4.2 hours; inference latency is 12ms per flow.",
]
for ln in lines_p3:
    c.drawString(PAGE_TEXT_MARGIN, y, ln)
    y -= 16
c.showPage()

# ---- Page 4: Limitations + Conclusion + References ----
y = H - PAGE_TEXT_MARGIN
lines_p4 = [
    "4",
    "Proceedings of the 2024 IEEE Symposium on Security and Privacy",
    "",
    "8. Limitations",
    "Our approach requires labeled training data and may not generalize to zero-day",
    "attacks. Adversarial robustness was not evaluated.",
    "",
    "9. Conclusion",
    "We presented an effective deep learning approach for IoT intrusion detection.",
    "Future work includes self-supervised pretraining and adversarial robustness.",
    "",
    "References",
    "[1] A. Smith, 'Network intrusion detection: a survey,' IEEE Trans. 2020.",
    "[2] B. Jones, 'Deep learning for cybersecurity,' ACM CCS 2022.",
]
for ln in lines_p4:
    c.drawString(PAGE_TEXT_MARGIN, y, ln)
    y -= 16
c.showPage()

c.save()
print(f"Generated: {OUT} ({OUT.stat().st_size} bytes)")
