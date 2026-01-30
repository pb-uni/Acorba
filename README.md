# ACORBA (macOS Port) 

### based on ACORBA v1.3: February 2021

**ACORBA** (Automatic Calculation Of Root Bending Angles) is an automated software tool designed to measure root bending angle dynamics over time. This repository is a **macOS-compatible port** of the original Windows-only software.

## 🍎 macOS Port Information
This version has been adapted to run on macOS (Apple Silicon).
*   **Main Entry Point:** Run `python gui.py` to start the application.
*   **Key Changes:** 
    *   Updated codebase to remove Windows-specific pathing and dependencies.
    *   Replaced Windows `.exe` installers with a standard Python environment setup.
    *   Adjusted libraries for compatibility with macOS architectures (e.g., specific TensorFlow/OpenCV versions).

---

## 🔬 About ACORBA
ACORBA provides an unbiased, highly reproducible, and automated workflow for plant biologists studying root gravitropism and waving. It eliminates the human bias often found in manual measurements (e.g., using ImageJ).

### Key Features
*   **Fully Automated Workflow:** Measures primary root bending angles from microscope or flatbed scanner images.
*   **Dual-Step Analysis:** 
    1.  **Segmentation:** Identifies the root surface using traditional image processing or Deep Machine Learning (UNET architecture).
    2.  **Angle Calculation:** Generates Euclidean vectors to determine root tip orientation in a 360° space (exported in a +180°/-180° format).
*   **Versatile Inputs:** Supports vertical-stage microscopy, flatbed scanners, and semi-automated analysis for mobile phone or stereomicroscope images.

### Original Technical Stack
ACORBA was originally developed using:
*   **Language:** Python
*   **ML Architecture:** Modified UNET for semantic segmentation.
*   **Primary Frameworks:** TensorFlow, Keras, OpenCV, Scikit-Image, FreeSimpleGUI (migrated to a free drop-in alternative due to paywall in version 5).

---

## 📝 Attribution & Licensing
This software is a derivative work based on the original **ACORBA** project. 

*   **Original Project:** [ACORBA on SourceForge](https://sourceforge.net/projects/acorba/)
*   **Original Authors:** Nelson B. C. Serre and Matyáš Fendrych (Department of Experimental Plant Biology, Charles University).
*   **Original Source:** [sf.net/p/acorba/code](https://git.code.sf.net)
*   **Reference Paper:** [Serre & Fendrych (2022), *Quantitative Plant Biology*](https://doi.org/10.1017/qpb.2022.4)
*   **License:** This project is licensed under the [Creative Commons Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0)](https://creativecommons.org) license.

> **Non-Commercial Use Only:** Per the terms of the CC BY-NC 2.0 license, this software and its derivatives may **not** be used for commercial purposes.

---

## 🚀 Getting Started on macOS
### Prerequisites
*   Python 3.10 (Recommended for TensorFlow compatibility)
*   [Homebrew](https://brew.sh) (Optional, for managing Python versions)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/pb-uni/Acorba.git
    cd Acorba
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Software
Launch the Graphical User Interface by running:
```bash
python gui.py