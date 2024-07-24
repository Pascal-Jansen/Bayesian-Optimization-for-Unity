# Bayesian Optimization for Unity

by [Pascal Jansen](https://scholar.google.de/citations?user=cR1_0-EAAAAJ&hl=en)

### About

Unity asset that enables eased access to a Bayesian Optimization process (based on botorch.py) within any project.
It can optimize design parameter values to maximize (or minimize) design objective values. 

The asset uses a **Human-in-the-Loop** approach that iteratively queries user feedback (i.e., objective values) to designs (i.e., parameter values).

The hyperparameters for the optimization can be freely set in Unity.
The communication between Unity and the Python process running the botorch-based implementation is handled automatically.

### Usage

- The *explicit* Human-in-the-Loop process currently requires the [QuestionnaireToolkit](https://assetstore.unity.com/packages/tools/gui/questionnairetoolkit-157330) to receive explicit subjective feedback by users (e.g., measuring the usability of a design using the System Usability Scale).
This feedback is used as design objective value in the optimization process.

- There is currently no *implicit* variant of this asset that can, for example, process the user's physiology as input for the design target values.
