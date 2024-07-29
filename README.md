# Bayesian Optimization for Unity

by [Pascal Jansen](https://scholar.google.de/citations?user=cR1_0-EAAAAJ&hl=en)

### About

This Unity asset integrates Bayesian Optimization (based on [botorch.org](https://botorch.org/)) into your projects, enabling the optimization of design parameters to maximize or minimize objective values. It utilizes a **Human-in-the-Loop** approach, iteratively querying user feedback on designs to refine parameter values.

Key features include:
- Set optimization hyperparameters directly in Unity.
- Automatic communication between Unity and the Python process running the botorch-based implementation.

### Usage

#### Human-in-the-Loop Process

To utilize the Human-in-the-Loop optimization, this asset requires the [QuestionnaireToolkit](https://assetstore.unity.com/packages/tools/gui/questionnairetoolkit-157330) for collecting explicit subjective feedback from users. This feedback serves as the design objective value in the optimization process. 

Example use case:
- Measure the usability of a design using the System Usability Scale (SUS) and use this data for optimization.

#### Limitations

Currently, there is no *implicit* feedback variant of this asset. It cannot, for example, process physiological data from users as input for design objective values.


### Installation


### Prerequisites
This package requires Python. We recommend using the latest 3.11 version (3.12 should also work).

#### Install using pip
After downloading, install requirements (could also work with older versions but kept up-to-date for performance reasons):
```
pip install -r requirements.txt
```
