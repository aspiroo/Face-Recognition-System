# Face-Recognition-System
Project Prompt: 
Design a face recognition system. Assemble a dataset comprising facial images of 10 individuals exhibiting various facial expressions. Demonstrate the performance of your method using this dataset.


## Project Structure

```
Face-Recognition-System/
│
├── data/                # Dataset (10 persons, each in separate folder)
├── support/             # Training and recognition scripts
├── others/              # Reports, presentations, demo video
├── requirements.txt     # Python dependencies
├── .gitignore          # Ignored files
├── README.md           # Project documentation
└── venv/               # Virtual environment (not pushed to GitHub)
```

---

## Setting Up

### Prerequisites

* Python 3.11+
* Git

---

### 1. Clone the repository

```bash
git clone https://github.com/aspiroo/Face-Recognition-System.git
cd Face-Recognition-System
```

---

### 2. Create and activate virtual environment

#### Windows

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

```bash
venv\Scripts\activate
```

#### macOS/Linux

Create virtual environment

```bash
python3 -m venv venv
```

Activate virtual environment

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Inside the `data/` folder, create subfolders for each person:

```
data/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── person2/
│   ├── img1.jpg
│   └── img2.jpg
...
```

Requirements:

* 10 individuals
* 5–15 images per person
* Different facial expressions recommended

---

## Running the Project

### Train the model

```bash
python support/train_model.py
```

This will:

* Load dataset
* Train recognition model
* Save trained model

---

### Run recognition

```bash
python support/recognize.py
```

This will:

* Load trained model
* Predict person identity from image

---

## Demonstrating Performance

The system will output:

* Accuracy score
* Predictions
* Model performance metrics

---

## Dependencies

Main libraries used:

* numpy
* pandas
* matplotlib
* scikit-learn
* opencv-python

Install using:

```bash
pip install -r requirements.txt
```

---

## Author

CSE445 Face Recognition System Project

---

<p align="right">(<a href="#top">back to top</a>)</p>

