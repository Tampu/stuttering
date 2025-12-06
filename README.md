# Stuttering Detection and Analysis

This repository provides notebooks and code for the detection and analysis of stuttering using machine learning and audio processing techniques.

## Table of Contents
- [Overview](#overview)
- [Contents](#contents)
- [Installation](#installation)
- [Usage](#usage)
- [Approaches](#approaches)
- [Contributing](#contributing)
- [License](#license)

## Overview

Stuttering is a speech disorder involving frequent disruptions in the flow of speech. This project applies advanced machine learning models—for both audio feature extraction and classification—to facilitate automatic stuttering detection. The repository contains Jupyter notebooks and utility functions for preparing data, extracting features, and building stuttering classifiers leveraging state-of-the-art libraries such as PyTorch, LibROSA, and HuggingFace Transformers.

## Contents

- [`stuttering_notebook_approach1.ipynb`](https://github.com/Tampu/stuttering/blob/main/stuttering_notebook_approach1.ipynb): First approach to data processing and stuttering classification.
- [`stuttering_notebook_approach2.ipynb`](https://github.com/Tampu/stuttering/blob/main/stuttering_notebook_approach2.ipynb): Second methodology exploring advanced feature engineering and model architectures.
- `feature_to_json.py`: Utility functions for summarizing and exporting audio features to JSON format.

## Installation

Before running the notebooks, ensure you have Python 3.9+ and the following dependencies installed:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.41.2 transformers_stream_generator einops accelerate
pip install matplotlib tiktoken pandas soundfile librosa scikit-learn
```

Some environments may require specific CUDA drivers for GPU acceleration (see notebook for details).

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Tampu/stuttering.git
   cd stuttering
   ```
2. Install the dependencies as shown above.
3. Open either notebook (`stuttering_notebook_approach1.ipynb` or `stuttering_notebook_approach2.ipynb`) in Jupyter Lab or Notebook.
4. Follow the instructions within the notebook to process audio data and run the stuttering detection pipeline.

Both notebooks include sections for:
- Audio feature extraction (using LibROSA, SoundFile)
- Machine learning modeling (using PyTorch, Transformers)
- Visualization (Matplotlib)
- Model evaluation

## Approaches

### Approach 1

- Provides baseline audio preprocessing and traditional machine learning classification.

### Approach 2

- Explores additional feature engineering, a wider set of deep learning models, HuggingFace Transformers, and advanced visualization.
- Utilities for exporting feature summaries to JSON for further analysis.

## Contributing

Contributions are welcome! Please submit issues and pull requests to help improve data processing, modeling, or pipeline robustness.

## License

This repository is provided under the MIT License. See the [LICENSE](LICENSE) file for more details.


