# Prompt Injection Detector 🛡️

A machine learning-based system to detect and prevent prompt injection attacks in AI language models.

## Overview 📝

This project implements a binary classifier that can identify potentially malicious prompts, including:

- Jailbreak attempts
- Prompt injections
- Malicious instructions

The system uses DistilBERT for text classification and achieves high accuracy in detecting harmful prompts while maintaining low false positive rates.

## Project Structure 📂

```
final_project/
├── dataset.csv          # Training data with labeled prompts
├── index.py            # Main training script
├── test.py            # Testing and evaluation script
└── prompt_injection_detector/ # Saved model directory
```

## Features ✨

- Binary classification (Safe/Malicious)
- Text embedding generation
- Real-time prompt safety checking
- Visualization of prompt embeddings
- Detailed classification metrics

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/Kiraneswar/Prompt-Injection-Detector.git
cd Prompt-Injection-Detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 💻

### Training the Model

```python
python index.py
```

This will:
- Load the dataset
- Train the DistilBERT classifier
- Save the model checkpoints
- Generate evaluation metrics

### Testing Prompts

```python
python test.py
```

Use this to:
- Test new prompts
- Get safety scores
- View detailed analysis

## Model Performance 📊

The current model achieves:
- Accuracy: ~99%
- Precision: 93% (Malicious class)
- Recall: 96% (Malicious class)
- F1-Score: 95% (Malicious class)

## Data Format 📋

The training data (`dataset.csv`) contains:
- `category`: Prompt category (conversation, jailbreak, etc.)
- `base_class`: Base classification
- `text`: The actual prompt text
- `is_malicious`: Binary label (0 = safe, 1 = malicious)

## Visualization 📈

The project includes visualization tools to:
- View prompt embeddings in 2D using t-SNE
- Analyze cluster formations
- Inspect decision boundaries

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Thanks to the Hugging Face team for their transformers library
- Special thanks to the open-source ML community

## Contact 📧

- Kiraneswar
- GitHub: [@Kiraneswar](https://github.com/Kiraneswar)

---

⚠️ **Note**: This tool is meant for research purposes and should be used responsibly. Always respect ethical AI guidelines and privacy concerns.