# Hybrid Text Summarization for Telugu Language 📝🇮🇳

This project implements a **hybrid text summarization** approach for the **Telugu language**, combining both **extractive** and **abstractive** methods. It uses **TextRank** for extractive summarization and a fine-tuned **mT5-small transformer model** for abstractive summarization.

---

## 📂 Repository Contents

- `EXTRACTIVE_SUMMARIZATION.ipynb`: Implementation using **TextRank** algorithm.
- `ABSTRACTIVE_TRANSFORMER.ipynb`: Fine-tuning **mT5-small** for abstractive summarization.
- `SEQ2SEQ_pytorch.ipynb` and `seq2seq_tensorflow.ipynb`: Experimental comparison with other seq2seq models.
- `model_compare.ipynb`: ROUGE score evaluation and model performance.
- `mini_t5model/`: Contains saved fine-tuned mT5 model files.
- `telugu_XLSum_v2.0.tar.bz2`: Dataset used for training and evaluation.
- `A12_report.pdf / A12_report.zip`: Final project documentation.
- `LICENSE`: MIT License.

---

## 🚀 Getting Started

### 🔁 Clone the Repository

```bash
git clone https://github.com/Shxam/Hybrid-Text-Summarization-Telugu-Language.git
cd Hybrid-Text-Summarization-Telugu-Language
📦 Setup Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
🧠 Methodology
📌 Extractive Summarization
Algorithm: TextRank

Input: Raw Telugu document

Output: Ranked key sentences

📌 Abstractive Summarization
Model: google/mt5-small

Fine-tuned on extractive summaries and human-written targets

Language: Telugu

Training: 5 epochs, batch size 10, max input length 512

Tokenizer: mt5 tokenizer (supports multilingual text)

)

📊 ROUGE Evaluation
Metric	Score
ROUGE-1	0.4321
ROUGE-2	0.3625
ROUGE-L	0.4114

💡 Fine-tuning with extractive summaries as input improved ROUGE scores by ~22% over baseline mT5 generation.

.

🧪 Sample Inference (Abstractive)
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("./mini_t5model")

text = "హైదరాబాద్ సెంట్రల్ యూనివర్శిటీ (HCU) భూముల వివాదం రాష్ట్రవ్యాప్తంగా తీవ్ర చర్చనీయాంశంగా మారింది..."
input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

output = model.generate(input_ids, max_length=90, num_beams=4, early_stopping=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
Hyderabad యూనివర్శిటీ పరిధిలోని సుమారు 400 ఎకరాల భూమిని ప్రభుత్వ అభివృద్ధి కార్యక్రమాల కోసం టెక్నాలజీ పార్క్ నిర్మాణానికి కేటాయించనున్నట్టు సమాచారం వెలుగులోకి రాగానే, విద్యార్థులు, పర్యావరణ ప్రేమికులు, స్థానికులు తీవ్ర వ్యతిరేకత వ్యక్తం చేస్తున్నారు.

📂 Project Structure
pgsql
Copy
Edit
Hybrid-Text-Summarization-Telugu-Language/
│
├── mini_t5model/                  # Fine-tuned mT5 model checkpoint
├── A12_report.pdf / .zip          # Final report
├── telugu_XLSum_v2.0.tar.bz2      # Dataset
├── EXTRACTIVE_SUMMARIZATION.ipynb # TextRank implementation
├── ABSTRACTIVE_TRANSFORMER.ipynb  # mT5 fine-tuning
├── SEQ2SEQ_pytorch.ipynb          # Experimental PyTorch model
├── seq2seq_tensorflow.ipynb       # TensorFlow-based baseline
├── model_compare.ipynb            # Model evaluation and plots
├── LICENSE
└── README.md



🙌 Acknowledgements
Hugging Face Transformers

Indic NLP Toolkit

TextRank Algorithm

XL-Sum Telugu Dataset
