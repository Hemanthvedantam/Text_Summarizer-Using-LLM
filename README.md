<h1 align="center">📝 Text Summarizer using LLM</h1> <p align="center"> Summarize large text documents using powerful LLMs, visualize word clouds, evaluate readability, and download clean summaries — all in a Flask-powered web app. </p>

**✨ Features**

**🤖 LLM-Powered Summarization** (Facebook_mistral)

📄 Upload plain text or paste content directly

⬇️ Download the clean **summarized output** as a .txt file

☁️ Auto-generate a **Word Cloud** from original or summarized text

📚 **Text Quality Metrics:**

Flesch Reading Ease

Flesch-Kincaid Grade Level

Lexical Density

⚡ Lightweight, fast, and minimal UI

**🛠 Tech Stack**

**👨‍💻 Backend**

**Python 3.8+**

**Flask** – Web server

**transformers** – LLM-based summarization

**textstat** – Readability scoring

**wordcloud** – Word cloud generation

**NLTK / SpaCy** – Tokenization & NLP tools

**🌐 Frontend**

**HTML/CSS** – Basic UI (Jinja templating)

**⚙️ Getting Started**

**🔧 Installation**

git clone https://github.com/Hemanthvedantam/Text_Summarizer-Using-LLM.git
cd Text_Summarizer-Using-LLM
pip install -r requirements.txt

**▶️ Run the App**

python app.py

**📁 Output**

**Text Summary (.txt)** – Clean downloadable summary

**Word Cloud (.png)** – Saved automatically

**Metrics** – Flesch Readability, Grade Level, Lexical Density

