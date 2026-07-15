# 📚 Amazon KDP Metadata & Title Optimizer

An AI-powered publishing assistant designed to maximize your book's organic search visibility on the Amazon Kindle Store. Powered by **Groq's ultra-fast LLaMA 3.3 (70B)** model and built with an interactive **Streamlit** frontend, this tool generates highly targeted backend keywords, exact browse category paths, and search-optimized title/subtitle variations.

---

## 🚀 Features

* **7 Long-Tail Backend Keywords:** Generates exactly 7 natural search phrases (under 50 characters each) mapped to real reader intent. It strictly avoids repeating words already in your title or subtitle to maximize Amazon's indexing real estate.
* **Exact Amazon Category Mapping:** Provides 3 highly specific, standard Amazon Kindle Store browse paths so you can place your book in less competitive niches directly during KDP setup.
* **SEO Title & Subtitle Generator:** Recommends 3 click-optimized variations of your title and subtitle to hook readers while naturally containing critical search terms.
* **Sub-Second AI Inference:** Leverages Groq's high-speed API endpoints to provide optimization insights in the blink of an eye.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM Orchestration:** Groq Python SDK (utilizing `llama-3.3-70b-versatile` in JSON Object Mode)
* **Environment Management:** Python-dotenv
* **Language:** Python 3.9+

---

## ⚙️ Installation & Setup

Follow these steps to run the application locally:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/kdp-metadata-optimizer.git](https://github.com/your-username/kdp-metadata-optimizer.git)
cd kdp-metadata-optimizer
