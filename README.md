# BankAIAgent – Intelligent Bank Statement Parser (Colab + AI)

This project is a **machine learning-powered tool** for extracting structured data from bank statement PDFs using OCR, NLP, and automation pipelines.

## Features
- Parse scanned or digital bank statements (e.g. HDFC, ICICI, SBI, YES Bank).
- Extract structured fields: `Date`, `Narration`, `Debit`, `Credit`, `Balance`, `Cheque Number`.
- Handle multi-line entries and noisy data using AI + rule-based logic.
- Export results to Excel or Google Sheets.
- Built and trained using **Google Colab**, **Python**, **OCR (Tesseract)**, and **spaCy**.

## Notebooks
| File | Description |
|------|-------------|
| [`ML_B2E.ipynb`](notebooks/ML_B2E.ipynb) | Core AI pipeline for parsing and structuring bank statement data |

## Technologies Used
- Python, Pandas, Regex
- Tesseract OCR, PaddleOCR
- spaCy NLP, scikit-learn
- Google Colab, GitHub

## Future Enhancements
- UI for uploading PDFs and downloading Excel files
- Smart categorization of transactions
- Integration with financial dashboards

## Getting Started
1. Open [`ML_B2E.ipynb`](https://colab.research.google.com/github/bhumihere234/BankAIAgent/blob/main/notebooks/ML_B2E.ipynb) in Google Colab
2. Upload sample bank statement
3. Run all cells and export to CSV/Excel

## Contact
For questions or collaboration:
- Created by **[Your Name]**
- Email: [your.email@example.com]
