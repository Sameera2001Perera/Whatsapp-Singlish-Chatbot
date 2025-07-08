# WhatsApp Singlish Chatbot

A sophisticated WhatsApp chatbot that transliterates Singlish (English-Sinhala mixed language) to proper Sinhala text using advanced NLP techniques and AI models. The bot also provides document reading capabilities and intelligent conversational AI features.

## Features

- **Singlish to Sinhala Transliteration**: Converts Singlish text to proper Sinhala using a combination of:
  - Dictionary-based lookup
  - Rule-based transliteration
  - BERT-based masked language modeling
  - Contextual word prediction
- **Document Processing**: Upload and process PDF documents with vector search capabilities
- **Conversational AI**: Powered by Anthropic's Claude model for intelligent responses
- **WhatsApp Integration**: Seamless integration with WhatsApp Business API
- **Multi-Modal Support**: Handles text messages and document uploads
- **Secure Webhook**: Signature verification for WhatsApp API security

## System Architecture

The application consists of several key components:

### Core Components

- **Flask Web Server**: Handles WhatsApp webhook requests
- **Transliterator Engine**: Multi-stage transliteration pipeline
- **Document Reader**: PDF processing and vector search
- **WhatsApp Utils**: Message processing and API integration
- **Security Layer**: Webhook signature verification

### Transliteration Pipeline

1. **Dictionary Lookup**: Primary translation using curated dictionary
2. **Rule-based Fallback**: Phonetic conversion rules for unknown words
3. **BERT Model**: Context-aware word prediction and validation
4. **Chunking**: Handles long sentences efficiently

## Installation

### Prerequisites

- Python 3.8+
- WhatsApp Business API access
- OpenAI/Anthropic API keys
- VoyageAI API key (for document embeddings)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Whatsapp-Singlish-Chatbot.git
   cd Whatsapp-Singlish-Chatbot
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   Create a `.env` file with the following variables:

   ```env
   # WhatsApp Business API
   ACCESS_TOKEN=your_whatsapp_access_token
   YOUR_PHONE_NUMBER=your_phone_number
   APP_ID=your_app_id
   APP_SECRET=your_app_secret
   RECIPIENT_WAID=recipient_whatsapp_id
   VERSION=v17.0
   PHONE_NUMBER_ID=your_phone_number_id
   VERIFY_TOKEN=your_verify_token

   # AI Service APIs
   ANTHROPIC_API_KEY=your_anthropic_api_key
   VOYAGER_API_KEY=your_voyageai_api_key
   ```

4. **Download Language Models**
   The application uses the Sinhala BERT model from Hugging Face:

   ```python
   model_directory = "Ransaka/sinhala-bert-medium-v2"
   ```

5. **Prepare Dictionary**
   Ensure the `data/dictionary.txt` file contains your Singlish-Sinhala word mappings.

## Usage

### Running the Application

1. **Start the Flask server**

   ```bash
   python run.py
   ```

   The server will start on `http://0.0.0.0:8000`

2. **Configure WhatsApp Webhook**
   - Set your webhook URL to: `https://yourdomain.com/webhook`
   - Use the `VERIFY_TOKEN` from your `.env` file

### Testing

Run the test script to verify transliteration:

```bash
python test.py
```

### API Endpoints

- `GET /webhook` - WhatsApp webhook verification
- `POST /webhook` - Process incoming WhatsApp messages

## How It Works

### Message Processing Flow

1. **Webhook Reception**: WhatsApp sends messages to the `/webhook` endpoint
2. **Security Verification**: Signature validation ensures request authenticity
3. **Message Parsing**: Extract text content and metadata
4. **Content Analysis**: Determine if message contains Singlish text or documents
5. **Processing**: Apply appropriate handler (transliteration or document processing)
6. **Response Generation**: Create appropriate response using AI model
7. **Reply**: Send processed response back via WhatsApp API

### Transliteration Process

1. **Input Analysis**: Parse Singlish sentence into words
2. **Dictionary Lookup**: Find direct translations for known words
3. **Rule-based Processing**: Apply phonetic rules for unknown words
4. **Context Modeling**: Use BERT to validate word choices in context
5. **Optimization**: Select best word combinations based on probability scores
6. **Output Generation**: Return properly formatted Sinhala text

### Document Processing

1. **PDF Upload**: Receive document via WhatsApp
2. **Text Extraction**: Extract text content from PDF
3. **Chunking**: Split text into manageable segments
4. **Embedding**: Generate vector embeddings using VoyageAI
5. **Storage**: Store in FAISS vector database
6. **Query Processing**: Enable similarity search for user questions

## Project Structure

```
Whatsapp-Singlish-Chatbot/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── doc_reader.py             # Document processing
│   ├── views.py                  # Flask routes and handlers
│   ├── decorators/
│   │   └── security.py           # Security decorators
│   └── utils/
│       └── whatsapp_utils.py     # WhatsApp API utilities
├── transliterator/               # Transliteration engine
│   ├── __init__.py
│   ├── chunker.py               # Text chunking utilities
│   ├── dictionary.py            # Dictionary management
│   ├── forward.py               # Forward transliteration
│   ├── model.py                 # BERT model wrapper
│   ├── rule_based.py            # Rule-based transliteration
│   ├── transliteration.py       # Main transliteration logic
│   └── utils.py                 # Utility functions
├── data/
│   └── dictionary.txt           # Singlish-Sinhala dictionary
├── media/                       # Uploaded media files
├── requirements.txt             # Python dependencies
├── run.py                       # Application entry point
├── test.py                      # Testing script
└── README.md                    # This file
```

## Dependencies

### Core Dependencies

- **Flask**: Web framework for webhook handling
- **transformers**: Hugging Face transformers for BERT model
- **torch**: PyTorch for deep learning models
- **langchain-anthropic**: Anthropic Claude integration
- **langchain-community**: Community LangChain components
- **faiss-cpu**: Vector similarity search

### API Integrations

- **requests**: HTTP client for API calls
- **aiohttp**: Async HTTP client
- **easygoogletranslate**: Google Translate fallback

### Document Processing

- **pypdf**: PDF text extraction
- **langchain-text-splitters**: Text chunking
- **langchain_voyageai**: VoyageAI embeddings
- **openpyxl**: Excel file support

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ransaka/sinhala-bert-medium-v2](https://huggingface.co/Ransaka/sinhala-bert-medium-v2) for the Sinhala BERT model
- WhatsApp Business API for messaging platform
- Anthropic Claude for conversational AI
- VoyageAI for document embeddings
- LangChain for AI integration framework

## Support

For support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a research/educational project. Ensure compliance with WhatsApp's Terms of Service and local regulations when deploying in production.
