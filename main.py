from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
# from transformers import pipeline
from keybert import KeyBERT
from fpdf import FPDF

import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from typing import  List,Optional
import urllib.parse
from googletrans import Translator
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Configuration
class Config:
    MAX_TRANSCRIPT_LENGTH = 1024 * 4  # 4K characters
    MAX_SUMMARY_LENGTH = 500
    MIN_SUMMARY_LENGTH = 200
    PDF_OUTPUT_DIR = "pdfs"
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'hi': 'Hindi',
        'te': 'Telugu',
        'af': 'Afrikaans',
        'sq': 'Albanian',
        'am': 'Amharic',
        'ar': 'Arabic',
        'hy': 'Armenian',
        'az': 'Azerbaijani',
        'eu': 'Basque',
        'be': 'Belarusian',
        'bs': 'Bosnian',
        'bg': 'Bulgarian',
        'my': 'Burmese',
        'ca': 'Catalan',
        'ceb': 'Cebuano',
        'co': 'Corsican',
        'hr': 'Croatian',
        'cs': 'Czech',
        'da': 'Danish',
        'nl': 'Dutch',
        'eo': 'Esperanto',
        'et': 'Estonian',
        'fil': 'Filipino',
        'fi': 'Finnish',
        'fr': 'French',
        'gl': 'Galician',
        'ka': 'Georgian',
        'de': 'German',
        'el': 'Greek',
        'gu': 'Gujarati',
        'ht': 'Haitian Creole',
        'ha': 'Hausa',
        'haw': 'Hawaiian',
        'iw': 'Hebrew',
        'hmn': 'Hmong',
        'hu': 'Hungarian',
        'is': 'Icelandic',
        'ig': 'Igbo',
        'id': 'Indonesian',
        'ga': 'Irish',
        'it': 'Italian',
        'ja': 'Japanese',
        'kn': 'Kannada',
        'kk': 'Kazakh',
        'kha': 'Khasi',
        'km': 'Khmer',
        'ko': 'Korean',
        'kri': 'Krio',
        'ku': 'Kurdish',
        'ky': 'Kyrgyz',
        'lo': 'Lao',
        'la': 'Latin',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'lb': 'Luxembourgish',
        'mk': 'Macedonian',
        'mg': 'Malagasy',
        'ms': 'Malay',
        'ml': 'Malayalam',
        'mt': 'Maltese',
        'mi': 'Māori',
        'mr': 'Marathi',
        'mn': 'Mongolian',
        'ne': 'Nepali',
        'no': 'Norwegian',
        'fa': 'Persian',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'pa': 'Punjabi',
        'ro': 'Romanian',
        'ru': 'Russian',
        'sm': 'Samoan',
        'gd': 'Scottish Gaelic',
        'sr': 'Serbian',
        'sn': 'Shona',
        'sd': 'Sindhi',
        'si': 'Sinhala',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'so': 'Somali',
        'es': 'Spanish',
        'su': 'Sundanese',
        'sw': 'Swahili',
        'sv': 'Swedish',
        'tg': 'Tajik',
        'ta': 'Tamil',
        'th': 'Thai',
        'tr': 'Turkish',
        'uk': 'Ukrainian',
        'ur': 'Urdu',
        'uz': 'Uzbek',
        'vi': 'Vietnamese',
        'cy': 'Welsh',
        'xh': 'Xhosa',
        'yi': 'Yiddish',
        'yo': 'Yoruba',
        'zu': 'Zulu'
    }

# Initialize NLP models
try:
    keyword_extractor = KeyBERT()
    translator=Translator()
    logger.info("NLP models loaded successfully")
except Exception as e:
    logger.error(f"Error loading NLP models: {e}")
    raise

# Create PDF directory if it doesn't exist
os.makedirs(Config.PDF_OUTPUT_DIR, exist_ok=True)



def chunk_text(text: str, max_length: int) -> List[str]:
    """
    Split text into chunks based on common sentence delimiters.
    
    Args:
        text: Input text to be chunked
        max_length: Maximum number of words per chunk
    
    Returns:
        List of text chunks
    """
    # Define sentence ending punctuation
    delimiters = '[.!?]+'
    
    # Split text into rough sentences
    rough_sentences = [s.strip() for s in re.split(delimiters, text) if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in rough_sentences:
        sentence_length = len(sentence.split())
        
        # If single sentence is longer than max_length, split by words
        if sentence_length > max_length:
            words = sentence.split()
            for i in range(0, len(words), max_length):
                chunk = " ".join(words[i:i + max_length])
                chunks.append(chunk)
            continue
            
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
        
    return chunks


def translate_text(text: str, dest: str, src: Optional[str] = None):
    """
    Translates the input text to the specified language.

    Args:
        text (str): The text to be translated.
        dest (str): The target language code.
        src (Optional[str]): The source language code (optional).

    Returns:
        str: The translated text.

    Raises:
        ValueError: If the input text is empty or translation fails.
        Exception: For other unexpected errors.
    """
    try:
        if not text:
            raise ValueError("Empty text provided for translation")

        # If input is a list, join into a single string
        if isinstance(text, list):
            text = ' '.join(text)

        # Split the text into manageable chunks
        max_chunk_size = 4500
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

        translated_chunks = []

        for chunk in chunks:
            if chunk.strip():  # Only translate non-empty chunks
                if src:
                    translation = translator.translate(chunk, dest=dest, src=src)
                else:
                    translation = translator.translate(chunk, dest=dest)

                # Append the translated chunk to the results
                if translation and translation.text:
                    translated_chunks.append(translation.text)

        if not translated_chunks:
            raise ValueError("Translation failed - no valid output")

        # Combine translated chunks into a single string
        return ' '.join(translated_chunks)

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    url = urllib.parse.unquote(url)
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL format")

def get_transcript(video_id: str, language: str) -> str:
    """Fetch and format video transcript."""
    try:
        if not video_id:
            raise ValueError("Invalid video ID")
        key_list=list(Config.SUPPORTED_LANGUAGES.keys())
        if language in key_list:
           key_list.remove(language)  # Remove it if already present
           key_list.insert(0, language)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=key_list)
        # print("trnscripted_list:",transcript_list)
        if not transcript_list:
            raise ValueError("No transcript available")
        transcript_text = str("\n".join(line.get("text") for line in transcript_list))
        print("transcripted text:",transcript_text)
        if not transcript_text:
            raise ValueError("Empty transcript")
        # Truncate if too long
        return transcript_text
    except Exception as e:
        logger.error(f"Error fetching transcript for video {video_id}: {e}")
        return f"no Transcription found for the {video_id}"

def generate_summary(text:str):
    model=genai.GenerativeModel("gemini-1.5-flash")
    PROMPT=f"""Summarize the key points of this YouTube video, highlighting the main topics, insights, or steps discussed. Provide a concise and clear summary without missing important details, and structure it in bullet points or a short paragraph for easy understanding. Ignore any unrelated filler content or repetitive information.->youtube content:{text}"""
    if model.count_tokens(PROMPT).total_tokens:
       response=model.generate_content(PROMPT)
    else:
        return "The following video content is longer than expected."
    print(response.text)
    keywords = keyword_extractor.extract_keywords(response.text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return response.text,keywords

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Return supported languages."""
    return jsonify(Config.SUPPORTED_LANGUAGES)

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate summary from YouTube video."""
    try:
        data = request.json
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
            
        video_url = data['url']
        language = data.get('language', 'en')
        if not video_url:
            return jsonify({'error': 'Empty URL provided'}), 400
        if language not in Config.SUPPORTED_LANGUAGES:
            return jsonify({'error': 'Unsupported language'}), 400
        
        # Extract video ID and get transcript
        video_id = extract_video_id(video_url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        transcript = get_transcript(video_id, language)
        if language != 'en':
            transcript = translate_text(transcript,dest="en",src=language)
        # Generate summary and keywords
        summary, keywords = generate_summary(transcript)
        if not summary:
            return jsonify({'error': 'Failed to generate summary'}), 500
        if language!='en':
            summary=translate_text(summary,dest=language)
        response_data = {
            'summary': summary,
            'keywords': keywords,
            'video_id': video_id,
            'language': language
        }
        
        return jsonify(response_data)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@app.route('/download', methods=['POST'])
def download_pdf():
    """Generate and download PDF summary."""
    data = request.json
    summary = data['summary']
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.multi_cell(190, 10, summary)
    
    pdf_path = 'pdfs/summary.pdf'
    pdf.output(pdf_path)
    return send_file(
            pdf_path,
            as_attachment=True,
            download_name=os.path.basename(pdf_path),
            mimetype='application/pdf'
        )
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
