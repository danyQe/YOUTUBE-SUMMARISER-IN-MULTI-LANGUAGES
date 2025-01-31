# YOUTUBE-SUMMARISER-IN-MULTI-LANGUAGES
## Overview

This project is a Flask-based web application that summarizes YouTube videos in multiple languages. It extracts the transcript of a YouTube video, translates it if necessary, and generates a concise summary along with keywords.

## Features

- Extracts YouTube video transcripts.
- Supports multiple languages for transcript extraction and translation.
- Generates summaries and keywords from video transcripts.
- Provides an option to download the summary as a PDF.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd YOUTUBE-SUMMARISER-IN-MULTI-LANGUAGES
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the project root.
    - Add your Gemini API key:
        ```
        GEMINI_API_KEY=your_api_key_here
        ```

## Usage

1. Run the Flask application:
    ```sh
    python main.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Use the web interface to input a YouTube video URL and select the desired language for the summary.

## API Endpoints

- `GET /api/languages`: Returns the supported languages.
- `POST /summarize`: Generates a summary from a YouTube video.
- `POST /download`: Generates and downloads a PDF summary.

## Configuration

The application configuration is managed in the `Config` class within `main.py`. Key configurations include:

- `MAX_TRANSCRIPT_LENGTH`: Maximum length of the transcript.
- `MAX_SUMMARY_LENGTH`: Maximum length of the summary.
- `MIN_SUMMARY_LENGTH`: Minimum length of the summary.
- `PDF_OUTPUT_DIR`: Directory for storing generated PDFs.
- `SUPPORTED_LANGUAGES`: Dictionary of supported languages.

## Error Handling

The application includes error handling for common issues such as invalid URLs, unsupported languages, and internal server errors. Custom error messages are returned in JSON format.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- [FPDF](http://www.fpdf.org/)
- [Google Translate API](https://pypi.org/project/googletrans/)

## Contact

For any inquiries or issues, please contact [raogoutham374@gmail.com].