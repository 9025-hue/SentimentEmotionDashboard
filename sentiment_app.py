import gradio as gr
import pandas as pd
import io
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import base64
import tempfile
import os
import functools
import spacy

from transformers import pipeline

# Common colors for emotions/sentiments for plotting
DEFAULT_COLORS = {
    'positive': 'green', 'negative': 'red', 'neutral': 'blue',
    'joy': 'green', 'sadness': 'blue', 'anger': 'red', 'fear': 'purple',
    'disgust': 'darkgreen', 'surprise': 'orange', 'love': 'pink',
    'optimism': 'lightgreen', 'anticipation': 'gold', 'trust': 'cyan',
    'guilt': 'brown', 'shame': 'darkred', 'ambiguous': 'gray',
    'NONE': 'gray' # For models that might output 'NONE' or similar
}

# --- NEW: Emoji mapping for labels ---
EMOJI_MAP = {
    'positive': '😊', 'negative': '😞', 'neutral': '😐',
    'joy': '😁', 'sadness': '😭', 'anger': '😡', 'fear': '😨',
    'disgust': '🤢', 'surprise': '😲', 'love': '❤️',
    'optimism': '✨', 'anticipation': '⏳', 'trust': '🤝',
    'guilt': '😔', 'shame': '😳', 'ambiguous': '🤔',
    'NONE': '➖', # Default for unmapped or 'NONE'
    'POSITIVE': '👍', 'NEGATIVE': '👎', # For distilbert sst2 model's uppercase labels
}

@functools.lru_cache(maxsize=5)
def get_sentiment_analyzer(model_name):
    """Loads and caches the sentiment analysis/emotion analysis pipeline for the given model name."""
    print(f"Loading model: {model_name}...")
    try:
        analyzer = pipeline(
            "sentiment-analysis", # This pipeline type can also be used for emotion classification
            model=model_name,
            tokenizer=model_name
        )
        return analyzer
    except Exception as e:
        print(f"Error loading model {model_name}: {e}. Falling back to default sentiment model.")
        return pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

@functools.lru_cache(maxsize=1)
def get_spacy_nlp():
    """Loads and caches the spaCy English NLP model."""
    print("Loading spaCy model (en_core_web_sm)...")
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        print("Please ensure you have run: python -m spacy download en_core_web_sm")
        return None

def analyze_sentiment_or_emotion(text, analyzer, nlp_model):
    """Analyzes sentiment/emotion for a single piece of text and extracts noun chunks as keywords."""
    if not text.strip():
        return "N/A", 0.0, []

    result = analyzer(text)[0]
    label = result['label'] # This will be sentiment or emotion label
    confidence = result['score']
    
    keywords = []
    if nlp_model:
        doc = nlp_model(text)
        keywords = [chunk.text for chunk in doc.noun_chunks]
        keywords = [k for k in keywords if len(k.split()) > 1 or (len(k.split()) == 1 and len(k) > 3)]
    
    return label, confidence, keywords

def process_texts_batch(texts, analyzer, nlp_model):
    """Analyzes sentiment/emotion for a list of texts (batch processing) and extracts noun chunks as keywords."""
    if not texts:
        return [], [], []

    results = analyzer(texts)
    labels = [res['label'] for res in results]
    confidences = [res['score'] for res in results]

    all_keywords = []
    if nlp_model:
        for text in texts:
            doc = nlp_model(text)
            keywords = [chunk.text for chunk in doc.noun_chunks]
            keywords = [k for k in keywords if len(k.split()) > 1 or (len(k.split()) == 1 and len(k) > 3)]
            all_keywords.append(keywords)
    else:
        for text in texts:
            keywords = [word for word in text.split() if len(word) > 3]
            all_keywords.append(keywords)
        
    return labels, confidences, all_keywords

def generate_distribution_plot(df, label_column, plot_type="count", title_suffix="", colors_map=DEFAULT_COLORS):
    """Generates a bar plot of label distribution (count or percentage) and saves it to a temp file."""
    fig, ax = plt.subplots(figsize=(6, 4))

    if df.empty:
        ax.text(0.5, 0.5, "No data to display", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if plot_type == "count":
            label_data = df[label_column].value_counts()
            ax.set_ylabel('Number of Texts')
            plot_title = f'{label_column} Distribution (Count){title_suffix}'
        elif plot_type == "percentage":
            total = len(df)
            label_data = (df[label_column].value_counts() / total * 100).sort_index() if total > 0 else pd.Series()
            ax.set_ylabel('Percentage (%)')
            ax.set_ylim(0, 100)
            plot_title = f'{label_column} Distribution (Percentage){title_suffix}'
        else:
            raise ValueError("plot_type must be 'count' or 'percentage'")
        
        # Sort labels alphabetically for consistent plotting
        label_data = label_data.sort_index()

        bar_colors = [colors_map.get(s.lower(), 'gray') for s in label_data.index]
        
        label_data.plot(kind='bar', ax=ax, color=bar_colors)
        ax.set_title(plot_title)
        ax.set_xlabel(label_column)
        
        if plot_type == "percentage":
            for i, percentage in enumerate(label_data.values):
                ax.text(i, percentage + 2, f"{percentage:.1f}%", ha='center', va='bottom')

    plt.tight_layout()
    
    temp_plot_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.png')
    plt.savefig(temp_plot_file, format='png')
    plt.close(fig)
    temp_plot_file.close()

    return temp_plot_file.name

def sentiment_dashboard(model_name, confidence_threshold, text_input, file_upload):
    
    sentiment_analyzer_instance = get_sentiment_analyzer(model_name)
    nlp_model_instance = get_spacy_nlp()

    results_data = []
    
    if text_input and text_input.strip():
        label, confidence, keywords = analyze_sentiment_or_emotion(text_input, sentiment_analyzer_instance, nlp_model_instance)
        results_data.append({
            'Text': text_input,
            'Label': label,
            'Confidence': f"{confidence:.2f}",
            'Keywords': ", ".join(keywords) if keywords else "N/A"
        })

    if file_upload is not None:
        try:
            if file_upload.name.endswith('.txt'):
                with open(file_upload.name, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
            elif file_upload.name.endswith('.csv'):
                df_file = pd.read_csv(file_upload.name)
                if 'text' in df_file.columns:
                    lines = df_file['text'].tolist()
                elif len(df_file.columns) > 0:
                    lines = df_file.iloc[:, 0].astype(str).tolist()
                else:
                    lines = []
            else:
                return "Unsupported file type. Please upload .txt or .csv.", None, None, None, None, None, None, None

            if lines:
                batch_labels, batch_confidences, batch_keywords = process_texts_batch(lines, sentiment_analyzer_instance, nlp_model_instance)
                for i, line in enumerate(lines):
                    results_data.append({
                        'Text': line,
                        'Label': batch_labels[i],
                        'Confidence': f"{batch_confidences[i]:.2f}",
                        'Keywords': ", ".join(batch_keywords[i]) if batch_keywords[i] else "N/A"
                    })
            else:
                return "File is empty or contains no valid text.", None, None, None, None, None, None, None

        except Exception as e:
            return f"Error processing file: {e}", None, None, None, None, None, None, None
            
    if not results_data:
        return "Please enter text or upload a file.", None, None, None, None, None, None, None

    df_original_results = pd.DataFrame(results_data)
    
    df_original_results['Confidence_Numeric'] = pd.to_numeric(df_original_results['Confidence'])
    
    initial_count = len(df_original_results)
    
    df_filtered_results = df_original_results[df_original_results['Confidence_Numeric'] >= confidence_threshold].copy()
    
    filtered_count = len(df_filtered_results)

    filter_message = ""
    if initial_count > 0:
        filter_message = f"\n\n**Processed {initial_count} text(s). Displaying {filtered_count} text(s) with confidence ≥ {confidence_threshold:.2f}.**"
        if initial_count > filtered_count:
            filter_message += f"\n({initial_count - filtered_count} text(s) filtered out due to low confidence)."
    else:
        filter_message = "\n\nNo texts analyzed."
        
    if 'Confidence_Numeric' in df_filtered_results.columns:
        df_filtered_results = df_filtered_results.drop(columns=['Confidence_Numeric'])
    
    # Determine if it's an emotion model or sentiment model
    is_emotion_model = "emotion" in model_name.lower() or "go_emotions" in model_name.lower()
    label_type = "Emotion" if is_emotion_model else "Sentiment"
    
    sentiment_plot_path = generate_distribution_plot(df_filtered_results, 'Label', plot_type="count", title_suffix=" (Threshold Applied)")
    percentage_plot_path = generate_distribution_plot(df_filtered_results, 'Label', plot_type="percentage", title_suffix=" (Threshold Applied)")
    
    temp_csv_file_path = None
    if not df_filtered_results.empty:
        # --- NEW: Add emoji to label for CSV export (optional, but good for consistency) ---
        df_csv_export = df_filtered_results.copy()
        df_csv_export['Label'] = df_csv_export['Label'].apply(lambda x: f"{x} {EMOJI_MAP.get(x.lower(), '')}".strip())
        # --- END NEW ---

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as temp_file:
            df_csv_export.to_csv(temp_file.name, index=False)
            temp_csv_file_path = temp_file.name

    display_output = "### Analysis Results\n\n" + filter_message + "\n\n"
    if not df_filtered_results.empty:
        for _, row in df_filtered_results.head(5).iterrows():
            emoji = EMOJI_MAP.get(row['Label'].lower(), '') # Get emoji for display
            display_output += f"**Text:** {row['Text']}\n"
            display_output += f"**{label_type}:** {row['Label']} {emoji} (Confidence: {row['Confidence']})\n" # Dynamic label_type & emoji
            if row['Keywords'] != "N/A":
                highlighted_text = row['Text']
                for keyword in row['Keywords'].split(', '):
                    if keyword.strip() and keyword.strip() in highlighted_text:
                        highlighted_text = highlighted_text.replace(keyword.strip(), f"**{keyword.strip()}**")
                display_output += f"**Sentiment Drivers:** {highlighted_text}\n\n"
            else:
                display_output += "**Sentiment Drivers:** N/A\n\n"
        
        if len(df_filtered_results) > 5:
            display_output += f"... and {len(df_filtered_results) - 5} more results. See the table below for full details.\n\n"

    # --- NEW: Add emoji to label for table output ---
    df_table_display = df_filtered_results.copy()
    df_table_display['Label'] = df_table_display['Label'].apply(lambda x: f"{x} {EMOJI_MAP.get(x.lower(), '')}".strip())
    table_output = df_table_display.to_markdown(index=False)
    # --- END NEW ---
    
    stats_output = ""
    if not df_filtered_results.empty and file_upload is not None:
        label_counts_filtered = df_filtered_results['Label'].value_counts()
        total_texts_filtered = len(df_filtered_results)
        
        all_present_labels = sorted(df_filtered_results['Label'].unique())

        stats_output = f"### {label_type} Statistics (for Uploaded File - Threshold Applied)\n\n"
        stats_output += f"Total texts displayed (after filter): {total_texts_filtered}\n\n"
        
        if total_texts_filtered > 0:
            for label in all_present_labels:
                percentage = (label_counts_filtered.get(label, 0) / total_texts_filtered) * 100
                stats_output += f"- **{label.capitalize()}:** {percentage:.2f}%\n"
        else:
            stats_output += "No data after filtering for statistics.\n"
    elif text_input and not df_filtered_results.empty and file_upload is None:
         stats_output = f"### {label_type} Statistics\n\nStatistics only shown for uploaded files."
    else:
        stats_output = f"### {label_type} Statistics\n\nNo texts analyzed for statistics."
    
    model_descriptions = {
        "cardiffnlp/twitter-roberta-base-sentiment-latest": """
        ### Model Information & Limitations (cardiffnlp/twitter-roberta-base-sentiment-latest)
        * **Type:** Sentiment Analysis
        * **Labels:** 'negative', 'neutral', 'positive'.
        * **Strengths:** Excellent for social media text (Twitter-like).
        * **Limitations:** May not generalize well to formal text or other domains.
        """,
        "distilbert-base-uncased-finetuned-sst-2-english": """
        ### Model Information & Limitations (distilbert-base-uncased-finetuned-sst-2-english)
        * **Type:** Sentiment Analysis
        * **Labels:** 'POSITIVE', 'NEGATIVE'. (Note: No 'neutral' label).
        * **Strengths:** Good for general English text sentiment, simpler binary classification.
        * **Limitations:** Only two labels (positive/negative), might struggle with truly neutral statements.
        """,
        "j-hartmann/emotion-english-distilroberta-base": """
        ### Model Information & Limitations (j-hartmann/emotion-english-distilroberta-base)
        * **Type:** Emotion Detection
        * **Labels:** 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'.
        * **Strengths:** Provides fine-grained emotion classification.
        * **Limitations:** Trained on specific emotion datasets; may interpret nuances differently than human perception.
        """,
         "distilbert-base-uncased-emotion": """
        ### Model Information & Limitations (distilbert-base-uncased-emotion)
        * **Type:** Emotion Detection
        * **Labels:** 'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'.
        * **Strengths:** Another good option for detecting common emotions.
        * **Limitations:** No 'neutral' label; like all emotion models, can be sensitive to phrasing.
        """
    }
    model_info = model_descriptions.get(model_name, "### Model Information & Limitations\nNo specific description available for this model, or it's a fallback.")
    
    model_info += """
    \n### Keyword Extraction Method
    This dashboard now uses **spaCy's noun chunk extraction** for identifying keywords/sentiment drivers. This is more advanced than simple word splitting, focusing on meaningful noun phrases. Ensure 'en_core_web_sm' is downloaded (`python -m spacy download en_core_web_sm`).
    """

    return display_output, sentiment_plot_path, stats_output, percentage_plot_path, table_output, temp_csv_file_path, model_info

available_models = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "j-hartmann/emotion-english-distilroberta-base",
    "distilbert-base-uncased-emotion"
]

inputs = [
    gr.Dropdown(
        choices=available_models,
        value=available_models[0],
        label="Select Analysis Model (Sentiment or Emotion)"
    ),
    gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.5,
        step=0.01,
        label="Minimum Confidence Threshold (0.0 - 1.0)",
        info="Only show predictions with confidence at or above this value."
    ),
    gr.Textbox(lines=5, placeholder="Enter text here for sentiment/emotion analysis...", label="Text Input"),
    gr.File(label="Upload Text File (.txt or .csv)", file_types=[".txt", ".csv"])
]

outputs = [
    gr.Markdown(label="Analysis Summary & Highlighted Text"),
    gr.Image(label="Emotion/Sentiment Distribution (Count)", show_download_button=True),
    gr.Markdown(label="Batch Emotion/Sentiment Statistics"),
    gr.Image(label="Emotion/Sentiment Distribution (Percentage)", show_download_button=True),
    gr.Markdown(label="Full Results Table"),
    gr.File(label="Download Results (CSV)", file_count="single"),
    gr.Markdown(label="Model Information & Limitations")
]

demo = gr.Interface(
    fn=sentiment_dashboard,
    inputs=inputs,
    outputs=outputs,
    title="Interactive Sentiment & Emotion Analysis Dashboard",
    description="Analyze emotional tone or sentiment in text data. Select a model, set a confidence threshold, enter text directly or upload a .txt/.csv file for classification, confidence scoring, keyword extraction, and visualization.",
    flagging_mode="never",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()