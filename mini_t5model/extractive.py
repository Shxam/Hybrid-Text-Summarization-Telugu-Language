# %%
from transformers import AutoTokenizer, AutoModel


# %%
import torch
import numpy as np
from tqdm import tqdm


# %%
model_name = "ai4bharat/IndicBERTv2-MLM-only"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
model = AutoModel.from_pretrained(model_name)
model.eval()


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# %%
import pandas as pd
train_df = pd.read_json("train_data.json", lines=True)

# %%
train_df['text'][0]

# %%
test_df= pd.read_json("test_data.json", lines=True)
valid_df = pd.read_json("valid_data.json", lines=True)

# %%
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load model and move to device
model_name = "ai4bharat/IndicBERTv2-MLM-only"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mean-pooling based sentence embeddings
def get_sentence_embeddings(sentences, batch_size=32, max_len=128):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        batch_embeds = output.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeds)
    return np.array(embeddings)

# Simple TextRank based on similarity sum
def textrank_summary(sentences, embeddings):
    if not sentences:
        return []
    top_k = max(1, int(len(sentences) * 0.70))
    if len(sentences) == 1:
        return sentences
    if len(sentences) <= top_k:
        return sentences
    sim_matrix = cosine_similarity(embeddings)
    scores = sim_matrix.sum(axis=1)
    ranked_ids = np.argsort(scores)[::-1]
    selected = sorted(ranked_ids[:top_k])
    return [sentences[i] for i in selected]

# Process a batch of rows from the DataFrame
def summarize_batch(df_batch, text_col="text"):
    summaries = []
    for sentences in df_batch[text_col]:
        if not sentences:
            summaries.append("")
            continue
        embeddings = get_sentence_embeddings(sentences)
        summary = textrank_summary(sentences, embeddings)
        summaries.append(" ".join(summary))
    return summaries


# Main loop to process DataFrame in batches
def process_dataframe_in_batches(df, text_col="text", batch_size=100):
    summaries = []
    for i in tqdm(range(0, len(df), batch_size), desc="Batch processing"):
        df_batch = df.iloc[i:i + batch_size]
        batch_summaries = summarize_batch(df_batch, text_col=text_col)
        summaries.extend(batch_summaries)
    df["extractive_summary"] = summaries
    return df



# %%
# Ensure the first sentence is included in the best_sentences
best_sentences = (article_sentences[0],) + tuple(sent for sent in best_sentences if sent != article_sentences[0])
print("Updated Best Sentences:")
for sent in best_sentences:
    print("-", sent)

# %%
from rouge import Rouge
from itertools import combinations
def find_best_extractive_summary(article_sentences, abstractive_summary, length_percentage=1.0):
    rouge = Rouge()
    best_score = 0
    best_combination = []
    total_length = sum(len(sentence) for sentence in article_sentences)
    max_length = int(total_length * length_percentage)

    for r in range(1, len(article_sentences) + 1):
        for combo in combinations(article_sentences, r):
            combined = " ".join(combo)
            combined_length = sum(len(sentence.split()) for sentence in combo)
            if combined_length > max_length:
                continue
            score = rouge.get_scores(combined, abstractive_summary, avg=True)["rouge-l"]["f"]
            if score > best_score:
                best_score = score
                best_combination = combo

    return best_combination, best_score

# %%

# Process the first row of the DataFrame as an example
article_sentences = train_df['text']  # Extract the list of sentences from the first row
abstractive_summary = train_df['summary']  # Extract the summary from the first row


# %%
len(article_sentences)

# %%

# Flatten the article_sentences to a single list of strings
flat_article_sentences = [sentence for sublist in article_sentences for sentence in article_sentences]


# %%

# Call the function with the flattened list
best_sentences, best_score = find_best_extractive_summary(flat_article_sentences, abstractive_summary, length_percentage=1)


# %%

print("Best Extractive Sentences:")
for sent in best_sentences:
    print("-", sent)
print("\nROUGE-L F1 Score:", best_score)


# %%
# Assuming your original DataFrame is df and it has a 'sentences' column
train_df = process_dataframe_in_batches(train_df, text_col="text",batch_size=100)

# %%
train_df['extractive_summary'][0]  # Check the summary of the first row

# %%
train_df['extractive_summary'][0]  # Check the summary of the first row

# %%
train_df['summary'][0]

# %%
test_df = process_dataframe_in_batches(test_df, text_col="text", batch_size=100)

# %%
valid_df = process_dataframe_in_batches(valid_df, text_col="text", batch_size=100)

# %%
test_df['extractive_summary'][0]  # Check the summary of the first row

# %%
len(test_df['extractive_summary'][0])

# %%
test_df['text'][0]  # Check the summary of the first row

# %%
len(" ".join(test_df['text'][0]))

# %%
telugu_stopwords = sorted(list(set([
    'అక్కడ', 'అడగండి', 'అడగడం', 'అన్ని', 'అన్ని విధాలుగా', 'ఆయన', 'అనుగుణంగా', 'అనుమతించు',
    'అనుమతిస్తుంది', 'అయితే', 'అయితేనూ', 'అయినా', 'అయినా సరే', 'అయినప్పటికీ', 'అరా',
    'అవిన్ని', 'అవి', 'అవి మాత్రమే', 'అప్పుడు', 'అప్పటికి', 'అప్పట్లో', 'అతను',
    'అతని', 'అతనిని', 'అతనితో', 'అతను మాత్రమే', 'అధికంగా', 'అది', 'అని', 'అపుడు',
    'అడ్డంగా', 'అందులో', 'అందులోని', 'అందరూ', 'అందుబాటులో', 'అందువల్ల', 'అంతవరకు',
    'అయిన', 'ఇక్కడ', 'ఇక్కడికి', 'ఇంకా', 'ఇందులో', 'ఇతర', 'ఇతరులు', 'ఇప్పటికీ',
    'ఇప్పటివరకు', 'ఇప్పుడే', 'ఇప్పటికే', 'ఇప్పుడు', 'ఇంతవరకు', 'ఇది', 'ఈ', 'ఉన్నారు',
    'ఉన్న', 'ఉన్నప్పుడు', 'ఉన్నప్పటికీ', 'ఉన్నదే', 'ఉంటుంది', 'ఉంటూ', 'ఉంటారు', 'ఉండదు',
    'ఉండలేదు', 'ఉండగా', 'ఉండాలి', 'ఉండేది', 'ఉండవచ్చు', 'ఉంటే', 'ఎక్కడ', 'ఎక్కడైనా',
    'ఎవరైనా', 'ఎవరైనా', 'ఎవరూ', 'ఎవరు', 'ఎవరికైనా', 'ఎవరో', 'ఎప్పటికప్పుడు', 'ఎప్పటికీ',
    'ఎప్పుడూ', 'ఎప్పుడు', 'ఎంత', 'ఎలా', 'ఎవరి', 'ఎందుకు', 'ఎందుకంటే', 'ఏ', 'ఏదైనా',
    'ఏది', 'ఏమైనప్పటికి', 'ఒక', 'ఒకటి', 'ఒకరు', 'ఒక ప్రక్కన', 'కాబట్టి', 'కాస్త',
    'కావచ్చు', 'కనుక', 'కనిపిస్తాయి', 'కింద', 'కెళ్లారు', 'కూడదు', 'కూడా', 'కంటే',
    'కాదు', 'కానీ', 'కి', 'కేసి', 'గురించి', 'గా', 'చాలా', 'చేయగలిగింది', 'చేయబడింది',
    'చేయి', 'చేయాల్సి', 'చుట్టూ', 'చివరి', 'చేసారు', 'చేసిన', 'తగిన', 'తక్కువ', 'తర్వాత',
    'తర్వాతనే', 'తర్వాతి', 'తరువాత', 'తరువాతి', 'తిరిగి', 'తప్ప', 'తీసుకున్నారు', 'తీసి',
    'దాదాపు', 'దయచేసి', 'దూరంగా', 'నిజంగా', 'పైన', 'పోయింది', 'పోయిన', 'పొందిన', 'ప్రతి',
    'ప్రకారం', 'ప్రక్కన', 'బహుశా', 'మధ్య', 'మధ్యలో', 'మరియు', 'మరొక', 'మరొకటి', 'మామూలుగా',
    'మాత్రమే', 'మాత్రం', 'మొదటి', 'ముఖ్యంగా', 'ముందు', 'మళ్ళీ', 'మేము', 'మెచ్చుకో', 'మీరు',
    'మీద', 'యొక్క', 'రెండు', 'లేకపోతే', 'లేదంటే', 'లేదని', 'లేదూ', 'లేవు', 'లో', 'లోపల',
    'వంటి', 'వెంట', 'వెంటనే', 'వెళ్లారు', 'వెనుక', 'వద్ద', 'వద్దు', 'వారు', 'వేరుగా', 'వైపు',
    'విజయవంతంగా', 'విషయంలో', 'వ్యతిరేకంగా', 'వచ్చారు', 'వచ్చింది', 'వచ్చిన', 'వచ్చినప్పుడు',
    'వచ్చేందుకు', 'వాటితో', 'వాటిని', 'వీరి', 'వీరిని', 'వీరిలో', 'వీరు', 'వీటిని', 'శుభంగా',
    'సంబంధం', 'సమయం', 'హఠాత్తుగా','వలన', 'తో', 'నుంచి', 'నించి', 'మీద', 'పై', 'దగ్గర', 'పట్ల', 'వైపు', 'కలిగి', 'నుండి',
    'దగ్గరకి', 'లోపల', 'క్రింద', 'పైకి', 'బయట', 'మొదలైనవి', 'చే', 'చేత', 'కొరకు', 'కోసం',
    'వంటి', 'మొదలు', 'వరకు', 'తరువాత', 'తరువాతి', 'తిరిగి', 'తప్ప', 'ఇప్పటి వరకు',
    'అంత వరకు', 'ఎవరైనా', 'ఎవరూ', 'ఎప్పుడైనా', 'ఎప్పుడూ', 'ఎప్పటికీ'
])))


# %%
def remove_telugu_stopwords(text):
    if not isinstance(text, str):
        return text
    words = text.split()
    filtered_words = [word for word in words if word not in telugu_stopwords]
    return ' '.join(filtered_words)


# %%

# 3. Apply to DataFrame column
test_df['cleaned_summary'] = test_df['extractive_summary'].apply(remove_telugu_stopwords)
valid_df['cleaned_summary'] = valid_df['extractive_summary'].apply(remove_telugu_stopwords)
train_df['cleaned_summary'] = train_df['extractive_summary'].apply(remove_telugu_stopwords)

# %%
train_df.to_json("train_data_with_clean_extractive_summaries.json", orient="records", lines=True)
test_df.to_json("test_data_with_clean_extractive_summaries.json", orient="records", lines=True)
valid_df.to_json("valid_data_with_clean_extractive_summaries.json", orient="records", lines=True)

# %%



