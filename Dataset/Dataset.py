import pandas as pd
from sklearn.model_selection import train_test_split
import ftfy
import html
import re
import matplotlib.pyplot as plt
import os

# ================================================================================================================ START
# Dataset
# ======================================================================================================================
# ----------------------------------------------------------------------------- START
# DATASET SETUP
# -----------------------------------------------------------------------------
def label_sentiment(score):
    score_text = str(score)
    match = re.search(r"\d", score_text)

    if match is None:
        print("No Rating!")
        return None

    score = int(match.group())
    if int(score) <= 2:
        return "neg"
    elif int(score) == 3:
        return "neu"
    else:
        return "pos"

# Fashion
df_fashion1 = pd.read_csv(r'Dataset/Reviews/amazon-fashion-800k+-user-reviews-dataset.csv', engine='python', on_bad_lines='warn')
df_fashion1 = df_fashion1.rename(columns={
    'rating': 'Score',
    'text': 'Text'
})
df_fashion1 = df_fashion1.dropna(subset=['Text', 'Score'])
df_fashion1 = df_fashion1[df_fashion1['Text'].str.len() > 0]
df_fashion1['Sentiment'] = df_fashion1['Score'].apply(label_sentiment)
df_fashion1['Source'] = 'fashion_1'
df_fashion1['Domain'] = 'Fashion'
df_fashion1 = df_fashion1[['Text', 'Score', 'Sentiment', 'Source', 'Domain']]

# Books
df_books1 = pd.read_csv(r'Dataset/Reviews/Books_rating.csv', engine='python', on_bad_lines='warn')
df_books1 = df_books1.rename(columns={
    'review/score': 'Score',
    'review/text': 'Text'
})
df_books1 = df_books1.dropna(subset=['Text', 'Score'])
df_books1 = df_books1[df_books1['Text'].str.len() > 0]
df_books1['Sentiment'] = df_books1['Score'].apply(label_sentiment)
df_books1['Source'] = 'books_1'
df_books1['Domain'] = 'Books'
df_books1 = df_books1[['Text', 'Score', 'Sentiment', 'Source', 'Domain']]

# Electronics
df_electronics1 = pd.read_csv(r'Dataset/Reviews/electronics_small.csv', engine='python', on_bad_lines='warn')
df_electronics1 = df_electronics1.rename(columns={
    'overall': 'Score',
    'reviewText': 'Text'
})
df_electronics1 = df_electronics1.dropna(subset=['Text', 'Score'])
df_electronics1 = df_electronics1[df_electronics1['Text'].str.len() > 0]
df_electronics1['Sentiment'] = df_electronics1['Score'].apply(label_sentiment)
df_electronics1['Source'] = 'electronics_1'
df_electronics1['Domain'] = 'Electronics'
df_electronics1 = df_electronics1[['Text', 'Score', 'Sentiment', 'Source', 'Domain']]

df = pd.concat([df_fashion1, df_books1, df_electronics1], ignore_index=True)
df['Text_clean_tmp'] = df['Text'].str.lower().str.strip()
df = df.drop_duplicates(subset=['Text_clean_tmp'])
df = df.drop(columns=['Text_clean_tmp'])
balanced_df = (df.groupby(["Domain", "Sentiment"], group_keys=False)
               .sample(n=100000, random_state=142)
               .sample(frac=1, random_state=142)
               .reset_index(drop=True)
               )
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# DATASET PREPROCESSING
# -----------------------------------------------------------------------------
print("\n========== DATA PREPROCESSING ==========")
bad_encoding_pattern = r'[ÃÂâ]'
bad_encoding_before = balanced_df['Text'].str.contains(
    bad_encoding_pattern,
    regex=True,
    na=False
).sum()
print("Reviews With Encoding Issues:" + str(bad_encoding_before))

balanced_df['Text'] = balanced_df['Text'].astype(str).apply(ftfy.fix_text)  # Fix broken encoding (Ã, â, etc.)
bad_encoding_after = balanced_df['Text'].str.contains(
    bad_encoding_pattern,
    regex=True,
    na=False
).sum()
print("Fixed " + str(bad_encoding_before - bad_encoding_after) + " Reviews...")
balanced_df = balanced_df[~balanced_df['Text'].str.contains(bad_encoding_pattern, regex=True, na=False)]
print("Dropped " + str(bad_encoding_after) + " Reviews...")

balanced_df['Text'] = balanced_df['Text'].str.replace(r'<[^>]+>', '', regex=True)  # Remove HTML tags (<br>, <a>, etc)
print("Removed HTML Tags...")

balanced_df['Text'] = balanced_df['Text'].apply(html.unescape)  # Decode HTML characters (&eacute;, &amp;)
print("Decoded HTML Characters...")

balanced_df['Text'] = balanced_df['Text'].str.replace('�', '', regex=False)
print("Removed Broken Characters...")

balanced_df['Text'] = balanced_df['Text'].str.replace(r'\s+', ' ', regex=True).str.strip()  # Replace consecutive white spaces
print("Removed Consecutive Whitespaces...")

before_drop = len(balanced_df)
balanced_df = balanced_df.dropna(subset=["Text", "Sentiment"]).copy()
balanced_df["Text"] = balanced_df["Text"].astype(str).str.strip()
balanced_df = balanced_df[balanced_df["Text"] != ""]
balanced_df = balanced_df.reset_index(drop=True)
after_drop = len(balanced_df)
print("Dropped empty or invalid rows after preprocessing:", before_drop - after_drop)
print("Remaining missing Text:", balanced_df["Text"].isna().sum())
print("Remaining empty Text:", (balanced_df["Text"].str.strip() == "").sum())
balanced_df["Text"] = (
    balanced_df["Text"]
    .str.replace("’", "'", regex=False)
    .str.replace("‘", "'", regex=False)
    .str.replace("`", "'", regex=False)
)

# Continue with more <==================================================================================================

balanced_df = (balanced_df.groupby(["Domain", "Sentiment"], group_keys=False)
               .sample(n=50000, random_state=142)
               .sample(frac=1, random_state=142)
               .reset_index(drop=True)
               )
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# DATASET INFORMATION
# -----------------------------------------------------------------------------
print("\n========== DATASET INFO ==========")
print("DATASET SIZE: " + str(len(balanced_df)))
print(balanced_df['Sentiment'].value_counts())
print(balanced_df["Domain"].value_counts())

# balanced_df = balanced_df.sample(frac=1, random_state=42)
# balanced_df = balanced_df.head(1000)
# balanced_df = balanced_df.reset_index(drop=True)
# print("\nSUBSET SIZE: " + str(len(balanced_df)))

balanced_df["Stratify"] = (balanced_df["Domain"] + "_" + balanced_df["Sentiment"])
text = balanced_df['Text']
sentiment = balanced_df['Sentiment']
train_val_df, test_df = train_test_split(
    balanced_df,
    test_size=0.2,
    random_state=42,
    stratify=balanced_df["Stratify"]
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    random_state=42,
    stratify=train_val_df['Stratify']
)

text_train = train_df['Text']
sentiment_train = train_df['Sentiment']

text_test = test_df['Text']
sentiment_test = test_df['Sentiment']

text_val = val_df['Text']
sentiment_val = val_df['Sentiment']

text_train = text_train.reset_index(drop=True)
sentiment_train = sentiment_train.reset_index(drop=True)

text_test = text_test.reset_index(drop=True)
sentiment_test = sentiment_test.reset_index(drop=True)

text_val = text_val.reset_index(drop=True)
sentiment_val = sentiment_val.reset_index(drop=True)

print("\n" + str(int((len(sentiment_train) / len(balanced_df)) * 100)) + "% TRAIN SPLIT: "
      + str(len(sentiment_train)) + "\n" + str(sentiment_train.value_counts())
     )
print("\n" + str(int((len(sentiment_val) / len(balanced_df)) * 100)) + "% VALIDATION SPLIT: "
      + str(len(sentiment_val)) + "\n" + str(sentiment_val.value_counts())
     )
print("\n" + str(int((len(sentiment_test) / len(balanced_df)) * 100)) + "% TEST SPLIT: "
      + str(len(sentiment_test)) + "\n" + str(sentiment_test.value_counts())
     )
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# META POLARITY FEATURES
# -----------------------------------------------------------------------------
NEGATION_CONTRACTION_REPLACEMENTS = {
        r"\bcan't\b": "can not",
        r"\bcant\b": "can not",
        r"\bcannot\b": "can not",
        r"\bwon't\b": "will not",
        r"\bwont\b": "will not",
        r"\bdon't\b": "do not",
        r"\bdont\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdoesnt\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bdidnt\b": "did not",
        r"\bisn't\b": "is not",
        r"\bisnt\b": "is not",
        r"\baren't\b": "are not",
        r"\barent\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bwasnt\b": "was not",
        r"\bweren't\b": "were not",
        r"\bwerent\b": "were not",
        r"\bhasn't\b": "has not",
        r"\bhasnt\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhavent\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bhadnt\b": "had not",
        r"\bwouldn't\b": "would not",
        r"\bwouldnt\b": "would not",
        r"\bshouldn't\b": "should not",
        r"\bshouldnt\b": "should not",
        r"\bcouldn't\b": "could not",
        r"\bcouldnt\b": "could not",
        r"\bmustn't\b": "must not",
        r"\bmustnt\b": "must not",
        r"\bneedn't\b": "need not",
        r"\bneednt\b": "need not",
        r"\bshan't\b": "shall not",
        r"\bshant\b": "shall not",
        r"\bain't\b": "is not",
        r"\baint\b": "is not",
    }

NEG_AUX = (
    r"(?:"
    r"not|"
    r"cannot|can\s+not|can't|cant|"
    r"do\s+not|don't|dont|"
    r"does\s+not|doesn't|doesnt|"
    r"did\s+not|didn't|didnt|"
    r"is\s+not|isn't|isnt|"
    r"was\s+not|wasn't|wasnt|"
    r"are\s+not|aren't|arent|"
    r"were\s+not|weren't|werent|"
    r"will\s+not|won't|wont|"
    r"would\s+not|wouldn't|wouldnt|"
    r"should\s+not|shouldn't|shouldnt|"
    r"could\s+not|couldn't|couldnt|"
    r"has\s+not|hasn't|hasnt|"
    r"have\s+not|haven't|havent|"
    r"had\s+not|hadn't|hadnt"
    r")"
)

OPTIONAL_DEGREE = r"(?:really\s+|very\s+|so\s+|too\s+|quite\s+|at\s+all\s+)?"

NEGATION_WORDS = {
    "not", "no", "never",
    "cannot", "cant", "can't",
    "dont", "don't",
    "doesnt", "doesn't",
    "didnt", "didn't",
    "isnt", "isn't",
    "wasnt", "wasn't",
    "werent", "weren't",
    "wont", "won't",
    "wouldnt", "wouldn't",
    "shouldnt", "shouldn't",
    "couldnt", "couldn't",
    "aint", "ain't",
    "arent", "aren't",
    "hasnt", "hasn't",
    "havent", "haven't",
    "hadnt", "hadn't",
    "n't", "nt",
    "without"
}

INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "highly",
    "super", "ultra", "absolutely", "completely", "totally",
    "surprisingly", "ridiculously", "seriously", "terribly"
}

DIMINISHERS = {
    "slightly", "somewhat", "mildly", "partly",
    "partially", "kinda", "sorta", "barely", "hardly",
    "almost", "fairly"
}

POST_CONTRAST_MARKERS = {
    "but",
    "however",
    "yet",
    "nevertheless",
    "nonetheless"
}

CONCESSIVE_STARTERS = {
    "although",
    "though"
}

UNIVERSAL_PHRASE_RULES = [
    # =========================================================================
    # Neutral / mixed phrases
    # =========================================================================
    {
        "rule_key": "phrase:neutral_good_but_not_great",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:good|decent|okay|ok|fine)\s+but\s+not\s+"
            r"(?:great|amazing|excellent|perfect|the\s+best)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_not_bad_not_great",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\bnot\s+(?:bad|terrible|awful)\s+but\s+not\s+"
            r"(?:great|amazing|excellent|perfect)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_okay_but_issues",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:okay|ok|fine|decent|good)\s+but\s+"
            r"(?:has|have|had|with)\s+(?:some\s+)?"
            r"(?:issues|problems|flaws|drawbacks|downsides)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_works_but",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:works|worked|work)\s+but\s+"
            r"(?:not\s+perfect|not\s+great|has\s+issues|could\s+be\s+better|"
            r"there\s+are\s+issues|with\s+some\s+problems)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_decent_for_price",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:decent|okay|ok|fine|acceptable|reasonable)\s+"
            r"(?:for|given)\s+(?:the\s+)?(?:price|money|cost)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_average_nothing_special",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:average|mediocre|ordinary)\s+"
            r"(?:product|item|quality|book|read|purchase)\b|"
            r"\bnothing\s+(?:special|amazing|great|exceptional)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_pros_and_cons",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:pros\s+and\s+cons|good\s+and\s+bad|"
            r"some\s+good\s+and\s+some\s+bad|mixed\s+feelings|mixed\s+review)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_somewhat_disappointed",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:somewhat|slightly|a\s+little|kind\s+of|kinda)\s+"
            r"(?:disappointed|underwhelmed)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:neutral_expected_more",
        "polarity": "neu",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:expected|was\s+expecting)\s+"
            r"(?:a\s+)?(?:little\s+)?more\b|"
            r"\bnot\s+(?:quite|really)\s+what\s+i\s+expected\b",
            flags=re.IGNORECASE
        )
    },

    # =========================================================================
    # General negative phrases
    # =========================================================================
    {
        "rule_key": "phrase:not_worth_it",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\bnot\s+" + OPTIONAL_DEGREE +
            r"worth\s+(?:it|the\s+money|the\s+price|buying|getting|keeping)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:waste_of_money",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:a\s+)?(?:complete\s+|total\s+|real\s+|absolute\s+)?"
            r"waste\s+of\s+(?:money|time)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:low_quality",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:low|poor|bad|terrible|awful|horrible)\s+quality\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:cheaply_made",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:cheaply\s+made|"
            r"(?:feel|feels|felt|feeling)\s+cheap|"
            r"(?:material|fabric|plastic|product|item)\s+"
            r"(?:feel|feels|felt|feeling)\s+cheap)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:fell_apart",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:fell|came|comes|coming)\s+apart\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_lasting",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:"
            r"(?:did\s+not|didn't|didnt)\s+last|"
            r"not\s+lasting|"
            r"only\s+lasted|"
            r"lasted\s+(?:only\s+)?(?:a\s+)?(?:day|week|month|few\s+days|few\s+weeks)|"
            r"broke\s+(?:after|within)"
            r")\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_as_described",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:not|isn't|isnt|wasn't|wasnt|is\s+not|was\s+not)\s+"
            r"(?:as\s+)?described\b",
            flags=re.IGNORECASE
        )
    },

    # =========================================================================
    # Delivery / return / order negative phrases
    # =========================================================================
    {
        "rule_key": "phrase:wrong_item",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\bwrong\s+(?:item|product|model|version|book|charger|case)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:missing_parts",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\bmissing\s+(?:parts?|pieces?|accessories|components|items?)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:never_received",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:never\s+received|"
            r"did\s+not\s+receive|didn't\s+receive|didnt\s+receive|"
            r"have\s+not\s+received|haven't\s+received|havent\s+received)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_delivered",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:not|never|was\s+not|wasn't|wasnt|"
            r"has\s+not\s+been|hasn't\s+been|hasnt\s+been)"
            r"\s+delivered\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:had_to_return",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:had\s+to\s+return|"
            r"returned\s+(?:it|this|the\s+item|the\s+product|the\s+book))\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:want_refund",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:want|wanted|need|needed|request(?:ed)?|asking\s+for)"
            r"\s+(?:a\s+)?refund\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_satisfied",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\bnot\s+(?:very\s+|really\s+|fully\s+|completely\s+)?"
            r"(?:satisfied|happy)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_not_recommend",
        "polarity": "neg",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:would\s+not|wouldn't|wouldnt)\s+recommend\b",
            flags=re.IGNORECASE
        )
    },

    # =========================================================================
    # General positive phrases
    # =========================================================================
    {
        "rule_key": "phrase:not_bad",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bnot\s+(?:too\s+|that\s+|so\s+|very\s+)?bad\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_complaints",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?complaints?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_issues",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?issues?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_problems",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bno\s+(?:real\s+|major\s+|serious\s+)?problems?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_regrets",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bno\s+regrets?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_great",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bworks?\s+(?:really\s+|very\s+|so\s+)?great\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_perfectly",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bworks?\s+(?:perfectly|flawlessly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:works_as_expected",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bwork(?:s|ed)?\s+(?:exactly\s+)?as\s+expected\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:highly_recommend",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:highly|strongly|definitely)\s+recommend\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_recommend",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:would|will)\s+(?:definitely\s+|highly\s+)?recommend\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:would_buy_again",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:would|will)\s+(?:definitely\s+)?buy\s+(?:it\s+|this\s+)?again\b|"
            r"\bbuy\s+(?:it\s+|this\s+)?again\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:worth_every_penny",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bworth\s+every\s+(?:penny|cent)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:better_than_expected",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bbetter\s+than\s+(?:i\s+)?expected\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:exceeded_expectations",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\bexceeded\s+(?:my\s+)?expectations\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:could_not_be_happier",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:could\s+not|couldn't|couldnt)\s+be\s+happier\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:cannot_recommend_enough",
        "polarity": "pos",
        "domain": "general",
        "pattern": re.compile(
            r"\b(?:cannot|can\s+not|can't|cant)\s+recommend"
            r"(?:\s+(?:it|this|these|them))?\s+enough\b",
            flags=re.IGNORECASE
        )
    },

    # =========================================================================
    # Electronics negative phrases
    # =========================================================================
    {
        "rule_key": "phrase:does_not_work",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+work(?:s|ed|ing)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:stopped_working",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:stopped|stop|stops|quit|quits)\s+working\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:no_longer_works",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\bno\s+longer\s+work(?:s|ed|ing)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:dead_on_arrival",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:dead\s+on\s+arrival|doa)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_powering_on",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+(?:power\s+on|powering\s+on)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_turning_on",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+(?:turn\s+on|turning\s+on)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:not_charging",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+charg(?:e|es|ed|ing)\b|"
            r"\b(?:stopped|stop|stops|quit|quits)\s+charging\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:battery_drains_fast",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:battery|batteries)\s+"
            r"(?:drain|drains|drained|die|dies|died)\s+"
            r"(?:too\s+)?(?:fast|quickly|rapidly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:does_not_hold_charge",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            rf"\b(?:battery\s+)?(?:{NEG_AUX})\s+hold\s+"
            r"(?:a\s+|the\s+)?charge\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:keeps_disconnecting",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:keep|keeps|kept)\s+disconnecting\b|"
            r"\blosing\s+connection\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poor_connection",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:poor|bad|weak|unstable)\s+(?:connection|signal)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:overheats_quickly",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:overheat|overheats|overheated|overheating)\s+"
            r"(?:too\s+)?(?:quickly|fast|easily)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:screen_cracked",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:screen\s+cracked|cracked\s+screen)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:flickering_screen",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:screen|display)\s+(?:is\s+|was\s+|keeps\s+|kept\s+|started\s+)?"
            r"(?:flickering|flickers|flicker)\b|"
            r"\b(?:flickering|flickers|flicker)\s+(?:screen|display)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:touchscreen_not_responsive",
        "polarity": "neg",
        "domain": "electronics",
        "pattern": re.compile(
            r"\b(?:touch\s*screen|touchscreen|screen)\s+"
            r"(?:is\s+|was\s+)?not\s+responsive\b|"
            r"\b(?:touch\s*screen|touchscreen|screen)\s+"
            r"(?:does\s+not|doesn't|doesnt)\s+respond\b",
            flags=re.IGNORECASE
        )
    },

    # =========================================================================
    # Fashion phrases
    # =========================================================================
    {
        "rule_key": "phrase:not_fitting",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            rf"\b(?:{NEG_AUX})\s+fit(?:s|ted|ting)?\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:wrong_size",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\bwrong\s+size\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:wrong_color",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\bwrong\s+(?:color|colour)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:runs_small",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\b(?:run|runs|ran)\s+(?:too\s+|a\s+little\s+|a\s+bit\s+|really\s+)?small\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:runs_large",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\b(?:run|runs|ran)\s+(?:too\s+|a\s+little\s+|a\s+bit\s+|really\s+)?(?:large|big)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poor_fit",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\b(?:poor|bad|terrible|awkward|weird)\s+fit\b|"
            r"\bfit(?:s|ted)?\s+(?:poorly|badly|terribly|awkwardly)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:see_through",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\b(?:see\s*through|see-through|too\s+sheer|transparent)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:shrunk_after_wash",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\b(?:shrank|shrunk|shrinked)\s+"
            r"(?:after|following)\s+(?:a\s+)?(?:wash|washing)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:color_faded",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\b(?:color|colour|colors|colours)\s+(?:faded|fades|fade)\b|"
            r"\b(?:faded|fades|fade)\s+(?:color|colour|colors|colours)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:fabric_feels_cheap",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\b(?:fabric|material)\s+(?:feel|feels|felt|feeling)\s+cheap\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:seam_ripped",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\bseam\s+(?:ripped|torn)\b|"
            r"\bstitching\s+(?:came\s+loose|undone|ripped|torn)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:zipper_broken",
        "polarity": "neg",
        "domain": "fashion",
        "pattern": re.compile(
            r"\bzipper\s+(?:broken|stuck|jammed|broke)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:true_to_size",
        "polarity": "pos",
        "domain": "fashion",
        "pattern": re.compile(
            r"\btrue\s+to\s+size\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:comfortable_fit",
        "polarity": "pos",
        "domain": "fashion",
        "pattern": re.compile(
            r"\bcomfortable\s+fit\b|"
            r"\bfit(?:s|ted)?\s+comfortably\b",
            flags=re.IGNORECASE
        )
    },

    # =========================================================================
    # Books phrases
    # =========================================================================
    {
        "rule_key": "phrase:missing_pages",
        "polarity": "neg",
        "domain": "books",
        "pattern": re.compile(
            r"\bmissing\s+pages?\b|"
            r"\bpages?\s+(?:is\s+|are\s+|was\s+|were\s+)?missing\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:poorly_written",
        "polarity": "neg",
        "domain": "books",
        "pattern": re.compile(
            r"\b(?:poorly|badly|terribly)\s+written\b|"
            r"\bwriting\s+(?:is\s+|was\s+)?(?:poor|bad|terrible|awful)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:hard_to_follow",
        "polarity": "neg",
        "domain": "books",
        "pattern": re.compile(
            r"\b(?:hard|difficult|confusing)\s+to\s+follow\b|"
            r"\bnot\s+easy\s+to\s+follow\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:bad_translation",
        "polarity": "neg",
        "domain": "books",
        "pattern": re.compile(
            r"\b(?:bad|poor|terrible|awful)\s+translation\b|"
            r"\btranslation\s+(?:is\s+|was\s+)?(?:bad|poor|terrible|awful)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:printing_error",
        "polarity": "neg",
        "domain": "books",
        "pattern": re.compile(
            r"\b(?:printing|print)\s+errors?\b|"
            r"\b(?:misprint|misprinted|misprints)\b|"
            r"\bpages?\s+(?:printed|print)\s+incorrectly\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:great_read",
        "polarity": "pos",
        "domain": "books",
        "pattern": re.compile(
            r"\bgreat\s+read\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:well_written",
        "polarity": "pos",
        "domain": "books",
        "pattern": re.compile(
            r"\bwell\s+written\b|"
            r"\bwriting\s+(?:is\s+|was\s+)?(?:excellent|great|clear)\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:easy_to_follow",
        "polarity": "pos",
        "domain": "books",
        "pattern": re.compile(
            r"\b(?:easy|clear)\s+to\s+follow\b|"
            r"\bclear\s+and\s+easy\s+to\s+follow\b",
            flags=re.IGNORECASE
        )
    },
    {
        "rule_key": "phrase:highly_informative",
        "polarity": "pos",
        "domain": "books",
        "pattern": re.compile(
            r"\b(?:highly|very|really)\s+informative\b|"
            r"\binformative\s+and\s+useful\b",
            flags=re.IGNORECASE
        )
    }
]

CONTRAST_WORDS = POST_CONTRAST_MARKERS.union(CONCESSIVE_STARTERS)

def normalise_for_meta_features(text):
    text = str(text)
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("`", "'")
            .lower()
    )

    for pattern, replacement in NEGATION_CONTRACTION_REPLACEMENTS.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


def count_word_set_matches(text, word_set):
    text = normalise_for_meta_features(text)

    tokens = re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        text
    )

    clean_word_set = {
        str(word).lower()
        for word in word_set
    }

    return sum(
        token in clean_word_set
        for token in tokens
    )


def find_non_overlapping_phrase_matches(text):
    text = normalise_for_meta_features(text)

    phrase_matches = []

    for rule in UNIVERSAL_PHRASE_RULES:
        rule_domain = str(rule.get("domain", "general")).lower()

        for match in rule["pattern"].finditer(text):
            phrase_matches.append({
                "rule_key": rule["rule_key"],
                "polarity": rule["polarity"],
                "domain": rule_domain,
                "start": match.start(),
                "end": match.end(),
                "length": match.end() - match.start()
            })

    phrase_matches = sorted(
        phrase_matches,
        key=lambda item: (
            item["start"],
            -item["length"]
        )
    )

    selected_matches = []
    occupied_ranges = []

    for phrase_match in phrase_matches:
        start = phrase_match["start"]
        end = phrase_match["end"]

        overlaps = False

        for occupied_start, occupied_end in occupied_ranges:
            if start < occupied_end and end > occupied_start:
                overlaps = True
                break

        if overlaps:
            continue

        selected_matches.append(phrase_match)
        occupied_ranges.append((start, end))

    return selected_matches


def extract_meta_polarity_features(text):
    negator_count = count_word_set_matches(text, NEGATION_WORDS)
    intensifier_count = count_word_set_matches(text, INTENSIFIERS)
    diminisher_count = count_word_set_matches(text, DIMINISHERS)
    contrast_count = count_word_set_matches(text, CONTRAST_WORDS)

    phrase_matches = find_non_overlapping_phrase_matches(
        text=text
    )

    neg_phrase_count = sum(
        match["polarity"] == "neg"
        for match in phrase_matches
    )

    neu_phrase_count = sum(
        match["polarity"] == "neu"
        for match in phrase_matches
    )

    pos_phrase_count = sum(
        match["polarity"] == "pos"
        for match in phrase_matches
    )

    return {
        "meta_negator_count": negator_count,
        "meta_intensifier_count": intensifier_count,
        "meta_diminisher_count": diminisher_count,
        "meta_contrast_count": contrast_count,

        "meta_neg_phrase_count": neg_phrase_count,
        "meta_neu_phrase_count": neu_phrase_count,
        "meta_pos_phrase_count": pos_phrase_count,

        "meta_has_negator": int(negator_count > 0),
        "meta_has_intensifier": int(intensifier_count > 0),
        "meta_has_diminisher": int(diminisher_count > 0),
        "meta_has_contrast": int(contrast_count > 0),

        "meta_has_neg_phrase": int(neg_phrase_count > 0),
        "meta_has_neu_phrase": int(neu_phrase_count > 0),
        "meta_has_pos_phrase": int(pos_phrase_count > 0),
    }


def create_meta_polarity_feature_df(split_df):
    rows = []

    split_df = split_df.reset_index(drop=True)

    for _, row in split_df.iterrows():
        rows.append(
            extract_meta_polarity_features(
                text=row["Text"]
            )
        )

    return pd.DataFrame(rows)


train_meta_polarity_features_df = create_meta_polarity_feature_df(train_df)
test_meta_polarity_features_df = create_meta_polarity_feature_df(test_df)

print("\n========== META POLARITY FEATURES ==========")
print("Train meta polarity feature shape:", train_meta_polarity_features_df.shape)
print("Test meta polarity feature shape:", test_meta_polarity_features_df.shape)

print("\nTrain meta polarity feature preview:")
print(train_meta_polarity_features_df.head(5))
print("\nTest meta polarity feature preview:")
print(test_meta_polarity_features_df.head(5))
# ----------------------------------------------------------------------------- END

# ----------------------------------------------------------------------------- START
# SAVE OUTPUTS
# -----------------------------------------------------------------------------
dataset_table = (balanced_df.groupby(["Domain", "Sentiment"]).size().unstack(fill_value=0))
dataset_table["Total"] = dataset_table.sum(axis=1)
dataset_table.loc["Total"] = dataset_table.sum(axis=0)

output_folder = "Dataset/Tables"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "balanced_dataset_table.png")

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")
table = ax.table(
    cellText=dataset_table.values,
    rowLabels=dataset_table.index,
    colLabels=dataset_table.columns,
    loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.show()

print("\nSaved Dataset Table to:", output_folder)

output_folder = "Dataset/Preprocessed"
os.makedirs(output_folder, exist_ok=True)

train_split_df = train_df[["Text", "Sentiment"]].reset_index(drop=True)
val_split_df = val_df[["Text", "Sentiment"]].reset_index(drop=True)
test_split_df = test_df[["Text", "Sentiment"]].reset_index(drop=True)

train_split_df.to_csv(
    os.path.join(output_folder, "train_split.csv"),
    index=False
)

val_split_df.to_csv(
    os.path.join(output_folder, "val_split.csv"),
    index=False
)

test_split_df.to_csv(
    os.path.join(output_folder, "test_split.csv"),
    index=False
)

print("Saved train/validation/test splits to:", output_folder)

meta_feature_folder = "Dataset/Meta_Features"
os.makedirs(meta_feature_folder, exist_ok=True)

train_meta_polarity_features_df.to_csv(
    os.path.join(meta_feature_folder, "train_meta_polarity_features.csv"),
    index=False
)

test_meta_polarity_features_df.to_csv(
    os.path.join(meta_feature_folder, "test_meta_polarity_features.csv"),
    index=False
)

print("Saved Meta Polarity Features to:", meta_feature_folder)
# ----------------------------------------------------------------------------- END
# ================================================================================================================== END
