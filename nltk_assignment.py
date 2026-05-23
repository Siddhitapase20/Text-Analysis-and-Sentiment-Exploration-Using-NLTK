import nltk

nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

from nltk.corpus import gutenberg   
text= gutenberg.raw('shakespeare-hamlet.txt')
print(text[:1000])  

#sentence tokenization
from nltk.tokenize import sent_tokenize
sentences=sent_tokenize(text)
print(sentences[:5])

# word tokenization
from nltk.tokenize import word_tokenize
words= word_tokenize(text)
print(words[:10])

#covert to lowercase
words=[word.lower() for word in words]
print("covert to lowercase:",words[:10])

# remove punctuation
words=[word for word in words if word.isalnum()]
print("remove punctuation:",words[:10])

# remove stopwords
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
filtered_words= [word for word in words if word not in stop_words]
print("remove stopwords:",filtered_words[:20])

# lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatized_words=[lemmatizer.lemmatize(word) for word in filtered_words]
print("lemmatization:",lemmatized_words[:20])

#frequency distribution
from nltk import FreqDist
freq_dist=FreqDist(lemmatized_words)
print("frequency distribution:",freq_dist.most_common(20))

#bar plot
import matplotlib.pyplot as plt
freq_dist.plot(20, cumulative=False)
plt.savefig("frequency_plot.png")
plt.show()

#pos tagging
pos_tags=nltk.pos_tag(lemmatized_words)
print("pos tagging:",pos_tags[:20])

# most frequent nouns
nouns=[word for word , pos in pos_tags if pos.startswith('NN')]
noun_freq=FreqDist(nouns)
print("most frequent nouns:",noun_freq.most_common(20))

#Named Entity Recognition (NER)
chunked = nltk.ne_chunk(pos_tags)
entities = []
for chunk in chunked:
    if hasattr(chunk, 'label'):
        entity = " ".join(c[0] for c in chunk)
        entities.append(entity)

entity_freq = FreqDist(entities)
print(entity_freq.most_common(10))


#Sentiment Analysis
from nltk.sentiment import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()

#Analyze Sentiment
positive = 0
negative = 0
neutral = 0

for sentence in sentences:
    
    score = sia.polarity_scores(sentence)

    if score['compound'] > 0:
        positive += 1
        
    elif score['compound'] < 0:
        negative += 1
        
    else:
        neutral += 1

print("Positive:", positive)
print("Negative:", negative)
print("Neutral:", neutral)

# pie chart
labels = ['Positive', 'Negative', 'Neutral']

sizes = [positive, negative, neutral]

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Sentiment Distribution")
plt.savefig("sentiment_chart.png")
plt.show()