import string
import math

# 1. FAQs dataset
faqs = [
    {"question": "What is your return policy?", "answer": "You can return products within 30 days of purchase."},
    {"question": "How can I track my order?", "answer": "You can track your order from the My Orders section in your account."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship worldwide with additional delivery charges."},
    {"question": "How can I contact customer support?", "answer": "You can reach our support team via email or our 24/7 chat service."}
]

# 2. Preprocess text: lowercase + remove punctuation
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

# 3. Convert text to word frequency dictionary
def word_freq(tokens):
    freq = {}
    for word in tokens:
        freq[word] = freq.get(word, 0) + 1
    return freq

# 4. Cosine similarity function
def cosine_sim(freq1, freq2):
    all_words = set(freq1.keys()) | set(freq2.keys())
    v1 = [freq1.get(word, 0) for word in all_words]
    v2 = [freq2.get(word, 0) for word in all_words]
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0

# 5. Chatbot matching
faq_freqs = [word_freq(preprocess(faq["question"])) for faq in faqs]

def chatbot_response(user_input):
    user_freq = word_freq(preprocess(user_input))
    similarities = [cosine_sim(user_freq, faq_freq) for faq_freq in faq_freqs]
    best_match = similarities.index(max(similarities))
    return faqs[best_match]["answer"]

# 6. Chat loop
print("FAQ Chatbot (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", chatbot_response(user_input))




