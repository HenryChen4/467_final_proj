from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight

from src.classifier.classifier_base import Classifier_Base

class Naive_Bayes_Classifier(Classifier_Base):
    def __init__(self, config):
        super().__init__(None, config)
        self.alpha = config.alpha
        
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = MultinomialNB(alpha=self.alpha)
    
    def train(self, train_loader):
        train_texts = [text[0] for text, _ in train_loader]
        text_train_labels = [label[0] for _, label in train_loader]

        X_train = self.vectorizer.fit_transform(train_texts)
        train_labels = self.label_encoder.fit_transform(text_train_labels)

        sample_weights = compute_sample_weight(class_weight='balanced', y=train_labels)

        # Training the model
        self.model.fit(X_train, train_labels, sample_weight=sample_weights)
        y_train_pred = self.model.predict(X_train)
        
        return y_train_pred

    def predict(self, test_loader):
        test_texts = [text[0] for text, label in test_loader]

        X_test = self.vectorizer.transform(test_texts)
        y_test_pred = self.model.predict(X_test)

        return y_test_pred
