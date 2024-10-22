from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertModel

# Ініціалізація Flask-додатку
app = Flask(__name__)

# Визначаємо клас моделі
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids=None):  # Додано token_type_ids
        outputs = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs[0][:, 0, :]  # Взяти перший токен [CLS] для кожного прикладу
        output_2 = self.l2(cls_output)
        output = self.l3(output_2)
        return output

# Завантажуємо модель та токенізатор
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERTClass()
model.load_state_dict(torch.load('model_balanced_aug_with_2_epoch.pth', map_location=device))  # Використовуємо load_state_dict
model.eval()  # Переводимо модель в режим оцінки
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Функція для токенізації тексту
def tokenize_data(text, tokenizer, max_length=64):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    return {
        'input_ids': inputs['input_ids'].to(device),
        'attention_mask': inputs['attention_mask'].to(device),
        'token_type_ids': inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids'])).to(device)  # Можливо, не буде token_type_ids
    }

# Головна сторінка (форма для введення коментаря)
@app.route('/')
def home():
    return render_template('index.html')

# Обробка форми та прогнозування
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']  # Отримуємо введений коментар
        inputs = tokenize_data(comment, tokenizer)
        
        # Прогнозування
        with torch.no_grad():
            outputs = model(**inputs)  # Передаємо аргументи
            predictions = torch.sigmoid(outputs)

        # Обробка результатів
        toxic_labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
        predicted_labels = (predictions > 0.5).float().squeeze().tolist()

        # Створюємо відповідь
        response = {label: 'Yes' if predicted_labels[i] > 0.5 else 'No' for i, label in enumerate(toxic_labels)}

        return render_template('result.html', comment=comment, result=response)

# Запуск Flask-додатку
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
