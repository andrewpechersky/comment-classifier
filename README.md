# Сomment-classifier

Це веб-додаток на базі Flask, який використовує попередньо натреновану модель BERT для класифікації коментарів за різними категоріями токсичності. Модель була донавчена для шести категорій: Toxic, Severe Toxic, Obscene, Threat, Insult та Identity Hate  
Додаток упакований у Docker-контейнер, що полегшує розгортання та дозволяє користувачам швидко розпочати класифікацію коментарів на власних системах.

# Продуктивність Моделі
| Class          | Precision | Recall  | F1-Score | Support |
|----------------|-----------|---------|----------|---------|
| Toxic          | 0.98      | 0.93    | 0.96     | 3844    |
| Severe Toxic   | 0.69      | 0.70    | 0.70     | 648     |
| Obscene        | 0.92      | 0.91    | 0.92     | 2209    |
| Threat         | 0.87      | 0.78    | 0.82     | 218     |
| Insult         | 0.92      | 0.82    | 0.87     | 2253    |
| Identity Hate  | 0.90      | 0.73    | 0.81     | 654     |

# Встановлення та Використання
На вашому комп'ютері має бути встановлений Docker
# Завантаження Docker-образу та установка
`docker pull andrewpechersky007/comment-classifier:latest`  
`docker run -p 5000:5000 andrewpechersky007/comment-classifier`
# Доступ до Додатку
Після запуску контейнера, додаток буде доступний за адресою http://localhost:5000/
# Використання
Перейдіть на URL: http://localhost:5000/.  
Введіть коментар у відповідну форму.  
Надішліть коментар для отримання передбачених міток токсичності.  




