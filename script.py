import gc
import csv
import cv2
from tqdm import tqdm
from easyocr import Reader
from ultralytics import YOLO
import re
import numpy as np
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoModelForSequenceClassification


font_path = 'arial.ttf'
font_size = 50
font = ImageFont.truetype(font_path, font_size)

# Specify the text and color
text_color = (0, 0, 0)  # Black color


categories = {0: 'Макаронные изделия',
 1: 'Ячневая крупа',
 2: 'Горох',
 3: 'Овсяная крупа',
 4: 'Свекла',
 5: 'Рис',
 6: 'Хлеб',
 7: 'Кукурузная крупа',
 8: 'Свинина',
 9: 'Печенье',
 10: 'Масло',
 11: 'Рыба',
 12: 'Творог',
 13: 'Молоко пастеризованное',
 14: 'Манная крупа',
 15: 'Гречневая крупа',
 16: 'Лимон',
 17: 'Пшено',
 18: 'Сыр',
 19: 'Нектар',
 20: 'Кефир',
 21: 'Пшеничная крупа',
 22: 'Сливки',
 23: 'Яйца',
 24: 'Картофель',
 25: 'Бананы',
 26: 'Творожок',
 27: 'Полба',
 28: 'Перловая крупа',
 29: 'Вода',
 30: 'Мука',
 31: 'Индейка',
 32: 'Яблоки',
 33: 'Курица',
 34: 'Огурцы',
 35: 'Грейпфрут',
 36: 'Сок',
 37: 'Овсяные хлопья',
 38: 'Сало',
 39: 'Капуста',
 40: 'Киноа',
 41: 'Йогурт',
 42: 'Сметана',
 43: 'Кукуруза',
 44: 'Перец',
 45: 'Лук',
 46: 'Маргарин',
 47: 'Маш',
 48: 'Пепси',
 49: 'Фасоль',
 50: 'Ряженка',
 51: 'Нут',
 52: 'Сгущенное молоко',
 53: 'Кускус',
 54: 'Чечевица',
 55: 'Молоко стерилизованное',
 56: 'Булгур'}

folder_path = 'ds/train_dataset_dnr-train/test'

# Получение списка файлов в папке
file_list = os.listdir(folder_path)

# Фильтрация только файлов с расширением .jpg, .png, .jpeg (или любым другим поддерживаемым форматом)
image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Удаление расширений из названий файлов
image_names_no_extension = [os.path.splitext(f)[0] for f in image_files]

# Путь к CSV-файлу
csv_file_path = 'ds/train_dataset_dnr-train/test.csv'

# Открытие CSV-файла для записи
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Запись заголовка
    writer.writerow(['Наименование'])
    # Запись названий файлов без расширения
    for filename in image_names_no_extension:
        writer.writerow([filename])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'DeepPavlov/rubert-base-cased')


model = YOLO('runs/cennic_detector.pt')
config = AutoConfig.from_pretrained('runs/BERT_model2')
bert = AutoModelForSequenceClassification.from_pretrained('runs/BERT_model2', config=config)
rublkop = YOLO('runs/prise_detector.pt')
file = ''
reader = Reader(['ru'])

def replace_multiple_spaces(text):
    # Заменяем несколько подряд идущих пробелов на один пробел
    return re.sub(' +', ' ', text)
# Функция для фильтрации текста
def filter_text(text):
    # Удаление цифр и знаков пунктуации
    filtered_text = re.sub(r'\d|\W+', ' ', text)
    # Удаление лишних пробелов
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
    #filtered_text = list(filter(None, filtered_text))
    return "".join(filtered_text) #filtered_text
def expand_box(box, image, percentage=5):
    # Распаковка координат бокса
    x1, y1, x2, y2 = box

    # Вычисление центра бокса
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Вычисление новых размеров бокса с учетом процента
    width = x2 - x1
    height = y2 - y1
    new_width = int(width * (1 + percentage / 100))
    new_height = int(height * (1 + percentage / 100))

    # Вычисление новых координат бокса
    new_x1 = center_x - new_width // 2
    new_y1 = center_y - new_height // 2
    new_x2 = center_x + new_width // 2
    new_y2 = center_y + new_height // 2

    # Проверка границ изображения
    new_x1 = int(max(0, new_x1))
    new_y1 = int(max(0, new_y1))
    new_x2 = int(min(image.shape[1], new_x2))
    new_y2 = int(min(image.shape[0], new_y2))

    # Вырезать бокс из изображения
    cropped_image = image[new_y1:new_y2, new_x1:new_x2]
    return cropped_image

# Открытие CSV-файла
with open('ds/train_dataset_dnr-train/test.csv', 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    next(csvreader)  # Пропуск заголовка
    # Подсчет количества строк в CSV-файле
    row_count = sum(1 for _ in csvreader)
    csvfile.seek(0)  # Возврат к началу файла
    next(csvreader)  # Пропуск заголовка

    # Создание CSV-файла для результатов
    with open('MYDS.csv', 'w', newline='', encoding='utf-8') as outfile:
        csvwriter = csv.writer(outfile, delimiter=';')
        csvwriter.writerow(['Наименование файла','Категория продукта', 'Цена'])

        # Инициализация tqdm #-- прогресс бар
        images = []
        data = []
        for row in csvreader:
            filename,  = row
            data.append(row)
            image_path = f'ds/train_dataset_dnr-train/test/{filename}.jpg'

            # Открытие изображения
            image = cv2.imread(image_path)

            # print(image.shape)

            images.append(image)

            gc.collect()

        # Получение боксов с помощью модели
        results = model(images, conf=0.8)
        pbar = tqdm(total=row_count, desc="Processing images")

        for result, image, row in zip(results, images, data):

            # Для каждого бокса
            discr = []
            result_price = [['0'],['0']]
            filename,  = row

            for bbox in result.boxes.xyxy.cpu():
                x1, y1, x2, y2 = bbox.tolist()

                gc.collect()

                # Вырезание картинки из изображения
                cropped_image = expand_box(bbox, image, 0)
                res_rublkop = rublkop(cropped_image, conf=0.7)  # детекция областей цены на ценнике
                a = Image.fromarray(res_rublkop[0].plot())
                ad = ImageDraw.Draw(a)
                boxes = np.array(res_rublkop[0].boxes.xyxy.cpu(), dtype=int)
                classes = np.array(res_rublkop[0].boxes.cls.cpu(), dtype=int)
                for i in range(len(boxes)):
                    # print(np.array(res_rublkop[0].boxes.xyxy.cpu()))

                    try:
                        ppp = reader.readtext(expand_box(boxes[i], cropped_image, 10), detail=0, text_threshold=0.8, allowlist='0123456789')
                    except:
                        print(filename)
                        continue
                    if ppp!=[]:
                        result_price[classes[i]] = ppp

                # Распознавание текста с помощью easyOCR
                try:
                    ocr_results = reader.readtext(cropped_image, detail=0, text_threshold=0.01)
                except:
                    print(filename)
                    continue
                # Фильтрация текста
                filtered_results = [filter_text(text) for text in ocr_results]
                filtered_results = "".join(filtered_results)
                # BERT
                # ВСЕ НЕЙРОНКИ В ПАПКУ СОРС?
                # А ТРЕЙН В ??
                tokens = tokenizer.encode(filtered_results, add_special_tokens=True)
                tokens_tensor = torch.tensor([tokens])
                with torch.no_grad():
                    logits = bert(tokens_tensor)
                logits = logits[0].detach().numpy()
                predicted_class = np.argmax(logits, axis=1)
                # predicted_class = category_index_reverse[predicted_class[0]]
                # КАНЭЦ
                # Запись результата в CSV-файл
                discr = discr + list(filtered_results)

                gc.collect()
            if discr != [] and result_price!=[['0'],['0']]and result_price!=[['00'],['0']]:
                j = [str("".join(x)) for x in result_price]
                price = ".".join(j)
                ad.text((50, 50),price,fill=text_color, font=font)
                ad.text((50, 100),categories[predicted_class[0]],fill=text_color, font=font)
                a.save(f'out_imgs/{filename}.jpg')

                csvwriter.writerow([filename, categories[predicted_class[0]], replace_multiple_spaces(price)])
            gc.collect()  # гарбаж коллектор
            pbar.update(1)
        pbar.close()
# очистка gpu
torch.cuda.empty_cache()
