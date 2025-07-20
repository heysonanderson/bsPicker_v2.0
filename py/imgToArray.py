from PIL import Image
import numpy as np
import json

urlArray = [
    'dry_season',
    'hideout',
    'layer_cake',
    'shooting_star',
    'triple_dribble',
    'pinball_dreams',
    'sneaky_fields',
    'center_stage',
    'belles_rock',
    'flaring_phoenix',
    'new_horizons',
    'out_in_the_open',
    'below_zero',
    'cool_box',
    'starr_garden',
    'super_center',
    'parallel_plays',
    'dueling_beetles',
    'open_business',
    'ring_of_fire',
    'double_swoosh',
    'gem_fort',
    'hard_rock_mine',
    'undermine'
]

# Открываем изображения и преобразуем их в массивы
image_data = {}

for url in urlArray:
    try:
        # Открытие изображения
        image = Image.open(f"./data/maps/images/{url}.webp")
        gray_image = image.convert("L")

        # Изменяем размер изображения
        resized_image = gray_image.resize((35, 23))

        # Преобразуем изображение в массив numpy
        image_array = np.array(resized_image)

        # Добавляем массив в словарь
        image_data[url] = image_array.tolist()

    except Exception as e:
        print(f"Ошибка при обработке {url}: {e}")

# Сохраняем данные в JSON файл
with open("map_image.json", "w") as f:
    json.dump(image_data, f, indent=4)
