# python cats_and_dogs_detector.py

from ultralytics import YOLO

input_dir = 'images'
output_dir = 'predictions'

# импортирование сохраненных весов модели
model = YOLO("cats_and_dogs_weights.onnx", task='detect')

# применение модели к папке input_dir
results = model.predict(source=input_dir)

import os, shutil
if os.path.exists(output_dir):
    
    # если папка output_dir существует, то очищаем ее
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

else:

    # если папки output_dir нет, то создаем
    os.mkdir(output_dir)



# сохранение результатов  в папку output_dir
for i, result in enumerate(results):
    result.save(filename=f"{output_dir}/{i}.jpg")