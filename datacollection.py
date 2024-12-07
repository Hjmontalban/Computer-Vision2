import os
import cv2
import numpy as np

def create_image_dataset(root_folder, output_folder, img_size=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_index = 1

    for root, dirs, files in os.walk(root_folder):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            output_subfolder = os.path.join(output_folder, dir)

            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            count = 0
            for sub_root, _, sub_files in os.walk(folder_path):
                for file in sub_files:
                    if file.endswith(('jpg', 'jpeg', 'png')) and count < 500:
                        # Read and resize the image
                        file_path = os.path.join(sub_root, file)
                        img = cv2.imread(file_path)
                        img = cv2.resize(img, img_size)

                        # Save original image multiple times
                        for i in range(2):  # Save the original image twice
                            output_path_original = os.path.join(output_subfolder, f"{image_index}.jpg")
                            cv2.imwrite(output_path_original, img)
                            image_index += 1

                        count += 1

                        if count >= 500:
                            break
                if count >= 500:
                    break

    print(f'Images saved to {output_folder}')


if __name__ == "__main__":
    root_folder = "C:/Users/MSI`/Desktop/Version 2/Version 2/face Recog/training_images"  # Change to your root directory path
    output_folder = "C:/Users/MSI`/Desktop/Version 2/Version 2/face Recog/DATASETT2"  # Output folder for processed images
    create_image_dataset(root_folder, output_folder)
    print("Image dataset creation complete!")
