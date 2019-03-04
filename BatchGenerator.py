import cv2
import numpy as np
import matplotlib.image as mpimg

class BatchGenerator:
    def __init__(self, image_width, image_height, image_channels):
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels


    def read_image(self, image):
        return mpimg.imread(image)


    def choose_image(self, center, left, right, steering_angle):
        choice = np.random.choice(3)
        if choice == 0:
            return self.read_image(left), steering_angle + 0.2
        elif choice == 1:
            return self.read_image(right), steering_angle - 0.2
        return self.read_image(center), steering_angle


    def random_flip(self, image, steering_angle):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        
        return image, steering_angle


    def random_translate(self, image, steering_angle, range_x, range_y):
        transform_x = range_x * (np.random.rand() - 0.5)
        transform_y = range_y * (np.random.rand() - 0.5)
        steering_angle += transform_x * 0.002
        transform_m = np.float32([[1, 0, transform_x], [0, 1, transform_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, transform_m, (width, height))
        return image, steering_angle


    def augment(self, center, left, right, steering_angle, range_x=100, range_y=10):
        image, steering_angle = self.choose_image(center, left, right, steering_angle)
        image, steering_angle = self.random_flip(image, steering_angle)
        image, steering_angle = self.random_translate(image, steering_angle, range_x, range_y)

        return image, steering_angle


    def resize(self, image):
        return cv2.resize(image, (self.image_width, self.image_height), cv2.INTER_AREA)


    def rgb2yuv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


    def preprocess(self, image):
        image = self.resize(image)
        image = self.rgb2yuv(image)
        return image


    def generate_batch(self, image_paths, steering_angles, batch_size, is_training):
        images = np.empty([batch_size, self.image_height, self.image_width, self.image_channels])
        new_steering_angles = np.empty(batch_size)

        while True:
            index_in_batch = 0
            for image_index in np.random.permutation(image_paths.shape[0]):
                center, left, right = image_paths[image_index]
                steering_angle = steering_angles[image_index]

                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = self.augment(center, left, right, steering_angle)
                else:
                    image = self.read_image(center)
                
                images[index_in_batch] = self.preprocess(image)
                new_steering_angles[index_in_batch] = steering_angle

                index_in_batch += 1
                if index_in_batch == batch_size:
                    break

            yield images, new_steering_angles