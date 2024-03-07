import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import cv2
import numpy as np

class AdvancedGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
        
# تابع برای شناسایی ناحیه‌های سفید در تصویر
def find_white_mask(image_path):
    input_image = cv2.imread(image_path)
    #input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    white_mask = np.all(input_image == [255, 255, 255], axis=-1)
    return white_mask

def find_skin_mask(image_path):
    input_image = cv2.imread(image_path)
    # اضافه کردن یک فیلتر Canny برای تقویت لبه‌ها
    edges = cv2.Canny(input_image, 100, 200)
    # حذف نویز با استفاده از فیلتر گوسیان
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    # تبدیل تصویر به فضای رنگ HSV
    input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    # تعیین محدوده رنگی پوست (با توجه به مقادیر HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # اعمال ماسک رنگی
    skin_mask = cv2.inRange(input_image_hsv, lower_skin, upper_skin)
    # ادغام ماسک پوست با ماسک لبه‌ها
    final_skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=edges)
    return final_skin_mask

# تابع برای تصحیح رنگ تصویر تولید شده با توجه به ناحیه پوستی در تصویر اصلی
def correct_generated_color(input_image, generated_image, skin_mask):
    # تبدیل تصویر تولیدی به آرایه NumPy و اندازه‌گیری آن با اندازه تصویر ورودی
    generated_np = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0
    generated_np = generated_np[:input_image.shape[0], :input_image.shape[1], :]
    
    # تصحیح رنگ تصویر تولید شده با توجه به ناحیه پوستی
    output_image = np.copy(generated_np)
    output_image[skin_mask] = input_image[skin_mask]

    return output_image

# تابع برای جایگزین کردن نواحی سفید با تصویر تولید شده توسط مدل GAN
def replace_white_regions(input_image_path, output_image_path, generator, latent_dim):
    device = torch.device("cpu")
    input_image = cv2.imread(input_image_path)

    # شناسایی نواحی پوست
    skin_mask = find_white_mask(input_image_path)
    print(skin_mask)
    
    # تولید تصویر جدید با استفاده از مدل GAN
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated_image = generator(z)

    generated_np = (generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0
    #generated_np = generated_np[:input_image.shape[0], :input_image.shape[1], :]
    generated_np = cv2.cvtColor(generated_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite('sss.jpg', generated_np)
    # تصحیح رنگ تصویر تولید شده با توجه به ناحیه پوستی
    #corrected_generated_image = correct_generated_color(input_image, generated_image, skin_mask)
    
    # جایگزین کردن نواحی سفید با تصویر تولید شده
    output_image = np.copy(input_image)
    shift_amount = 10

    # ابعاد تصویر
    height, width, _ = input_image.shape

    # حلقه بر روی هر سطر تصویر
    for y in range(height - shift_amount):
    # جایگزینی سطر i با سطر i + shift_amount
        output_image[y, skin_mask[y]] = generated_np[y + shift_amount, skin_mask[y]]

    # نمایش و ذخیره تصویر تغییر یافته
    cv2.imwrite(output_image_path, output_image)

# Paths to input and output images
base_dir = os.getcwd()
input_image_path = base_dir + '/input/image.png'
output_image_path = base_dir + '/output/completed_image.jpg'

latent_dim = 100
img_shape = (3, 256, 256)

# Load the Generator model
generator = AdvancedGenerator(latent_dim, img_shape)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# ذخیره تصاویر تولید شده
os.makedirs("output", exist_ok=True)

for name in os.listdir(base_dir + '/input'):
    input_image_path = base_dir + f'/input/{name}'
    output_image_path = base_dir + f'/output/completed_{name}'
    # Complete white areas in the image
    replace_white_regions(input_image_path, output_image_path, generator, latent_dim)
