import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.img_list[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# تعریف معماری شبکه Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# تعریف معماری شبکه Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# تعریف تابعی برای آموزش مدل GAN
def train_gan(device, generator, discriminator, dataloader, latent_dim, epochs, lr, b1, b2):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # انتخاب تابع هزینه و بهینه‌ساز
    adversarial_loss = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(epochs):
        for imgs in tqdm(dataloader):
            # تبدیل تصاویر به بردارهای تصادفی
            # در تابع train_gan
            imgs = imgs.view(imgs.size(0), -1).to(device)
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            # آموزش Discriminator بر داده‌های واقعی
            discriminator_optimizer.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), real_labels)
            real_loss.backward()

            # آموزش Discriminator بر داده‌های مصنوعی
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
            fake_loss.backward()

            discriminator_optimizer.step()

            # آموزش Generator
            generator_optimizer.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            gen_imgs = generator(z)
            gen_loss = adversarial_loss(discriminator(gen_imgs), real_labels)
            gen_loss.backward()
            generator_optimizer.step()


# تعریف تابع برای آماده‌سازی داده‌ها
def prepare_data(data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# تعریف پارامترهای مورد نیاز
latent_dim = 100
img_shape = (3, 256, 256)  # تصاویر را بر اساس اندازه دلخواه خود تعیین کنید
lr = 0.0002
b1 = 0.5
b2 = 0.999
batch_size = 64
epochs = 10

device = torch.device("cpu")
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)


# آماده‌سازی داده‌ها
# تعریف تبدیل‌ها برای تصاویر
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# تعریف دیتاست سفارشی
custom_dataset = CustomDataset(data_dir="/home/skyboy/w/python/facer-generator/celeba_hq_256", transform=transform)

# تعریف دیتالودر برای دیتاست سفارشی
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# آموزش مدل GAN
train_gan(device, generator, discriminator, dataloader, latent_dim, epochs, lr, b1, b2)

# ذخیره مدل آموزش دیده
torch.save(generator.state_dict(), "generator.pth")
