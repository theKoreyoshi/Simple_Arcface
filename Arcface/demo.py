from model import Backbone
import torch
from PIL import Image
from mtcnn import MTCNN

from torchvision.transforms import Compose, ToTensor, Normalize

mtcnn = MTCNN()


def get_img(img_path, device):
    img = Image.open(img_path)
    face = mtcnn.align(img)
    transfroms = Compose(
        [ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transfroms(face).to(device).unsqueeze(0)


device = 'cuda'
img1 = get_img('data/1.png', device)
img2 = get_img('data/2.png', device)
# img3 = get_img('data/3.png', device)
# img4 = get_img('data/4.png', device)

print(img1.shape)    # torch.Size([1, 3, 112, 112])

model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
model.load_state_dict(torch.load('model_ir_se50.pth'))
model.eval()
model.to(device)
print("Successfully loaded arcface model")

emb1 = model(img1)[0]
emb2 = model(img2)[0]
# emb3 = model(img3)[0]
# emb4 = model(img4)[0]
print(emb1.shape)        # torch.Size([512])

sim_12 = emb1.dot(emb2).item()
# sim_13 = emb1.dot(emb3).item()
# sim_24 = emb2.dot(emb4).item()
# sim_34 = emb3.dot(emb4).item()

print(sim_12)
# print(sim_13)
# print(sim_24)
# print(sim_34)
