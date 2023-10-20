from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from torchvision import transforms
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img #Loadimg lets you load images in a specific format from local files


model = wrn28_10_cifar10(pretrained=True)
model = SoftNBDT(
  pretrained=True,
  dataset='CIFAR10',
  arch='wrn28_10_cifar10',
  model=model,
  )

transforms = transforms.Compose([
  transforms.Resize(32),
  transforms.CenterCrop(32),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# img1 = load_img('Mentorship/images/husky.jpg', target_size=(224, 224))
# plt.imshow(img1)
# plt.show()

im = load_image_from_path("Mentorship/images/Husky.jpg")
x = transforms(im)[None]
# print(x)

outputs = model(x)  # to get intermediate decisions, use `model.forward_with_decisions(x)` and add `hierarchy='wordnet' to SoftNBDT
dt = DATASET_TO_CLASSES['CIFAR10']
order = [(str(dt[i]), float(list(outputs.detach().numpy()[0])[i])) for i in range(len(dt))]
print((x + ": " + y for (x, y) in order))
_, predicted = outputs.max(1)
cls = dt[predicted]
print(cls)
