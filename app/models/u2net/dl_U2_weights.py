import os
import gdown

os.makedirs('app/models/u2net/u2netp.pth', exist_ok=True)
os.makedirs('app/models/u2net/u2net_portrait.pth', exist_ok=True)

gdown.download('https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
    'app/models/u2net/u2netp.pth',
    quiet=False)

gdown.download('https://drive.google.com/uc?id=1IG3HdpcRiDoWNookbncQjeaPN28t90yW',
    'app/models/u2net/u2net_portrait.pth',
    quiet=False)
