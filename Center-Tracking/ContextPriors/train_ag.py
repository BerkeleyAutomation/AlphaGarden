import paddle
from models.model_stages import CPNet
import paddleseg.transforms as T
from paddleseg.datasets import ADE20K
from paddle.optimizer.lr import PolynomialDecay
from paddleseg.models.losses import CrossEntropyLoss
from loss.affinityloss import AffinityLoss
from tool.train import train
from prepare_ag import AG_DS
import pickle
# backbone resnet预训练文件路径
backbonepath = None
print(backbonepath)
# 模型导入
model = CPNet(proir_size=60, am_kernel_size=11, groups=1, prior_channels=256, pretrained=backbonepath)

# 构建训练用的transforms
transform = [
    T.ResizeByLong(long_size = 750),
    T.ResizeStepScaling(0.5, 2.0, 0.25),
    T.RandomHorizontalFlip(),
    T.RandomPaddingCrop(crop_size=[750, 750]),
    T.RandomDistort(brightness_range=0.5,
                    contrast_range=0.5,
                    saturation_range=0.5),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

# 构建训练集
#train_dataset = ADE20K(
#    dataset_root='/app/ContextPrior_Paddle/ChallengeData',
#    transforms=transform,
#    mode='train'
# )


trains = None
with open('/app/Center-Tracking/ContextPriors/train.pkl','rb') as f:
    trains = pickle.load(f)
train_dataset = AG_DS(
    transforms=transform,
    dataset_root='/app/State-Estimation/out/cropped',
    mode = 'train',
    label_root='/app/State-Estimation/out/post_process',
    relevant_dates=trains
)
print(len(train_dataset))

# 构建验证用的transforms
transform_val = [
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]

# 构建验证集
#val_dataset = ADE20K(
#    dataset_root='/app/ContextPrior_Paddle/ChallengeData',
#    transforms=transform_val,
#    mode='val'
#)

vals = None
with open('/app/Center-Tracking/ContextPriors/val.pkl','rb') as f:
    vals = pickle.load(f)
val_dataset = AG_DS(
    transforms=transform,
    dataset_root='/app/State-Estimation/out/cropped',
    mode = 'val',
    label_root='/app/State-Estimation/out/post_process',
    relevant_dates=vals
)

# 设置学习率
base_lr = 0.02

lr = PolynomialDecay(
    learning_rate=base_lr,
    decay_steps=80000,
    power=0.9,
)

optimizer = paddle.optimizer.Momentum(lr,
                                      parameters=model.parameters(),
                                      momentum=0.9, )

losses = {}

losses['types'] = [
    CrossEntropyLoss(),
    CrossEntropyLoss(),
    AffinityLoss()
]
losses['coef'] = [1, 0.4, 1]

train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    val_scales=1,
    aug_eval=True,
    optimizer=optimizer,
    save_dir='output',
    iters=80000,
    batch_size=5,
    # resume_model='/home/aistudio/work/openContext/output/iter_129600', # checkpoint 文件
    resume_model=None,
    save_interval=400,
    log_iters=10,
    num_workers=0,  # 多线程
    losses=losses,
    use_vdl=True,
)
