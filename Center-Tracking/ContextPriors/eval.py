import paddle
from paddleseg.core import evaluate
from models.model_stages import CPNet
from paddleseg.datasets import ADE20K
import paddleseg.transforms as T

model = CPNet(proir_size=60, am_kernel_size=11, groups=1, prior_channels=256)
# 加载预训练模型
model.set_state_dict(paddle.load('/app/ContextPrior_Paddle/output/best_model/model.pdparams'))

# 构建验证用的transforms
transform_val = [
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]

# 构建验证集
val_dataset = ADE20K(
    dataset_root='/app/ContextPrior_Paddle/ChallengeData',
    transforms=transform_val,
    mode='val'
)

# 验证
mean_iou, acc, _, _, _ = evaluate(
    model,
    val_dataset,
    aug_eval=True,  # 使用多尺度
    scales=[1.0, 1.5, 1.75],
    num_workers=0)
