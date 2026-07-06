# 学习曲线绘制
from torch.utils.tensorboard import SummaryWriter
# 实验 1：存储在 logs/exp_A
writer_a = SummaryWriter('logs/exp_A')
# 实验 2：存储在 logs/exp_B
writer_b = SummaryWriter('logs/exp_B')
# tensorboard 的记录器
writer_a.add_scalar('a', 0.1, 1)
writer_a.add_scalar('a', 0.2, 2)
writer_a.add_scalar('a', 0.3, 3)
writer_a.add_scalar('a', 0.4, 4)
writer_a.add_scalar('a', 0.5, 5)
writer_a.add_scalar('a', 0.6, 6)
writer_a.add_scalar('a', 0.7, 7)
writer_b.add_scalar('a', 0.1, 1)
writer_b.add_scalar('a', 1.2, 2)
writer_b.add_scalar('a', 11.3, 3)
writer_b.add_scalar('a', 20.4, 4)
writer_b.add_scalar('a', 30.5, 5)
writer_b.add_scalar('a', 0.6, 6)
writer_b.add_scalar('a', 0.7, 7)

writer_a.close()
writer_b.close()