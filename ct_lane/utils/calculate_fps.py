import torch
from tqdm import tqdm
import time
def _2CUDA(batch: dict) -> dict:
	for b in batch:
		if isinstance(batch[b], torch.Tensor):
			batch[b] = batch[b].cuda()
	return batch

def validateFPS(net, logger, img_shape, iters=100):
	fake_fps_data = torch.zeros((1, 3, img_shape[0], img_shape[1])) + 1.0
	data = {'img': fake_fps_data}
	fake_fps_data = _2CUDA(data)
	# Benchmark latency and FPS
	net.eval()
	total_time = 0
	total_imgs = iters
	gpu_warmup_iters = 100
	loop = tqdm(range(iters+gpu_warmup_iters), desc='FPS')
	tqdm_dict = {'Status':""}
	for i in loop:
		if i < gpu_warmup_iters:
			tqdm_dict['Status'] = f"Warmup GPU: {gpu_warmup_iters-i}"
			net(fake_fps_data)
		else:
			start_time = time.time()
			net(fake_fps_data)
			total_time += (time.time() - start_time)
			tqdm_dict['Status'] = f"Caculate: {iters+gpu_warmup_iters-i}"
		loop.set_postfix(tqdm_dict)
	logger.record(
		f"Average latency (ms): {int(total_time * 1000 / total_imgs)}, "
		f"Average FPS: {int(total_imgs/total_time)}.")