# GPU_test_pytorch
train picture classification model to compare the performance between cpu and gpu.

# Introduction
Pytorch GPU

单GPU加速
1. 确定GPU是否可用，torch.cuda.is_available()
2. 确定可用GPU数量，torch.cuda.device_count()，或通过nvidia-smi查看
3. 将数据从内存转到GPU，对模型和张量都是直接使用.to(device)或.cuda()就行了

单主机多GPU加速
1. 使用DataParallelDataParallel函数
2. 如果想使用所有的GPU，使用时直接用model传入torch.nn.DataParallel函数即可：
    1. net = torch.nn.DataParallel(model)
3. 如果只想使用部分的GPU，加入只想使用确定编号的GPU，可以用一下方法中的一种
    1. input_data = input_data.to(device = [1, 2, 3, 4]) net = torch.nn.DataParallel(model) net.to(device)
    2. os.environ[‘CUDA_VISIBLE_DEVICES’] = ‘,’.join(map(str, [0, 1, 2, 3])) net = torch.nn.DataParallel(model)
4. 单机多GPU也可以使用DistributedParallel，它多用于分布式训练，单也可以用在单机多GPU的训练，配置比使用nn.DataParallel稍微麻烦一点，但是训练速度和效果更好一点。配置方法如下：
    1. torch.distributed.init_process_group(backend = ‘nccl’) model = torch.nn.parallel.DistributedDataParallel(model)
    2. 单机运行时，使用下面方法启动： python -m torch.distributed.launch main.py

使用GPU注意事项
1. GPU数量尽量为偶数，基数的GPU有可能会出现异常中断的情况
2. GPU很快，单数据量较小时，效果可能没有单GPU好，甚至还不如CPU
3. 如果内存不够大，使用多GPU驯良的时候可通过设置pin_memory为False，当然使用进度稍微低一点的数据类型时也有效

## Test CPU performance
run picture classfication model with cpu, store the result at result_cpu.txt.<br>
- python codes/cpu_test.py

## Test GPU performance
run picture classfication model with gpu, store the result at result_gpu.txt.<br>
- python codes/gpu_test.py
