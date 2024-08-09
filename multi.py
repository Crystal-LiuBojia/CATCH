# import os
# from test import main
# import numpy as np
# import subprocess
#
# mac_f1_list = []
# mic_f1_list = []
# NMI_list = []
# ARI_list = []
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# seeds = [228, 7243, 3295, 4020, 8149, 4832, 741, 6538, 7850, 6391]
#
# for i in seeds:
#     p = subprocess.Popen("python main_0.py --seed {}".format(i), shell=True)
#     p.wait()
#     p.kill()
#     macro_f1_mean , micro_f1_mean, NMI, ARI = main()
#     mac_f1_list.append(macro_f1_mean)
#     mic_f1_list.append(micro_f1_mean)
#     NMI_list.append(NMI)
#     ARI_list.append(ARI)
# print('\n************************** Average results *******************************')
# print('Macro-F1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(*np.mean(np.array(mac_f1_list), axis=0).tolist()))
# print('Micro-F1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(*np.mean(np.array(mic_f1_list), axis=0).tolist()))
# print('NMI: {:.4f}, ARI: {:.4f}'.format(np.mean(NMI_list), np.mean(ARI_list)))
import os
import time
from test import main
import numpy as np
import subprocess

mac_f1_list = []
mic_f1_list = []
NMI_list = []
ARI_list = []
os.environ['MKL_THREADING_LAYER'] = 'GNU'
seeds = [228, 7243, 3295, 4020, 8149]#, 4832, 741, 6538, 7850, 6391]

log_dir_file = "./log/"  # os.path.join(args.log_dir, args.dataset, "gan/")
os.makedirs(log_dir_file, exist_ok=True)
cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
filename = str(log_dir_file + cur_time + "multi_log.txt")

for i in seeds:
    p = subprocess.Popen("python main.py --seed {}".format(i), shell=True)
    p.wait()
    p.kill()

    macro_f1_mean , micro_f1_mean, NMI, ARI = main()
    mac_f1_list.append(macro_f1_mean)
    mic_f1_list.append(micro_f1_mean)
    NMI_list.append(NMI)
    ARI_list.append(ARI)
    # filecontent = "Seed {%d} | Macro F1 {%f} | Micro F1 {%f} | NMI {%f} | ARI {%f}" % (i, macro_f1_mean, micro_f1_mean, NMI, ARI)
    # with open(filename, "a+") as f:
    #     f.write(filecontent + '\n')
    num_entries = len(macro_f1_mean)
    # 打开文件，并将内容写入
    with open(filename, "a+") as f:
        for j in range(num_entries):
            filecontent = "Seed {%d} | Macro F1 {%f} | Micro F1 {%f} | NMI {%f} | ARI {%f}" % (i, macro_f1_mean[j], micro_f1_mean[j], NMI, ARI)
            f.write(filecontent + '\n')
        f.write('\n')
print('\n************************** Average results *******************************')
print('Macro-F1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(*np.mean(np.array(mac_f1_list), axis=0).tolist()))
print('Micro-F1: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(*np.mean(np.array(mic_f1_list), axis=0).tolist()))
print('NMI: {:.4f}, ARI: {:.4f}'.format(np.mean(NMI_list), np.mean(ARI_list)))

# filecontent1 = "Macro-F1: {%f}, {%f}, {%f}, {%f}" % (np.mean(np.array(mac_f1_list), axis=0).tolist())
# filecontent2 = "Micro-F1: {%f}, {%f}, {%f}, {%f}" % (np.mean(np.array(mic_f1_list), axis=0).tolist())
# filecontent3 = "NMI: {%f}, ARI: {%f}" % (np.mean(NMI_list), np.mean(ARI_list))
#
# with open(filename, "a+") as f:
#     f.write(filecontent1 + '\n')
#     f.write(filecontent2 + '\n')
#     f.write(filecontent3 + '\n')
#     f.close()

mean_mac_f1 = np.mean(np.array(mac_f1_list), axis=0)
mean_mic_f1 = np.mean(np.array(mic_f1_list), axis=0)
mean_nmi = np.mean(NMI_list)
mean_ari = np.mean(ARI_list)

# 构建要写入文件的内容
filecontent1 = "Macro-F1: {%f}, {%f}, {%f}, {%f}" % tuple(mean_mac_f1)
filecontent2 = "Micro-F1: {%f}, {%f}, {%f}, {%f}" % tuple(mean_mic_f1)
filecontent3 = "NMI: {%f}, ARI: {%f}" % (mean_nmi, mean_ari)

# 打开文件，并将内容写入
with open(filename, "a+") as f:
    f.write(filecontent1 + '\n')
    f.write(filecontent2 + '\n')
    f.write(filecontent3 + '\n')
    # f.close() 不需要显式地调用，with语句会自动关闭文件
