import numpy as np

# # # RV B # # #

index_4500_5000 = [0, 1, 2, 6, 7, 8, 11, 14, 15, 16, 17, 18, 19]
index_5000_5500 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
index_5500_6225 = [0, 1, 2, 6, 7, 8, 10, 11, 15, 16, 17, 18, 19]
index_6350_6825 = [0, 2, 5, 7, 8, 9, 12, 15, 16, 17, 18]

t_4500_5000, rv_4500_5000, _ = np.loadtxt('4500_5000_rvB.txt', unpack=True)
t_5000_5500, rv_5000_5500, _ = np.loadtxt('5000_5500_rvB.txt', unpack=True)
t_5500_6225, rv_5500_6225, _ = np.loadtxt('5500_6225_rvB.txt', unpack=True)
t_6350_6825, rv_6350_6825, _ = np.loadtxt('6350_6825_rvB.txt', unpack=True)

t_values = [[] for _ in range(len(rv_6350_6825))]
rv_values = [[] for _ in range(len(rv_6350_6825))]
for i in range(0, len(rv_5000_5500)):
    if i in index_4500_5000:
        t_values[i].append(t_4500_5000[i])
        rv_values[i].append(rv_4500_5000[i])
    if i in index_5000_5500:
        t_values[i].append(t_5000_5500[i])
        rv_values[i].append(rv_5000_5500[i])
    if i in index_5500_6225:
        t_values[i].append(t_5500_6225[i])
        rv_values[i].append(rv_5500_6225[i])
    if i in index_6350_6825:
        t_values[i].append(t_6350_6825[i])
        rv_values[i].append(rv_6350_6825[i])

time_list = []
rv_mean_list = []
err_list = []
index_list = []

for i in range(0, len(rv_values)):
    if len(rv_values[i]) >= 3:
        index_list.append(i)
        time_list.append(t_values[i][0])
        rv_mean_list.append(np.mean(rv_values[i]))
        err_list.append(np.std(rv_values[i])/np.sqrt(len(rv_values[i])))

for i in range(0, len(rv_mean_list)):
    print(rv_mean_list[i], '   ', err_list[i])

save_array = np.empty((len(index_list), 4))
save_array[:, 0] = np.array(index_list)
save_array[:, 1] = np.array(time_list)
save_array[:, 2] = np.array(rv_mean_list)
save_array[:, 3] = np.array(err_list)
np.savetxt('mean_std_B.txt', save_array, fmt=['%.0f', '%.10f', '%.6f', '%.5f'], delimiter='\t',
           header='Index Time       Mean        Error')

# # # RV A # # #
t, rv_4500_5000 = np.loadtxt('4500_5000_rvA.txt', unpack=True)
_, rv_5000_5500 = np.loadtxt('5000_5500_rvA.txt', unpack=True)
_, rv_5500_6225 = np.loadtxt('5500_6225_rvA.txt', unpack=True)
_, rv_6350_6825 = np.loadtxt('6350_6825_rvA.txt', unpack=True)

rv_mean_list = []
err_list = []
for i in range(0, rv_4500_5000.size):
    rv_mean_list.append(np.mean([rv_4500_5000[i], rv_5000_5500[i], rv_5500_6225[i], rv_6350_6825[i]]))
    err_list.append(np.std([rv_4500_5000[i], rv_5000_5500[i], rv_5500_6225[i], rv_6350_6825[i]])/np.sqrt(4))
save_array = np.empty((len(rv_mean_list), 3))
save_array[:, 0], save_array[:, 1], save_array[:, 2] = t, np.array(rv_mean_list), np.array(err_list)
np.savetxt('mean_std_A.txt', save_array, fmt='%.10f', delimiter='\t', header='Time  Mean    Error')

