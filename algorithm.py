import scipy
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('2.mat')

I_o = np.array(mat['illuminant']).squeeze()
print('I_o shape:', I_o.shape)

r_o = np.array(mat['reflectance'])
print('r_o shape:', r_o.shape)

d = 9.5 # unit in meter

c_i = np.array(mat['responseFunction'])
print('c_i shape:', c_i.shape)

g = 3.868 # this is the slope of the equation.
          # full equation: 3.871x-0.009966

macbethScene = mat['macbethScene']
spectrum = macbethScene['spectrum'][0][0]
wave = np.array(spectrum['wave'][0][0])
print('wave shape:', wave.shape)
m_true = mat['pixelValues']

print('pixelValues shape:', m_true.shape)
waterProp = mat['waterProp']
absorption = waterProp['absorption'][0][0].transpose()
print('absorption shape:', absorption.shape)
true_scatter = waterProp['scattering'][0][0]

d_lambda = wave[0, 1] - wave[0, 0] #10
wave_bandth = wave.shape[1]  # 31



pre_alpha = 100 * np.ones((wave_bandth, 1))
pre_scatter = np.ones((wave_bandth, 1))
cur_alpha = np.zeros((wave_bandth, 1))
cur_scatter = np.zeros((wave_bandth, 1))
diff = np.linalg.norm(pre_alpha - cur_alpha, 1) + np.linalg.norm(pre_scatter - cur_scatter, 1)
iteration = 0
value = 10
while diff > 0.01:
    iteration += 1
    print('iteration: ', iteration)
    alpha = cp.Variable((wave_bandth, 1))
    scatter = cp.Variable((wave_bandth, 1))

    alpha_center = cur_alpha.flatten()
    m_center = g * d_lambda * r_o.transpose() @ np.diag(I_o) @ np.diag(np.exp(-2 * d * alpha_center)) @ c_i
    d_m1 = -2 * d * g * d_lambda * r_o.transpose() @ np.diag(I_o) # (24, 31)
    dm = []
    for i in range(wave_bandth):
        v1 = d_m1[:, i, np.newaxis]
        v2 = c_i[np.newaxis, i, :]
        d_i = np.exp(alpha_center[i]) * v1 @ v2
        dm.append(d_i)
    dm = np.array(dm)
    dm = np.moveaxis(dm, 0, -1)  # (24, 3, 31)
    obj = m_center - dm @ alpha_center  # (24, 3)
    co_scatter = cp.transpose(cp.transpose(scatter) @ np.diag(I_o) @ c_i)  # (3, 1)
    scatter2 = []
    for i in range(24):
        scatter2.append(np.identity(3))
    scatter2 = np.array(scatter2).reshape(-1, 3)
    obj = obj.reshape(-1, 1) + g * d_lambda * scatter2 @ co_scatter
    dm = dm.reshape(-1, wave_bandth)  # (72, 31)
    obj += dm @ alpha
    m_true = m_true.reshape(-1, 1)

    objective = cp.norm(obj - m_true, 1)
    constraints = [alpha >= 0,
                  scatter >= 0]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    print("status:", prob.status)
    print("value", prob.value)
    value = prob.value
    pre_alpha = cur_alpha
    pre_scatter = cur_scatter
    cur_alpha = alpha.value
    cur_scatter = scatter.value
    diff = np.linalg.norm(pre_alpha - cur_alpha, 1) + np.linalg.norm(pre_scatter - cur_scatter, 1)


print('absorption error:', np.mean(np.abs(np.divide(cur_alpha - absorption, absorption))))
print('scattering error:', np.mean(np.abs(np.divide(cur_scatter - true_scatter, true_scatter))))
print('alpha:', cur_alpha.transpose())
print('scatter:', cur_scatter.transpose())

plt.figure()
plt.plot(absorption, label='absorption')
plt.plot(cur_alpha, label='alpha')
plt.xlabel('waveband')
plt.title('absorption')
plt.legend()
plt.show()
plt.figure()
plt.plot(true_scatter, label='scatter')
plt.plot(cur_scatter, label='scatterEstimate')
plt.xlabel('waveband')
plt.title('scattering')
plt.legend()
plt.show()

