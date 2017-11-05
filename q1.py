import numpy as np
import matplotlib.pyplot as plt

def training_data():
  N = 25
  X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
  Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
  return N,X,Y

def generate_design_polynomial(order, X, Y, N):
  phi = np.zeros((N, order + 1))
  phi.fill(1)
  
  for i in range(N):
    for j in range(order + 1):
      phi[i][j] = X[i] ** j

  return phi

def phi_pol(order, x):
  return np.array([x**i for i in range(order + 1)])

def optimal_theta_vector(phi, y):
  step1 = np.linalg.inv(np.dot(phi.transpose(), phi))
  step2 = np.dot(step1, phi.transpose())
  return np.dot(step2, y)

def plot_by_order_pol(order, N, X, Y, lbl):
  phi = generate_design_polynomial(order, X, Y, N)
  theta = optimal_theta_vector(phi, Y)
  points_x = np.linspace(-0.3, 1.3, 200)
  points_y = [np.dot(phi_pol(order, x).transpose(), theta) for x in points_x]
  plt.plot(points_x, points_y, label=lbl)

def generate_plot_q1_i():
  N, X_training, Y_training = training_data()

  plot_by_order_pol(0, N, X_training, Y_training, "order=0")
  plot_by_order_pol(1, N, X_training, Y_training, "order=1")
  plot_by_order_pol(2, N, X_training, Y_training, "order=2")
  plot_by_order_pol(3, N, X_training, Y_training, "order=3")
  plot_by_order_pol(11, N, X_training, Y_training, "order=11")

  plt.legend(loc=3, borderaxespad=0.)
  plt.scatter(X_training, Y_training)
  plt.gca().set_ylim([-2,2])
  plt.show()
  return

