import numpy as np
import matplotlib.pyplot as plt

def training_data():
  N = 25
  X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
  Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
  return N,X,Y

def generate_design_trig(order, X, N):
  phi = np.zeros((N, 2*order + 1))
  phi.fill(1)

  for i in range(N):
    phi[i] = phi_trig(order, X[i])
  
  return phi 

def generate_design_polynomial(order, X, N):
  phi = np.zeros((N, order + 1))
  phi.fill(1)

  for i in range(N):
    for j in range(order + 1):
      phi[i][j] = X[i] ** j

  return phi

def phi_pol(order, x):
  return np.array([x**i for i in range(order + 1)])

def phi_trig(order, x):
  phi = np.zeros(2*order + 1);
  phi[0] = 1
  for i in xrange(1, order + 1):
    phi[2 * i - 1] = np.sin(2*np.pi*x*i)
  for i in xrange(1, order + 1):
    phi[2* i] = np.cos(2*np.pi*x*i)
  return phi

def optimal_theta_vector(phi, y):
  step1 = np.linalg.inv(np.dot(phi.transpose(), phi))
  step2 = np.dot(step1, phi.transpose())
  return np.dot(step2, y)

def plot_by_order_trig(order, N, X, Y, lbl):
  phi = generate_design_trig(order, X, N)
  theta = optimal_theta_vector(phi, Y)
  points_x = np.linspace(-1, 1.2, 200)
  points_y = [np.dot(phi_trig(order, x).transpose(), theta) for x in points_x]
  plt.plot(points_x, points_y, label=lbl)

def plot_by_order_pol(order, N, X, Y, lbl):
  phi = generate_design_polynomial(order, X, N)
  theta = optimal_theta_vector(phi, Y)
  points_x = np.linspace(-0.3, 1.3, 200)
  points_y = [np.dot(phi_pol(order, x).transpose(), theta) for x in points_x]
  plt.plot(points_x, points_y, label=lbl)

def trig_param(order, X, Y, N):
  phi = generate_design_trig(order, X, N)
  theta = optimal_theta_vector(order, Y)
  return phi, theta

def dot_prod(x):
  np.dot(x.transpose(), x)

def rmse(y, phi, theta, N):
  return np.sqrt(dot_prod(y - np.dot(phi, theta)) / N

##############################################################
# Graph generation

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

def generate_plot_q1_ii():
  N, X_training, Y_training = training_data()
  
  plot_by_order_trig(1, N, X_training, Y_training, "order=1")
  plot_by_order_trig(11, N, X_training, Y_training, "order=11")

  plt.legend(loc=3, borderaxespad=0.)
  plt.scatter(X_training, Y_training)
  plt.gca().set_ylim([-2,2])
  plt.show()
  return

def generate_plot_q1_iii():
  N, X, Y = training_data()

  # Leave-one-out cross validation.
  order_x = [0..10]
  test_y = []

  for i in order_x:
    rmse_c = 0.0
    for j in range(N):
      phi, theta = trig_param(order, X, Y, N)
      rmse_c += rmse(y[

    test_y.append(rmse_c)




