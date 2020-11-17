# we use this function to merge the images in google colab
# merge edge detected images with original imgs
# result in images with 4 channel (RGB + Edge)
def mergeImages():
  import matplotlib.pyplot as plt
  import numpy as np
  edge_dir = "/content/drive/My Drive/M499/3"
  original_dir = "/content/drive/My Drive/M499/tranch3/tranch3"
  import os
  for ori in sorted(os.listdir(original_dir)):
    ori_img = plt.imread(os.path.join(original_dir,ori))
    file_name = ori.rsplit(".")[0]
    edge_img = None
    for edge in sorted(os.listdir(edge_dir)):
      edge_file_name = edge.rsplit(".")[0]
      if (edge_file_name == file_name):
        edge_img = plt.imread(os.path.join(edge_dir, edge))
        break
    if edge_img is not None:
      X = np.zeros(edge_img.shape + (4,))
      X[...,:3] = ori_img
      X[...,-1] = edge_img
      try:
        plt.imsave("/content/drive/My Drive/M499/merge3/" + file_name + ".png", X)
      except ValueError as e: # 0-255 as jpg, but should be 0-1
        X = np.zeros(edge_img.shape + (4,))
        X[...,:3] = ori_img / 255
        X[...,-1] = edge_img / 255
        plt.imsave("/content/drive/My Drive/M499/merge3/" + file_name + ".png", X)
    else:
      print(ori, "not found")
