import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

from part3 import SFM


def visualize(prev_container, prev_frame_id, curr_container, curr_frame_id, focal, pp):
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = SFM.rotate(norm_prev_pts, R)
    rot_pts = [SFM.unnormalize(norm_rot_pts[i], focal, pp) for i in range(2)]
    foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))

    plt.imshow(curr_container.img)
    curr_p = [np.array(curr_container.traffic_light[i]) for i in range(2)]
    plt.plot(curr_p[0][:, 0], curr_p[0][:, 1], 'ro', markersize=4)
    plt.plot(curr_p[1][:, 0], curr_p[1][:, 1], 'go', markersize=4)

    for j in range(2):
        for i in range(len(curr_p[j])):
            plt.plot([curr_p[j][i, 0], foe[0]], [curr_p[j][i, 1], foe[1]], 'b')
            if curr_container.valid[j][i]:
                plt.text(curr_p[j][i, 0], curr_p[j][i, 1],
                         r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[j][i, 2]), color='w')
    plt.plot(foe[0], foe[1], 'y+')

class FrameContainer(object):
    def __init__(self, img_path):
        self.img = Image.open(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []
