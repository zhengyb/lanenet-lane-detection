# Process pitch_yaw_history.txt to get the pitch and yaw history

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt


def load_pitch_yaw_history(filename):
    pitch_yaw_history = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#Pitch_rad, Yaw_rad"):
                continue
            pitch_rad, yaw_rad = map(float, line.split(","))
            pitch_yaw_history.append([pitch_rad, yaw_rad])

    print("Load {} pitch_yaw_history from {}".format(len(pitch_yaw_history), filename))
    return np.array(pitch_yaw_history)

def convert_pitch_yaw_history_to_pitch_deg_yaw_deg(pitch_yaw_history):
    pitch_deg = pitch_yaw_history[:, 0] * 180 / np.pi
    yaw_deg = pitch_yaw_history[:, 1] * 180 / np.pi
    pitch_yaw_history_deg = np.column_stack((pitch_deg, yaw_deg))
    return pitch_yaw_history_deg

def display_pitch_yaw_history(pitch_yaw_history_deg, fig_name, color="r", marker="*"):
    # display the pitch and yaw history in a plot
    plt.figure(fig_name)
    # set range of x and y
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    # draw all the data as a dot
    plt.scatter(pitch_yaw_history_deg[:, 0], pitch_yaw_history_deg[:, 1], s=10, color=color, marker=marker)
    plt.xlabel("Pitch (deg)")
    plt.ylabel("Yaw (deg)")
    plt.title(fig_name)

    return plt
    # plt.show()
    plt.savefig(f"./output/{fig_name}.png")


def get_mean_pitch_yaw_history(pitch_yaw_history_deg):
    # get the mean of the pitch and yaw history
    mean_pitch = np.mean(pitch_yaw_history_deg[:, 0])
    mean_yaw = np.mean(pitch_yaw_history_deg[:, 1])
    return mean_pitch, mean_yaw

def test_pitch_yaw_history():
    pitch_yaw_history = load_pitch_yaw_history("./output/pitch_yaw_history_route28.txt")

    # remove the 2nd half of the pitch_yaw_history
    #pitch_yaw_history = pitch_yaw_history[:len(pitch_yaw_history)//2]

    pitch_yaw_history_deg = convert_pitch_yaw_history_to_pitch_deg_yaw_deg(pitch_yaw_history)
    mean_pitch, mean_yaw = get_mean_pitch_yaw_history(pitch_yaw_history_deg)
    pitch_std = np.std(pitch_yaw_history_deg[:, 0])
    yaw_std = np.std(pitch_yaw_history_deg[:, 1])
    print(f"S1: Mean pitch: {mean_pitch}, Mean yaw: {mean_yaw}, Pitch std: {pitch_std}, Yaw std: {yaw_std}")
    plt = display_pitch_yaw_history(pitch_yaw_history_deg, "pitch_yaw_history_mean")
    plt.scatter(mean_pitch, mean_yaw, s=100, color="g", marker="o")
    # draw a rectangle with radius of pitch_std and yaw_std
    plt.gca().add_artist(plt.Rectangle((mean_pitch - pitch_std, mean_yaw - yaw_std), 2 * pitch_std, 2 * yaw_std, color="b", fill=False))

    # filter the pitch_yaw_history_deg with the rectangle
    pitch_yaw_history_deg_filtered = pitch_yaw_history_deg[
        (pitch_yaw_history_deg[:, 0] > mean_pitch - pitch_std) &
        (pitch_yaw_history_deg[:, 0] < mean_pitch + pitch_std) &
        (pitch_yaw_history_deg[:, 1] > mean_yaw - yaw_std) &
        (pitch_yaw_history_deg[:, 1] < mean_yaw + yaw_std)
    ]
    print(f"pitch_yaw_history_deg_filtered: {len(pitch_yaw_history_deg_filtered)}")

    mean_pitch_filtered, mean_yaw_filtered = get_mean_pitch_yaw_history(pitch_yaw_history_deg_filtered)
    pitch_std_filtered = np.std(pitch_yaw_history_deg_filtered[:, 0])
    yaw_std_filtered = np.std(pitch_yaw_history_deg_filtered[:, 1])
    print(f"S2: Mean pitch: {mean_pitch_filtered}, Mean yaw: {mean_yaw_filtered}, Pitch std: {pitch_std_filtered}, Yaw std: {yaw_std_filtered}")

    # draw a dot on the mean of the pitch_yaw_history_deg
    plt.scatter(mean_pitch_filtered, mean_yaw_filtered, s=50, color="b", marker="o")

    # draw a rectangle with radius of pitch_std_filtered and yaw_std_filtered
    plt.gca().add_artist(plt.Rectangle((mean_pitch_filtered - pitch_std_filtered, mean_yaw_filtered - yaw_std_filtered), 2 * pitch_std_filtered, 2 * yaw_std_filtered, color="r", fill=False))

    # draw a circle with radius of pitch_std and yaw_std
    plt.savefig(f"./output/pitch_yaw_history_mean.png")



if __name__ == "__main__":
    test_pitch_yaw_history()



