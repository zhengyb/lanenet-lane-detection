import cv2
import numpy as np

# data example:
# %YAML:1.0
# ---
# remap_ipm_x: !!opencv-matrix
#    rows: 640
#    cols: 640
#    dt: f
#    data: [ 5.62353492e-01, 2.81194115e+00, 5.05978918e+00,
#        7.30625963e+00, 9.55108261e+00, 1.17944412e+01, 1.40361576e+01,
#        ...
#    ]
# remap_ipm_y: !!opencv-matrix
#    rows: 640
#    cols: 640
#    dt: f
#    data: [ ... ]


def load_ipm_remap(_ipm_remap_file_path):
        fs = cv2.FileStorage(_ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

def save_ipm_remap(_ipm_remap_file_path, remap_matrices):
    fs = cv2.FileStorage(_ipm_remap_file_path, cv2.FILE_STORAGE_WRITE)

    # replace  ipm_x with a all-zero matrix
    remap_matrices['remap_to_ipm_x'] = np.zeros((640, 640)).astype(np.float32)

    fs.write('remap_ipm_x', remap_matrices['remap_to_ipm_x'])
    fs.write('remap_ipm_y', remap_matrices['remap_to_ipm_y'])

    fs.release()

def main():
    ipm_remap_file_path = "data/tusimple_ipm_remap.yml"
    remap_matrices = load_ipm_remap(ipm_remap_file_path)

    remap_to_ipm_x = remap_matrices['remap_to_ipm_x']

    print(remap_to_ipm_x.shape)
    print(remap_to_ipm_x)

    save_ipm_remap(ipm_remap_file_path+'_wr.yml', remap_matrices)

if __name__ == "__main__":
    main()