import numpy as np
import scipy.misc

# =============================================================================
# General image processing functions
# =============================================================================

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def get_transform_mat(center, scale, rot=0):
    # Generate transformation matrix
    c_pre = np.eye(3)
    c_pre[2,0] = -center[0]
    c_pre[2,1] = -center[1]
    c_post = np.eye(3)
    c_post[2,0] = center[0]*scale[0]
    c_post[2,1] = center[1]*scale[1]

    s_mat = np.eye(3)
    s_mat[0,0] = scale[0]
    s_mat[1,1] = scale[1]

    r_mat = np.eye(3)
    rot_rad = rot * np.pi / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    r_mat[0:2,0:2] = [[cs, -sn],[sn, cs]]
    t = np.dot(c_pre,np.dot(s_mat,np.dot(r_mat,c_post)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    return scipy.misc.imresize(new_img, res)

def crop_center(img, center, scale):
    height,width = img.shape


def inv_mat(mat):
    ans = np.linalg.pinv(np.array(mat).tolist() + [[0,0,1]])
    return ans[:2]

def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    b = np.ones(kpt.shape[0])
    kpt = np.column_stack((kpt, b))
    return np.dot(mat,kpt.T).T

def kpt_change(kpt, mat):
    kpt = np.array(kpt)
    b = np.ones(kpt.shape[0])
    kpt = np.column_stack((kpt, b))
    return np.dot(kpt,mat)[:,0:2]

def resize(im, res):
    import cv2
    return np.array([cv2.resize(im[i],res) for i in range(im.shape[0])])

if __name__ == '__main__':
    print(get_transform([10,10],0.8,[1,1],30))
    print(get_transform_mat([10,10],[0.8,0.8],30))