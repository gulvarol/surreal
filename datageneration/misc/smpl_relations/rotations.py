import numpy as np
import math


def rotvec2rotmat(rotvec):
    # computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)


def euler2rotmat(e):
    Rx = np.array(((1, 0, 0),
                   (0, math.cos(e[0]), -math.sin(e[0])),
                   (0, math.sin(e[0]), math.cos(e[0]))))
    Ry = np.array(((math.cos(e[1]), 0, math.sin(e[1])),
                   (0, 1, 0),
                   (-math.sin(e[1]), 0, math.cos(e[1]))))
    Rz = np.array(((math.cos(e[2]), -math.sin(e[2]), 0),
                   (math.sin(e[2]), math.cos(e[2]), 0),
                   (0, 0, 1)))
    return np.dot(Rz, np.dot(Ry, Rx))


def rotmat2euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    rex = math.atan2(R[2, 1], R[2, 2])
    rey = math.atan2(-R[2, 0], sy)
    rez = math.atan2(R[1, 0], R[0, 0])  # euler
    return np.array((rex, rey, rez))


def normalize(v):
    v_mag = np.linalg.norm(v)
    if v_mag == 0:
        v = np.zeros(3)
        v[0] = 1
    else:
        v = v / v_mag
    return v


def rotmat2rotvec(R):
    # Convert rotation matrix representation to axis-angle (rotation vector) representation
    # R: Rotation matrix, u: unit vector, theta: angle
    # https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis.E2.80.93angle
    u = np.zeros(3)
    x = 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1)  # 1.0000000484288
    x = max(x, -1)
    x = min(x, 1)
    theta = math.acos(x)  # Tr(R) = 1 + 2 cos(theta)

    if(theta < 1e-4):  # avoid division by zero!
        print('theta ~= 0 %f' % theta)
        return u
    elif(abs(theta - math.pi) < 1e-4):
        print('theta ~= pi %f' % theta)
        if (R[0][0] >= R[2][2]):
            if (R[1][1] >= R[2][2]):
                u[0] = R[0][0] + 1
                u[1] = R[1][0]
                u[2] = R[2][0]
            else:
                u[0] = R[0][1]
                u[1] = R[1][1] + 1
                u[2] = R[2][1]
        else:
            u[0] = R[0][2]
            u[1] = R[1][2]
            u[2] = R[2][2] + 1

        u = normalize(u)
    else:
        d = 1 / (2 * math.sin(theta))  # ||u|| = 2sin(theta)
        u[0] = d * (R[2, 1] - R[1, 2])
        u[1] = d * (R[0, 2] - R[2, 0])
        u[2] = d * (R[1, 0] - R[0, 1])
    return u * theta


def makeRotationMatrix(R):
    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)

    # remove reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(U, Vt)
    return R


def euler2rotvec(e):
    R = euler2rotmat(e)
    return rotmat2rotvec(R)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def axisangle(v1, v2):
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    angle = angle_between(v1, v2)
    axis = normalize(np.cross(v1_u, v2_u))
    return axis * angle
