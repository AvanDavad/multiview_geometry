import numpy as np

class Pose3D:
    def __init__(self, E_self2wld=np.eye(4)):
        assert E_self2wld.shape == (4, 4)
        assert np.all(E_self2wld[-1] == np.eye(4)[-1])
        self.E_self2wld = E_self2wld.astype(np.float64)

    def translate(self, t):
        assert t.shape == (3,)
        self.E_self2wld[:3, -1] = t

    def rotate_R(self, R, right=False):
        assert R.shape == (3, 3)
        if not right:
            self.E_self2wld[:3, :3] = R @ self.E_self2wld[:3, :3]
        else:
            self.E_self2wld[:3, :3] = self.E_self2wld[:3, :3] @ R

    def rotate_X(self, phi, degrees=True, local=False):
        """
        rotate about the X axis.
        phi: rotation angle
        degrees: boolean, rotation angle phi in radians or degrees
        local: if true, rotate about the local X axis, otherwise global.
        """
        s, c = Pose3D._get_sin_cos(phi, degrees)
        R = np.array([[1., 0., 0.],[0., c, -s],[0., s, c]])
        self.rotate_R(R, right=local)

    def rotate_Y(self, phi, degrees=True):
        """
        see rotate_X method.
        """
        s, c = Pose3D._get_sin_cos(phi, degrees)
        R = np.array([[c, 0., s],[0., 1., 0.],[-s, 0., c]])
        self.rotate_R(R, right=local)

    def rotate_Z(self, phi, degrees=True):
        """
        see rotate_X method.
        """
        s, c = Pose3D._get_sin_cos(phi, degrees)
        R = np.array([[c, -s, 0.],[s, c, 0.],[0., 0., 1.]])
        self.rotate_R(R, right=local)

    @staticmethod
    def _get_sin_cos(phi, degrees):
        if degrees:
            phi *= np.pi/180.
        return np.sin(phi), np.cos(phi)
