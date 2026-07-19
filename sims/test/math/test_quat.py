import sys
import numpy as np

sys.path.append(".")

import csim.math.quaternion as Q
from csim.math.quaternion import DEG_TO_RAD


def test_conjugate():
    q = Q.conj(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(q, [1, 0, 0, 0])

    q = Q.conj(np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(q, [1, -2, -3, -4])


def test_unit():
    q = Q.unit(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(q, [1, 0, 0, 0])

    q = Q.unit(np.array([2.0, 2.0, 2.0, 2.0]))
    assert np.array_equal(q, [0.5, 0.5, 0.5, 0.5])


def test_mul():
    q1 = [2, 2, 2, 2]
    q2 = Q.mul(q1, q1)
    assert np.array_equal(q2, [-8, 8, 8, 8])

    q1 = [3, 1, -2, 1]
    q2 = [2, -1, 2, 3]
    q2 = Q.mul(q1, q2)
    assert np.array_equal(q2, [8, -9, -2, 11])


def test_hamilton_product():
    q = np.array([2.0, 2.0, 2.0, 2.0])
    w = np.array([1.0, 2.0, 3.0])
    result = Q.hamilton_product(q, w)
    assert np.array_equal(result, [-12.0, 4.0, 0, 8.0])

    q = np.array([100.0, 1.0, 4.0, 113.0])
    w = np.array([22.0, -3.0, -123.0])
    result = Q.hamilton_product(q, w)
    assert np.array_equal(result, [13889, 2047, 2309, -12391])


class TestAngleAxisToQuat:
    @staticmethod
    def equal(angle, axis, q, deg=False):
        assert np.array_equal(Q.angle_axis_to_q(angle, axis, deg), q)

    @staticmethod
    def close(angle, axis, q, deg=False):
        assert np.allclose(Q.angle_axis_to_q(angle, axis, deg), q)

    def test_identity(self):
        self.equal(0, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0]))
        self.equal(0, np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0]))
        self.equal(0, np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0]))

    def test_180(self):
        self.equal(
            180, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0]), deg=True
        )
        self.equal(
            180, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0]), deg=True
        )
        self.equal(
            180, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1]), deg=True
        )

    def test_90(self):
        rt2 = 1 / np.sqrt(2)
        self.close(
            90, np.array([1.0, 0.0, 0.0]), np.array([rt2, rt2, 0.0, 0.0]), deg=True
        )
        self.close(
            90, np.array([0.0, 1.0, 0.0]), np.array([rt2, 0.0, rt2, 0.0]), deg=True
        )
        self.close(
            90, np.array([0.0, 0.0, 1.0]), np.array([rt2, 0.0, 0.0, rt2]), deg=True
        )

    def test_all_axes(self):
        rt2 = 1 / np.sqrt(2)
        rt6 = 1 / np.sqrt(6)
        self.close(
            90, np.array([1.0, 1.0, 1.0]), np.array([rt2, rt6, rt6, rt6]), deg=True
        )
        self.close(
            90,
            np.array([-1.0, -1.0, -1.0]),
            np.array([rt2, -rt6, -rt6, -rt6]),
            deg=True,
        )

    def test_example_in_pdf(self):
        self.close(
            2 / 3 * np.pi, np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5, 0.5])
        )


class TestQuatToAxisAngle:
    @staticmethod
    def equal(angle, axis, q, deg=False):
        r_angle, r_axis = Q.q_to_angle_axis(q, deg)

        assert np.allclose(r_axis, axis)
        assert abs(angle - r_angle) < 1e-6

    @staticmethod
    def close(angle, axis, q, deg=False):
        assert np.allclose(Q.angle_axis_to_q(angle, axis, deg), q)

    def test_identity(self):
        self.equal(0, np.array([0, 0, 0]), np.array([1, 0, 0, 0]))

    def test_180(self):
        self.equal(180, np.array([1, 0, 0]), np.array([0, 1, 0, 0]), deg=True)
        self.equal(180, np.array([0, 1, 0]), np.array([0, 0, 1, 0]), deg=True)
        self.equal(180, np.array([0, 0, 1]), np.array([0, 0, 0, 1]), deg=True)

    def test_90(self):
        rt2 = 1 / np.sqrt(2)
        self.close(90, np.array([1, 0, 0]), [rt2, rt2, 0, 0], deg=True)
        self.close(90, np.array([0, 1, 0]), [rt2, 0, rt2, 0], deg=True)
        self.close(90, np.array([0, 0, 1]), [rt2, 0, 0, rt2], deg=True)

    def test_all_axes(self):
        rt2 = 1 / np.sqrt(2)
        rt6 = 1 / np.sqrt(6)
        self.close(90, np.array([1, 1, 1]), [rt2, rt6, rt6, rt6], deg=True)
        self.close(90, np.array([-1, -1, -1]), [rt2, -rt6, -rt6, -rt6], deg=True)

    def test_example_in_pdf(self):
        self.close(2 / 3 * np.pi, np.array([1, 1, 1]), [0.5, 0.5, 0.5, 0.5])


class TestDCMToQ:
    def test_identity(self):
        dcm = np.eye(3)
        assert np.array_equal(Q.DCM_to_q(dcm), np.array([1, 0, 0, 0]))

    def test_180_rot(self):
        dcm = np.diag([1, -1, -1])
        assert np.array_equal(Q.DCM_to_q(dcm), np.array([0, 1, 0, 0]))
        dcm = np.diag([-1, 1, -1])
        assert np.array_equal(Q.DCM_to_q(dcm), np.array([0, 0, 1, 0]))
        dcm = np.diag([-1, -1, 1])
        assert np.array_equal(Q.DCM_to_q(dcm), np.array([0, 0, 0, 1]))

    def test_random(self):
        """All of tehse are calculated with calculator that is for ROTATION / active matrices
        https://www.andre-gaschler.com/rotationconverter/"""
        dcm = np.linalg.inv(
            np.array(
                [
                    [0.9772839, -0.1380712, 0.1607873],
                    [0.1607873, 0.9772839, -0.1380712],
                    [-0.1380712, 0.1607873, 0.9772839],
                ]
            )
        )
        assert np.allclose(
            Q.DCM_to_q(dcm), [0.9914449, 0.0753593, 0.0753593, 0.0753593], rtol=1e-3
        )
        dcm = np.linalg.inv(
            np.array(
                [
                    [0.9205715, -0.0656093, 0.3850241],
                    [0.0701405, 0.9975345, 0.0022810],
                    [-0.3842245, 0.0249059, 0.9229037],
                ]
            )
        )
        assert np.allclose(Q.DCM_to_q(dcm), [0.9799247, 0.0057721, 0.196252, 0.0346327])
        dcm = np.linalg.inv(
            np.array(
                [
                    [0.0008382, -0.1452129, 0.9894001],
                    [0.2022121, 0.9689857, 0.1420454],
                    [-0.9793414, 0.1999496, 0.0301760],
                ]
            )
        )
        assert np.allclose(
            Q.DCM_to_q(dcm), [0.7071068, 0.0204722, 0.6960552, 0.1228333]
        )


class TestQToDCM:
    def test_identity(self):
        dcm = np.eye(3)
        assert np.array_equal(dcm, Q.q_to_DCM(np.array([1.0, 0.0, 0.0, 0.0])))

    def test_180_rot(self):
        dcm = np.diag([1, -1, -1])
        assert np.array_equal(dcm, Q.q_to_DCM(np.array([0.0, 1.0, 0.0, 0.0])))
        dcm = np.diag([-1, 1, -1])
        assert np.array_equal(dcm, Q.q_to_DCM(np.array([0.0, 0.0, 1.0, 0.0])))
        dcm = np.diag([-1, -1, 1])
        assert np.array_equal(dcm, Q.q_to_DCM(np.array([0.0, 0.0, 0.0, 1.0])))

    def test_random(self):
        """All of tehse are calculated with calculator that is for ROTATION / active matrices
        https://www.andre-gaschler.com/rotationconverter/"""
        dcm = np.linalg.inv(
            np.array(
                [
                    [0.9772839, -0.1380712, 0.1607873],
                    [0.1607873, 0.9772839, -0.1380712],
                    [-0.1380712, 0.1607873, 0.9772839],
                ]
            )
        )
        assert np.allclose(
            dcm,
            Q.q_to_DCM(np.array([0.9914449, 0.0753593, 0.0753593, 0.0753593])),
            rtol=1e-3,
        )
        dcm = np.linalg.inv(
            np.array(
                [
                    [0.9205715, -0.0656093, 0.3850241],
                    [0.0701405, 0.9975345, 0.0022810],
                    [-0.3842245, 0.0249059, 0.9229037],
                ]
            )
        )
        assert np.allclose(
            dcm, Q.q_to_DCM(np.array([0.9799247, 0.0057721, 0.196252, 0.0346327]))
        )
        dcm = np.linalg.inv(
            np.array(
                [
                    [0.0008382, -0.1452129, 0.9894001],
                    [0.2022121, 0.9689857, 0.1420454],
                    [-0.9793414, 0.1999496, 0.0301760],
                ]
            )
        )
        assert np.allclose(
            dcm,
            Q.q_to_DCM(np.array([0.7071068, 0.0204722, 0.6960552, 0.1228333])),
            rtol=1e-3,
        )


def test_dcm_sign_equiv():
    dcm = Q.q_to_DCM(np.array([0.7071068, 0.0204722, 0.6960552, 0.1228333]))
    dcm_n = Q.q_to_DCM(np.array([-0.7071068, -0.0204722, -0.6960552, -0.1228333]))
    assert np.array_equal(dcm, dcm_n)


def test_dcm_round_trip_consistency():

    quats = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9914449, 0.0753593, 0.0753593, 0.0753593],
        [0.7071068, 0.0204722, 0.6960552, 0.1228333],
    ]

    for q in quats:
        dcm = Q.q_to_DCM(np.array(q))
        q2 = Q.DCM_to_q(dcm)
        assert np.allclose(q2, q)


class TestApply:
    def test_Z_axis_45_around_X_axis_active(self):
        """
        Starts with quaternion representing a 90deg rotation around the x axis.
        The matrix rotates the vector `V` `[0,0,1]` to produce `V'` `[0,-1,0]`
        """
        q = Q.angle_axis_to_q(90, np.array([1, 0, 0]), True)
        v = [0, 0, 1]
        v_result = Q.quat_apply(q, v, passive=False)

        assert np.allclose(v_result, [0, -1, 0])

    def test_Z_axis_from_frame_rotated_90_deg_around_X_axis_passive(self):
        """
        Starts with quaternion representing a 90deg rotation around the x axis.
        The matrix expresses the vector in coordiante system A (`V_a`= `[0,0,1]`)
        as the same vector from a rotated coordinate system B (`V_b` = `[0,1,0]`)
        """
        q = Q.angle_axis_to_q(90, np.array([1, 0, 0]), True)
        v = [0, 0, 1]
        v_result = Q.quat_apply(q, v, passive=True)

        assert np.allclose(v_result, np.array([0, 1, 0]))

    def test_Z_axis_small_deg_around_X_axis_active(self):
        """
        Starts with quaternion representing a 1e-6 deg rotation around the x axis.
        The matrix rotates the vector `V` `[0,0,1]` by 1e-6 deg to make `V'` `[0, ~0-, ~1]`
        """
        q = Q.angle_axis_to_q(1e-6, np.array([1, 0, 0]), True)
        v = [0, 0, 1]
        v_result = Q.quat_apply(q, v, passive=False)
        angle = 1e-6 * DEG_TO_RAD
        assert np.allclose(v_result, [0, -np.sin(angle), np.cos(angle)])

    def test_Z_axis_from_rate_rotated_small_deg_around_X_axis_passive(self):
        """
        Starts with quaternion representing a 1e-6 rotation around the x axis.
        The matrix expresses the vector in coordiante system A (`V_a`= `[0,0,1]`)
        as the same vector from a rotated coordinate system B (`V_b` = `[0,~0+, ~1]`)
        """
        q = Q.angle_axis_to_q(1e-6, np.array([1, 0, 0]), True)
        v = [0, 0, 1]
        v_result = Q.quat_apply(q, v, passive=True)
        angle = 1e-6 * DEG_TO_RAD
        assert np.allclose(v_result, [0, np.sin(angle), np.cos(angle)])

        # @testset "Weird rotations" begin
        #     """
        #     Starts with quaternion representing a 90deg rotation around the x axis.
        #     The matrix rotates the vector `V` `[0,0,1]` to produce `V'` `[0,1,0]`
        #     """
        #     q = angle_axis_to_q(90.0, [0,0,1], true)
        #     v = Float64[1,1,1]
        #     v_result = quat_apply(q, v)
        #     @test isapprox(v_result, [1, -1, 1])

        #     # From https://www.vcalc.com/wiki/vector-rotation
        #     q = angle_axis_to_q(13.0, [2,13,5.6], true)
        #     v = Float64[1,2,7]
        #     v_result = quat_apply(q, v, false)
        #     @test isapprox(v_result, [2.2469464036,1.9261221082,6.7261642474])
        # end
