import numpy as np
import pytest

from twisterl.rl.encoders import OneHotEncoder, IdentityEncoder, ObservationEncoder, create_encoder


class TestOneHotEncoder:
    def test_single_observation(self):
        encoder = OneHotEncoder(obs_size=5)
        obs = [2]
        result = encoder.encode(obs)

        expected = np.zeros((1, 5), dtype=float)
        expected[0, 2] = 1.0

        np.testing.assert_array_equal(result, expected)

    def test_multiple_observations(self):
        encoder = OneHotEncoder(obs_size=4)
        obs = [0, 3, 1]
        result = encoder.encode(obs)

        expected = np.zeros((3, 4), dtype=float)
        expected[0, 0] = 1.0
        expected[1, 3] = 1.0
        expected[2, 1] = 1.0

        np.testing.assert_array_equal(result, expected)

    def test_empty_observations(self):
        encoder = OneHotEncoder(obs_size=3)
        obs = []
        result = encoder.encode(obs)

        expected = np.zeros((0, 3), dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape(self):
        encoder = OneHotEncoder(obs_size=10)
        obs = [1, 5, 9, 0]
        result = encoder.encode(obs)

        assert result.shape == (4, 10)
        assert result.dtype == float

    def test_all_zeros_except_one(self):
        encoder = OneHotEncoder(obs_size=6)
        obs = [3]
        result = encoder.encode(obs)

        # Check that only one position is 1.0 and rest are 0.0
        assert np.sum(result) == 1.0
        assert result[0, 3] == 1.0

        # Check all other positions are zero
        mask = np.ones(6, dtype=bool)
        mask[3] = False
        assert np.all(result[0, mask] == 0.0)


class TestIdentityEncoder:
    def test_single_observation(self):
        encoder = IdentityEncoder()
        obs = [2.5]
        result = encoder.encode(obs)

        expected = np.array([2.5], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_observations(self):
        encoder = IdentityEncoder()
        obs = [1.0, 3.14, -2.5, 0.0]
        result = encoder.encode(obs)

        expected = np.array([1.0, 3.14, -2.5, 0.0], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_integer_observations(self):
        encoder = IdentityEncoder()
        obs = [1, 2, 3]
        result = encoder.encode(obs)

        expected = np.array([1.0, 2.0, 3.0], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_empty_observations(self):
        encoder = IdentityEncoder()
        obs = []
        result = encoder.encode(obs)

        expected = np.array([], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_output_dtype(self):
        encoder = IdentityEncoder()
        obs = [1, 2, 3]
        result = encoder.encode(obs)

        assert result.dtype == float


class TestObservationEncoder:
    def test_is_abstract_base_class(self):
        with pytest.raises(TypeError):
            ObservationEncoder()

    def test_abstract_method(self):
        class IncompleteEncoder(ObservationEncoder):
            pass

        with pytest.raises(TypeError):
            IncompleteEncoder()


class TestCreateEncoder:
    def test_create_one_hot_encoder(self):
        encoder = create_encoder("one_hot", obs_size=5)
        assert isinstance(encoder, OneHotEncoder)
        assert encoder.obs_size == 5

    def test_create_identity_encoder(self):
        encoder = create_encoder("identity", obs_size=10)  # obs_size ignored for identity
        assert isinstance(encoder, IdentityEncoder)

    def test_unknown_encoder_type(self):
        with pytest.raises(ValueError, match="Unknown encoder type: invalid"):
            create_encoder("invalid", obs_size=5)

    def test_factory_creates_working_encoders(self):
        # Test that factory-created encoders work correctly
        one_hot = create_encoder("one_hot", obs_size=3)
        identity = create_encoder("identity", obs_size=3)

        obs = [1, 0, 2]

        # Test one-hot encoder
        one_hot_result = one_hot.encode(obs)
        expected_one_hot = np.zeros((3, 3), dtype=float)
        expected_one_hot[0, 1] = 1.0
        expected_one_hot[1, 0] = 1.0
        expected_one_hot[2, 2] = 1.0
        np.testing.assert_array_equal(one_hot_result, expected_one_hot)

        # Test identity encoder
        identity_result = identity.encode(obs)
        expected_identity = np.array([1.0, 0.0, 2.0], dtype=float)
        np.testing.assert_array_equal(identity_result, expected_identity)


class TestEncoderIntegration:
    def test_encoders_handle_edge_cases(self):
        # Test both encoders with various edge cases
        one_hot = OneHotEncoder(obs_size=1)
        identity = IdentityEncoder()

        # Single element
        obs = [0]
        one_hot_result = one_hot.encode(obs)
        identity_result = identity.encode(obs)

        assert one_hot_result.shape == (1, 1)
        assert one_hot_result[0, 0] == 1.0
        assert identity_result[0] == 0.0

    def test_consistent_output_format(self):
        # Both encoders should return numpy arrays
        one_hot = OneHotEncoder(obs_size=3)
        identity = IdentityEncoder()

        obs = [1, 2]

        one_hot_result = one_hot.encode(obs)
        identity_result = identity.encode(obs)

        assert isinstance(one_hot_result, np.ndarray)
        assert isinstance(identity_result, np.ndarray)
        assert one_hot_result.dtype == float
        assert identity_result.dtype == float