from itertools import product

def tensorisation_maker(lazy_tensorisation):
    # from a lazy_tensorisation for a n_1*..*n_m tensorisation we obtain a more detailed 
    # tensorisation, useful to make more complex truncature of the space than
    # rectangular one through inequalities.
    # as an exemple (2,3) should ouput ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)).

    tensorisation = tuple(product(*[range(dim) for dim in lazy_tensorisation]))
    
    return tensorisation

def unit_test_tensorisation_maker():
    def assert_equal(actual, expected):
        if actual == expected:
            return True
        else:
            return False
    # Test for a 2D tensor (2, 3)
    input_tensor = (2, 3)
    expected_output = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    if assert_equal(tensorisation_maker(input_tensor), expected_output): return False

    # user not expected to do this one though
    # Test for a 1D tensor (4,)
    input_tensor = (4,)
    expected_output = [(0,), (1,), (2,), (3,)]
    if assert_equal(tensorisation_maker(input_tensor), expected_output): return False

    # Test for a 3D tensor (2, 2, 2)
    input_tensor = (2, 2, 2)
    expected_output = [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
    ]
    if assert_equal(tensorisation_maker(input_tensor), expected_output): return False

    return True
