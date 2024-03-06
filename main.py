from src import DFT, EulerScheme

if __name__ == '__main__':
    print('Numericals')
    DFT.discrete_fourier_transformation_test('files\\58.txt')
    # EulerScheme.test_euler()