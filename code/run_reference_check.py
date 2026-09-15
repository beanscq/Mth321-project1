from code.reference import make_output_times, solve_reference, compare_references
from code.robertson import robertson_rhs


def main():
    t_eval = make_output_times()
    t_ref, y_ref = solve_reference(robertson_rhs, [1.0, 0.0, 0.0], t_eval)
    reference_difference = compare_references(robertson_rhs, [1.0, 0.0, 0.0], t_eval)

    print("Reference output time shape:", t_ref.shape)
    print("Reference solution shape:", y_ref.shape)
    print("Final reference value at t = 40:")
    print(y_ref[-1])
    print("Maximum difference between reference solves:")
    print(reference_difference)


if __name__ == "__main__":
    main()