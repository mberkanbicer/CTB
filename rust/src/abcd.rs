//! ABCD (transmission) matrix building blocks.
//!
//! Ports of `a_pi.m`, `a_t.m`, `a_series.m`, `a_shunt.m` and `a_mul.m`. All
//! matrices are `2 x 2 x F` with one entry per frequency.
//!
//! Reference: David M. Pozar, "Microwave Engineering".

use crate::C64;
use ndarray::{Array1, Array3};

/// Check that all frequency vectors have the same non-zero length.
fn common_length(what: &str, lengths: &[usize]) -> Result<usize, String> {
    let first = *lengths.first().unwrap_or(&0);
    if first == 0 {
        return Err(format!("{what} requires at least one frequency"));
    }
    for (index, len) in lengths.iter().enumerate() {
        if *len != first {
            return Err(format!(
                "{what}: argument {} has {} frequencies, expected {}",
                index + 1,
                len,
                first
            ));
        }
    }
    Ok(first)
}

/// ABCD matrix of a Pi network (`a_PI.m`).
///
/// ```text
/// a(1,1) = 1 + Y2/Y3
/// a(1,2) = 1/Y3
/// a(2,1) = Y1 + Y2 + Y1*Y2/Y3
/// a(2,2) = 1 + Y1/Y3
/// ```
///
/// `Y1` is the shunt admittance at port 1, `Y2` the shunt admittance at port 2
/// and `Y3` the admittance of the series element between them.
pub fn a_pi(y1: &Array1<C64>, y2: &Array1<C64>, y3: &Array1<C64>) -> Result<Array3<C64>, String> {
    let nf = common_length("a_pi", &[y1.len(), y2.len(), y3.len()])?;
    let one = C64::new(1.0, 0.0);
    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        a[[0, 0, f]] = one + y2[f] / y3[f];
        a[[0, 1, f]] = one / y3[f];
        a[[1, 0, f]] = y1[f] + y2[f] + y1[f] * y2[f] / y3[f];
        a[[1, 1, f]] = one + y1[f] / y3[f];
    }
    Ok(a)
}

/// ABCD matrix of a T network (`a_T.m`).
///
/// ```text
/// a(1,1) = 1 + Z1/Z2
/// a(1,2) = Z1 + Z2 + Z1*Z2/Z3
/// a(2,1) = 1/Z3
/// a(2,2) = 1 + Z2/Z3
/// ```
pub fn a_t(z1: &Array1<C64>, z2: &Array1<C64>, z3: &Array1<C64>) -> Result<Array3<C64>, String> {
    let nf = common_length("a_t", &[z1.len(), z2.len(), z3.len()])?;
    let one = C64::new(1.0, 0.0);
    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        a[[0, 0, f]] = one + z1[f] / z2[f];
        a[[0, 1, f]] = z1[f] + z2[f] + z1[f] * z2[f] / z3[f];
        a[[1, 0, f]] = one / z3[f];
        a[[1, 1, f]] = one + z2[f] / z3[f];
    }
    Ok(a)
}

/// ABCD matrix of a series impedance (`a_series.m`).
///
/// `a = [1 Z; 0 1]` per frequency.
pub fn a_series(z: &Array1<C64>) -> Array3<C64> {
    let nf = z.len();
    let one = C64::new(1.0, 0.0);
    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        a[[0, 0, f]] = one;
        a[[0, 1, f]] = z[f];
        a[[1, 0, f]] = C64::new(0.0, 0.0);
        a[[1, 1, f]] = one;
    }
    a
}

/// ABCD matrix of a shunt admittance (`a_shunt.m`).
///
/// `a = [1 0; Y 1]` per frequency.
pub fn a_shunt(y: &Array1<C64>) -> Array3<C64> {
    let nf = y.len();
    let one = C64::new(1.0, 0.0);
    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        a[[0, 0, f]] = one;
        a[[0, 1, f]] = C64::new(0.0, 0.0);
        a[[1, 0, f]] = y[f];
        a[[1, 1, f]] = one;
    }
    a
}

/// Multiply a sequence of ABCD matrices (`a_mul.m`): `a = a1 * a2 * ... * aN`.
///
/// The first argument is returned unchanged for a single argument, exactly like
/// the M-file (`a = varargin{1}`).
pub fn a_mul(arrays: &[&Array3<C64>]) -> Result<Array3<C64>, String> {
    let mut iter = arrays.iter();
    let first = match iter.next() {
        Some(first) => first,
        None => return Err("a_mul: no argument is given".to_string()),
    };
    let n = first.shape()[0];
    let m = first.shape()[1];
    if first.ndim() != 3 || n != 2 || m != 2 {
        return Err(format!(
            "a_mul requires 2x2xF matrices, got {}x{}x{}",
            n,
            m,
            first.shape()[2]
        ));
    }
    let nf = first.shape()[2];
    let mut result = (*first).clone();
    for other in iter {
        if other.shape() != first.shape() {
            return Err(format!(
                "a_mul: dimension mismatch, {:?} cannot be multiplied with {:?}",
                first.shape(),
                other.shape()
            ));
        }
        for f in 0..nf {
            let mut prod = ndarray::Array2::zeros((2, 2));
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = C64::new(0.0, 0.0);
                    for k in 0..2 {
                        sum += result[[i, k, f]] * other[[k, j, f]];
                    }
                    prod[[i, j]] = sum;
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    result[[i, j, f]] = prod[[i, j]];
                }
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec(values: &[C64]) -> Array1<C64> {
        Array1::from_vec(values.to_vec())
    }

    fn close(a: &Array3<C64>, b: &Array3<C64>, tol: f64) {
        assert_eq!(a.shape(), b.shape());
        for ((i, j, f), value) in a.indexed_iter() {
            let want = b[[i, j, f]];
            let diff = (*value - want).norm();
            assert!(
                diff <= tol * want.norm().max(1.0),
                "({i},{j},{f}) {value} vs {want}"
            );
        }
    }

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    #[test]
    fn series_and_shunt_have_the_pozar_layout() {
        let z = vec(&[c(1.0, 2.0), c(3.0, -1.0)]);
        let a = a_series(&z);
        assert_eq!(a[[0, 0, 0]], c(1.0, 0.0));
        assert_eq!(a[[1, 1, 1]], c(1.0, 0.0));
        assert_eq!(a[[0, 1, 0]], c(1.0, 2.0));
        assert_eq!(a[[1, 0, 1]], c(0.0, 0.0));

        let y = vec(&[c(0.1, 0.2)]);
        let b = a_shunt(&y);
        assert_eq!(b[[0, 0, 0]], c(1.0, 0.0));
        assert_eq!(b[[0, 1, 0]], c(0.0, 0.0));
        assert_eq!(b[[1, 0, 0]], c(0.1, 0.2));
        assert_eq!(b[[1, 1, 0]], c(1.0, 0.0));
    }

    #[test]
    fn pi_network_matches_a_cascade_of_its_elements() {
        let y1 = vec(&[c(0.01, 0.02), c(0.02, 0.01)]);
        let y2 = vec(&[c(0.03, -0.01), c(0.0, 0.05)]);
        let y3 = vec(&[c(0.5, 0.1), c(0.4, -0.2)]);
        let pi = a_pi(&y1, &y2, &y3).unwrap();
        let mut z = Array1::zeros(2);
        for f in 0..2 {
            z[f] = c(1.0, 0.0) / y3[f];
        }
        let cascade = a_mul(&[&a_shunt(&y1), &a_series(&z), &a_shunt(&y2)]).unwrap();
        close(&pi, &cascade, 1e-15);
    }

    #[test]
    fn t_network_matches_a_cascade_of_its_elements() {
        // a_T.m preserves a native defect: A(1,1) divides Z1 by Z2 while the
        // rest of the matrix uses Z3 as the shunt arm, so the result is not
        // reciprocal (det != 1). The M-file values are reproduced verbatim; this
        // test pins them together with the non-reciprocity.
        let z1 = vec(&[c(10.0, 5.0), c(20.0, -5.0)]);
        let z2 = vec(&[c(5.0, 1.0), c(1.0, 2.0)]);
        let z3 = vec(&[c(50.0, 0.0), c(30.0, -10.0)]);
        let t = a_t(&z1, &z2, &z3).unwrap();
        for f in 0..2 {
            assert!((t[[0, 0, f]] - (c(1.0, 0.0) + z1[f] / z2[f])).norm() < 1e-15);
            assert!((t[[1, 0, f]] - c(1.0, 0.0) / z3[f]).norm() < 1e-15);
            assert!((t[[1, 1, f]] - (c(1.0, 0.0) + z2[f] / z3[f])).norm() < 1e-15);
            assert!((t[[0, 1, f]] - (z1[f] + z2[f] + z1[f] * z2[f] / z3[f])).norm() < 1e-14);
            let det = t[[0, 0, f]] * t[[1, 1, f]] - t[[0, 1, f]] * t[[1, 0, f]];
            assert!(
                (det - c(1.0, 0.0)).norm() > 1e-3,
                "the native a_T defect would be gone if det == 1"
            );
        }
    }

    #[test]
    fn t_network_uses_its_third_impedance_as_the_shunt_arm_everywhere_else() {
        // The physically consistent T network is a_series(Z1) * a_shunt(1/Z3) *
        // a_series(Z2); the native A(1,1) is the only entry that deviates.
        let z1 = vec(&[c(10.0, 5.0)]);
        let z2 = vec(&[c(5.0, 1.0)]);
        let z3 = vec(&[c(50.0, 0.0)]);
        let t = a_t(&z1, &z2, &z3).unwrap();
        let mut y = Array1::zeros(1);
        y[0] = c(1.0, 0.0) / z3[0];
        let cascade = a_mul(&[&a_series(&z1), &a_shunt(&y), &a_series(&z2)]).unwrap();
        assert!((t[[0, 1, 0]] - cascade[[0, 1, 0]]).norm() < 1e-14);
        assert!((t[[1, 0, 0]] - cascade[[1, 0, 0]]).norm() < 1e-15);
        assert!((t[[1, 1, 0]] - cascade[[1, 1, 0]]).norm() < 1e-15);
        assert!((t[[0, 0, 0]] - cascade[[0, 0, 0]]).norm() > 1e-3);
    }

    #[test]
    fn pi_without_shunt_arms_is_a_series_element() {
        let zero = vec(&[c(0.0, 0.0), c(0.0, 0.0)]);
        let y3 = vec(&[c(0.02, 0.0), c(0.01, 0.01)]);
        let mut z = Array1::zeros(2);
        for f in 0..2 {
            z[f] = c(1.0, 0.0) / y3[f];
        }
        close(&a_pi(&zero, &zero, &y3).unwrap(), &a_series(&z), 1e-15);
    }

    #[test]
    fn multiplication_is_ordered_and_rejects_wrong_shapes() {
        let z = vec(&[c(2.0, 1.0)]);
        let y = vec(&[c(0.5, 0.1)]);
        let series_shunt = a_mul(&[&a_series(&z), &a_shunt(&y)]).unwrap();
        let shunt_series = a_mul(&[&a_shunt(&y), &a_series(&z)]).unwrap();
        assert!((series_shunt[[1, 1, 0]] - shunt_series[[1, 1, 0]]).norm() > 1e-6);
        // A*B = [1 + ZY, Z; Y, 1]
        assert!((series_shunt[[0, 0, 0]] - (c(1.0, 0.0) + z[0] * y[0])).norm() < 1e-15);
        assert!((series_shunt[[0, 1, 0]] - z[0]).norm() < 1e-15);
        assert!((series_shunt[[1, 0, 0]] - y[0]).norm() < 1e-15);
        assert!((series_shunt[[1, 1, 0]] - c(1.0, 0.0)).norm() < 1e-15);

        assert!(a_mul(&[]).unwrap_err().contains("no argument"));
        let one = Array3::<C64>::zeros((2, 2, 1));
        let other = Array3::<C64>::zeros((2, 2, 2));
        assert!(a_mul(&[&one, &other])
            .unwrap_err()
            .contains("dimension mismatch"));
        let mut bad = Array3::<C64>::zeros((3, 3, 1));
        bad[[0, 0, 0]] = c(1.0, 0.0);
        assert!(a_mul(&[&bad]).unwrap_err().contains("2x2xF"));
    }

    #[test]
    fn single_argument_multiplication_returns_the_input() {
        let z = vec(&[c(2.0, 1.0)]);
        let a = a_series(&z);
        close(&a_mul(&[&a]).unwrap(), &a, 0.0);
    }

    #[test]
    fn mismatched_lengths_are_errors() {
        let a = vec(&[c(1.0, 0.0)]);
        let b = vec(&[c(1.0, 0.0), c(2.0, 0.0)]);
        assert!(a_pi(&a, &b, &a).unwrap_err().contains("argument 2"));
        assert!(a_t(&a, &b, &a).unwrap_err().contains("argument 2"));
        let empty = Array1::<C64>::zeros(0);
        assert!(a_pi(&empty, &empty, &empty)
            .unwrap_err()
            .contains("at least one"));
    }
}
