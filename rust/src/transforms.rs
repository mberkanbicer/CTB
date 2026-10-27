//! Network parameter transforms (S, Z, Y, ABCD, H, G).
//!
//! Every function is a direct port of the corresponding CTB M-file:
//!
//! | Rust | M-file |
//! | --- | --- |
//! | [`s2z`] | `s2z.m` |
//! | [`z2s`] | `z2s.m` |
//! | [`s2y`] | `s2y.m` |
//! | [`y2s`] | `y2s.m` |
//! | [`z2y`] | `z2y.m` |
//! | [`y2z`] | `y2z.m` |
//! | [`s2a`] | `s2a.m` (2-port only) |
//! | [`a2s`] | `a2s.m` (2-port only) |
//! | [`s_renorm`] | `s_renorm.m` |
//!
//! `s2h`/`h2s`/`s2g`/`g2s` have no M-file counterpart; they are built from the
//! Z/Y transforms plus the classic hybrid (H) and inverse-hybrid (G)
//! definitions, which keeps them consistent with the native S/Z/Y results.

use crate::{RefImpedance, C64};
use ndarray::{Array2, Array3};

// ---------------------------------------------------------------------------
// Dense complex linear algebra
// ---------------------------------------------------------------------------

/// Invert a complex square matrix with Gauss-Jordan elimination and partial
/// pivoting.
pub(crate) fn cplx_inv(m: &Array2<C64>) -> Result<Array2<C64>, String> {
    let n = m.nrows();
    if n == 0 {
        return Err("cannot invert an empty matrix".to_string());
    }
    if m.ncols() != n {
        return Err(format!(
            "cannot invert a {}x{} matrix, it must be square",
            n,
            m.ncols()
        ));
    }
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = m[[i, j]];
        }
        aug[[i, n + i]] = C64::new(1.0, 0.0);
    }

    // Scale-relative singularity threshold: Octave's `inv` only fails for
    // matrices that are singular to machine precision, so a fixed absolute
    // epsilon would reject perfectly invertible matrices with small entries.
    let scale = m.iter().map(|value| value.norm()).fold(0.0f64, f64::max);
    let tolerance = (scale * 1e-14).max(f64::MIN_POSITIVE);

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = aug[[col, col]].norm();
        for row in (col + 1)..n {
            let val = aug[[row, col]].norm();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }
        if max_val.is_nan() || max_val <= tolerance {
            return Err(format!("Matrix is singular at column {}", col + 1));
        }
        if pivot_row != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                if factor != C64::new(0.0, 0.0) {
                    for j in 0..2 * n {
                        aug[[row, j]] = aug[[row, j]] - factor * aug[[col, j]];
                    }
                }
            }
        }
    }

    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

/// Matrix left division `A \ B = inv(A) * B`.
fn mldivide(a: &Array2<C64>, b: &Array2<C64>) -> Result<Array2<C64>, String> {
    if a.nrows() != b.nrows() {
        return Err(format!(
            "left division {}x{} \\ {}x{} is not defined",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        ));
    }
    Ok(cplx_inv(a)?.dot(b))
}

/// Solve `A * x = B`, i.e. MATLAB's `A \ B`.
pub(crate) fn solve(a: &Array2<C64>, b: &Array2<C64>) -> Result<Array2<C64>, String> {
    mldivide(a, b)
}

/// Matrix right division `A / B = A * inv(B)`.
fn mrdivide(a: &Array2<C64>, b: &Array2<C64>) -> Result<Array2<C64>, String> {
    if a.ncols() != b.ncols() {
        return Err(format!(
            "right division {}x{} / {}x{} is not defined",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        ));
    }
    Ok(a.dot(&cplx_inv(b)?))
}

/// `Zref = diag(z0)` and `G = diag(1/sqrt(real(z0)))`, as in the M-files.
fn make_zref_g(n: usize, z0: &[C64]) -> (Array2<C64>, Array2<C64>) {
    let mut zref = Array2::zeros((n, n));
    let mut g = Array2::zeros((n, n));
    for i in 0..n {
        zref[[i, i]] = z0[i];
        // s2z.m/s2y.m/z2s.m/y2s.m: G(n,n) = 1/sqrt(real(Z0(n)))
        g[[i, i]] = C64::new(1.0 / z0[i].re.sqrt(), 0.0);
    }
    (zref, g)
}

fn get_freq_matrix(arr: &Array3<C64>, f: usize) -> Array2<C64> {
    let n = arr.shape()[0];
    let m = arr.shape()[1];
    let mut out = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[[i, j]] = arr[[i, j, f]];
        }
    }
    out
}

fn set_freq_matrix(arr: &mut Array3<C64>, f: usize, val: &Array2<C64>) {
    let n = arr.shape()[0];
    let m = arr.shape()[1];
    for i in 0..n {
        for j in 0..m {
            arr[[i, j, f]] = val[[i, j]];
        }
    }
}

fn require_square(arr: &Array3<C64>, what: &str) -> Result<usize, String> {
    let n = arr.shape()[0];
    let m = arr.shape()[1];
    if n != m {
        return Err(format!(
            "{what} requires square matrices, got {n}x{m} per frequency"
        ));
    }
    if n == 0 {
        return Err(format!("{what} requires at least one port"));
    }
    Ok(n)
}

fn require_two_port(arr: &Array3<C64>, what: &str) -> Result<(), String> {
    let n = arr.shape()[0];
    let m = arr.shape()[1];
    if n != 2 || m != 2 {
        return Err(format!(
            "{what} requires a 2x2xF array, got {n}x{m}x{}",
            arr.shape()[2]
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// S <-> Z / Y
// ---------------------------------------------------------------------------

/// Scattering to impedance parameters.
///
/// `z = G \ ((E - s) \ (s + E)) * Zref * G` per frequency, with
/// `Zref = diag(z0)`, `G = diag(1/sqrt(real(z0)))` (`s2z.m`).
pub fn s2z(s: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    let n = require_square(s, "s2z")?;
    let nf = s.shape()[2];
    let z0 = ref_z.into().resolve(n)?;
    let (zref, g) = make_zref_g(n, &z0);
    let g_inv = cplx_inv(&g)?;
    let e: Array2<C64> = Array2::eye(n);

    let mut z = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let s_f = get_freq_matrix(s, f);
        let tmp = mldivide(&(&e - &s_f), &(&s_f + &e))?;
        let tmp = tmp.dot(&zref);
        let tmp = tmp.dot(&g);
        set_freq_matrix(&mut z, f, &g_inv.dot(&tmp));
    }
    Ok(z)
}

/// Impedance to scattering parameters.
///
/// `s = G * (z - Zref) / (z + Zref) / G` per frequency (`z2s.m`).
pub fn z2s(z: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    let n = require_square(z, "z2s")?;
    let nf = z.shape()[2];
    let z0 = ref_z.into().resolve(n)?;
    let (zref, g) = make_zref_g(n, &z0);

    let mut s = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let z_f = get_freq_matrix(z, f);
        let tmp = mrdivide(&(&z_f - &zref), &(&z_f + &zref))?;
        let tmp = g.dot(&tmp);
        set_freq_matrix(&mut s, f, &mrdivide(&tmp, &g)?);
    }
    Ok(s)
}

/// Scattering to admittance parameters.
///
/// `y = G \ (Zref \ ((s + E) \ (E - s))) * G` per frequency (`s2y.m`).
pub fn s2y(s: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    let n = require_square(s, "s2y")?;
    let nf = s.shape()[2];
    let z0 = ref_z.into().resolve(n)?;
    let (zref, g) = make_zref_g(n, &z0);
    let g_inv = cplx_inv(&g)?;
    let e: Array2<C64> = Array2::eye(n);

    let mut y = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let s_f = get_freq_matrix(s, f);
        let tmp = mldivide(&(&s_f + &e), &(&e - &s_f))?;
        let tmp = mldivide(&zref, &tmp)?;
        set_freq_matrix(&mut y, f, &g_inv.dot(&tmp).dot(&g));
    }
    Ok(y)
}

/// Admittance to scattering parameters.
///
/// `s = G * ((E - Zref*y) / (E + Zref*y)) / G` per frequency (`y2s.m`).
pub fn y2s(y: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    let n = require_square(y, "y2s")?;
    let nf = y.shape()[2];
    let z0 = ref_z.into().resolve(n)?;
    let (zref, g) = make_zref_g(n, &z0);
    let e: Array2<C64> = Array2::eye(n);

    let mut s = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let y_f = get_freq_matrix(y, f);
        let tmp = mrdivide(&(&e - &zref.dot(&y_f)), &(&e + &zref.dot(&y_f)))?;
        let tmp = g.dot(&tmp);
        set_freq_matrix(&mut s, f, &mrdivide(&tmp, &g)?);
    }
    Ok(s)
}

/// Impedance to admittance parameters (`y2z.m`, i.e. a per-frequency inverse).
pub fn z2y(z: &Array3<C64>) -> Result<Array3<C64>, String> {
    let n = require_square(z, "z2y")?;
    let nf = z.shape()[2];
    let mut y = Array3::zeros((n, n, nf));
    for f in 0..nf {
        set_freq_matrix(&mut y, f, &cplx_inv(&get_freq_matrix(z, f))?);
    }
    Ok(y)
}

/// Admittance to impedance parameters (`y2z.m`, i.e. a per-frequency inverse).
pub fn y2z(y: &Array3<C64>) -> Result<Array3<C64>, String> {
    let n = require_square(y, "y2z")?;
    let nf = y.shape()[2];
    let mut z = Array3::zeros((n, n, nf));
    for f in 0..nf {
        set_freq_matrix(&mut z, f, &cplx_inv(&get_freq_matrix(y, f))?);
    }
    Ok(z)
}

/// Scattering matrix renormalisation (`s_renorm.m`):
/// `s_new = z2s(s2z(s, Z), Z_new)`.
///
/// Both arguments accept a scalar or a per-port reference impedance.
pub fn s_renorm(
    s: &Array3<C64>,
    z: impl Into<RefImpedance>,
    z_new: impl Into<RefImpedance>,
) -> Result<Array3<C64>, String> {
    let z_params = s2z(s, z)?;
    z2s(&z_params, z_new)
}

// ---------------------------------------------------------------------------
// S <-> ABCD (2-port)
// ---------------------------------------------------------------------------

/// Scattering to ABCD parameters (2-port only, `s2a.m`).
///
/// With a scalar reference the formulas are exactly those of `s2a.m`:
///
/// ```text
/// A = ((1+S11)(1-S22) + S12 S21) / (2 S21)
/// B = Z0 ((1+S11)(1+S22) - S12 S21) / (2 S21)
/// C = 1/Z0 ((1-S11)(1-S22) - S12 S21) / (2 S21)
/// D = ((1-S11)(1+S22) + S12 S21) / (2 S21)
/// ```
///
/// With a per-port reference `[Z1, Z2]` the asymmetric generalisation
/// `B -> sqrt(Z1 Z2) ...` and `C -> 1/sqrt(Z1 Z2) ...` is used. The M-file
/// cannot do this: it broadcasts `Z0` against the 1xF row and aborts
/// ("nonconformant arguments") for a 2-element vector.
pub fn s2a(s: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    require_two_port(s, "s2a")?;
    let nf = s.shape()[2];
    let z0 = ref_z.into().resolve(2)?;
    let (z1, z2) = (z0[0], z0[1]);
    let sqrt_zz = geometric_mean(z1, z2);
    // A and D carry a sqrt(Z1/Z2) factor, B and C a sqrt(Z1 Z2) factor; both are
    // 1 for a scalar reference, which reproduces s2a.m exactly.
    let sqrt_ratio = if z1 == z2 {
        C64::new(1.0, 0.0)
    } else {
        (z1 / z2).sqrt()
    };

    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let s11 = s[[0, 0, f]];
        let s12 = s[[0, 1, f]];
        let s21 = s[[1, 0, f]];
        let s22 = s[[1, 1, f]];
        let one = C64::new(1.0, 0.0);
        let denom = C64::new(2.0, 0.0) * s21;

        a[[0, 0, f]] = sqrt_ratio * ((one + s11) * (one - s22) + s12 * s21) / denom;
        a[[0, 1, f]] = sqrt_zz * ((one + s11) * (one + s22) - s12 * s21) / denom;
        a[[1, 0, f]] = (one / sqrt_zz) * ((one - s11) * (one - s22) - s12 * s21) / denom;
        a[[1, 1, f]] = (one / sqrt_ratio) * ((one - s11) * (one + s22) + s12 * s21) / denom;
    }
    Ok(a)
}

/// ABCD to scattering parameters (2-port only, `a2s.m`).
///
/// With a scalar reference the formulas are exactly those of `a2s.m`:
///
/// ```text
/// den = A + B/Z0 + C Z0 + D
/// S11 = (A + B/Z0 - C Z0 - D) / den
/// S12 = 2 (A D - B C) / den
/// S21 = 2 / den
/// S22 = (-A + B/Z0 - C Z0 + D) / den
/// ```
///
/// For a per-port reference `[Z1, Z2]` the denominator becomes
/// `A Z2 + B + C Z1 Z2 + D Z1` and the off-diagonals are scaled by
/// `sqrt(Z1 Z2)`; for `Z1 == Z2` this reduces to the expression above.
pub fn a2s(a: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    require_two_port(a, "a2s")?;
    let nf = a.shape()[2];
    let z0 = ref_z.into().resolve(2)?;
    let (z1, z2) = (z0[0], z0[1]);
    let sqrt_zz = geometric_mean(z1, z2);

    let mut s = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let aa = a[[0, 0, f]];
        let bb = a[[0, 1, f]];
        let cc = a[[1, 0, f]];
        let dd = a[[1, 1, f]];
        let two = C64::new(2.0, 0.0);

        if z1 == z2 {
            // Verbatim a2s.m for a scalar reference.
            let z0 = z1;
            let den = aa + bb / z0 + cc * z0 + dd;
            s[[0, 0, f]] = (aa + bb / z0 - cc * z0 - dd) / den;
            s[[0, 1, f]] = two * (aa * dd - bb * cc) / den;
            s[[1, 0, f]] = two / den;
            s[[1, 1, f]] = (-aa + bb / z0 - cc * z0 + dd) / den;
        } else {
            let den = aa * z2 + bb + cc * z1 * z2 + dd * z1;
            s[[0, 0, f]] = (aa * z2 + bb - cc * z1 * z2 - dd * z1) / den;
            s[[0, 1, f]] = two * sqrt_zz * (aa * dd - bb * cc) / den;
            s[[1, 0, f]] = two * sqrt_zz / den;
            s[[1, 1, f]] = (-aa * z2 + bb - cc * z1 * z2 + dd * z1) / den;
        }
    }
    Ok(s)
}

/// `sqrt(Z1 * Z2)` for the asymmetric ABCD normalisation.
///
/// For equal references the value is returned unchanged, so the scalar case
/// reproduces `s2a.m`/`a2s.m` exactly (no round trip through `sqrt`).
fn geometric_mean(z1: C64, z2: C64) -> C64 {
    if z1 == z2 {
        z1
    } else {
        (z1 * z2).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Z <-> H and Y <-> G (2-port)
// ---------------------------------------------------------------------------

/// Z to hybrid parameters:
/// `h11 = det(Z)/Z22, h12 = Z12/Z22, h21 = -Z21/Z22, h22 = 1/Z22`.
fn z_to_h(z: &Array3<C64>) -> Result<Array3<C64>, String> {
    require_two_port(z, "z_to_h")?;
    let nf = z.shape()[2];
    let mut h = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let z11 = z[[0, 0, f]];
        let z12 = z[[0, 1, f]];
        let z21 = z[[1, 0, f]];
        let z22 = z[[1, 1, f]];
        let det_z = z11 * z22 - z12 * z21;
        h[[0, 0, f]] = det_z / z22;
        h[[0, 1, f]] = z12 / z22;
        h[[1, 0, f]] = -z21 / z22;
        h[[1, 1, f]] = C64::new(1.0, 0.0) / z22;
    }
    Ok(h)
}

/// Hybrid to Z parameters:
/// `z11 = det(H)/H22, z12 = H12/H22, z21 = -H21/H22, z22 = 1/H22`.
fn h_to_z(h: &Array3<C64>) -> Result<Array3<C64>, String> {
    require_two_port(h, "h_to_z")?;
    let nf = h.shape()[2];
    let mut z = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let h11 = h[[0, 0, f]];
        let h12 = h[[0, 1, f]];
        let h21 = h[[1, 0, f]];
        let h22 = h[[1, 1, f]];
        let det_h = h11 * h22 - h12 * h21;
        z[[0, 0, f]] = det_h / h22;
        z[[0, 1, f]] = h12 / h22;
        z[[1, 0, f]] = -h21 / h22;
        z[[1, 1, f]] = C64::new(1.0, 0.0) / h22;
    }
    Ok(z)
}

/// Y to inverse-hybrid parameters:
/// `g11 = det(Y)/Y22, g12 = Y12/Y22, g21 = -Y21/Y22, g22 = 1/Y22`.
fn y_to_g(y: &Array3<C64>) -> Result<Array3<C64>, String> {
    require_two_port(y, "y_to_g")?;
    let nf = y.shape()[2];
    let mut g = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let y11 = y[[0, 0, f]];
        let y12 = y[[0, 1, f]];
        let y21 = y[[1, 0, f]];
        let y22 = y[[1, 1, f]];
        let det_y = y11 * y22 - y12 * y21;
        g[[0, 0, f]] = det_y / y22;
        g[[0, 1, f]] = y12 / y22;
        g[[1, 0, f]] = -y21 / y22;
        g[[1, 1, f]] = C64::new(1.0, 0.0) / y22;
    }
    Ok(g)
}

/// Inverse-hybrid to Y parameters:
/// `y11 = det(G)/G22, y12 = G12/G22, y21 = -G21/G22, y22 = 1/G22`.
fn g_to_y(g: &Array3<C64>) -> Result<Array3<C64>, String> {
    require_two_port(g, "g_to_y")?;
    let nf = g.shape()[2];
    let mut y = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let g11 = g[[0, 0, f]];
        let g12 = g[[0, 1, f]];
        let g21 = g[[1, 0, f]];
        let g22 = g[[1, 1, f]];
        let det_g = g11 * g22 - g12 * g21;
        y[[0, 0, f]] = det_g / g22;
        y[[0, 1, f]] = g12 / g22;
        y[[1, 0, f]] = -g21 / g22;
        y[[1, 1, f]] = C64::new(1.0, 0.0) / g22;
    }
    Ok(y)
}

/// Scattering to hybrid parameters (2-port): `s -> z -> h`.
pub fn s2h(s: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    require_two_port(s, "s2h")?;
    let z = s2z(s, ref_z)?;
    z_to_h(&z)
}

/// Hybrid to scattering parameters (2-port): `h -> z -> s`.
pub fn h2s(h: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    require_two_port(h, "h2s")?;
    let z = h_to_z(h)?;
    z2s(&z, ref_z)
}

/// Scattering to inverse-hybrid parameters (2-port): `s -> y -> g`.
pub fn s2g(s: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    require_two_port(s, "s2g")?;
    let y = s2y(s, ref_z)?;
    y_to_g(&y)
}

/// Inverse-hybrid to scattering parameters (2-port): `g -> y -> s`.
pub fn g2s(g: &Array3<C64>, ref_z: impl Into<RefImpedance>) -> Result<Array3<C64>, String> {
    require_two_port(g, "g2s")?;
    let y = g_to_y(g)?;
    y2s(&y, ref_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn two_port(values: &[[C64; 2]; 2]) -> Array3<C64> {
        let mut a = Array3::zeros((2, 2, 1));
        for i in 0..2 {
            for j in 0..2 {
                a[[i, j, 0]] = values[i][j];
            }
        }
        a
    }

    fn three_port_2f() -> Array3<C64> {
        let mut s = Array3::zeros((3, 3, 2));
        let f0 = [
            [C64::new(0.1, 0.2), C64::new(0.3, -0.1), C64::new(0.0, 0.05)],
            [
                C64::new(0.2, 0.0),
                C64::new(-0.15, 0.4),
                C64::new(0.01, 0.0),
            ],
            [
                C64::new(0.0, 0.3),
                C64::new(0.1, 0.0),
                C64::new(0.25, -0.05),
            ],
        ];
        let f1 = [
            [
                C64::new(0.11, 0.21),
                C64::new(0.31, -0.11),
                C64::new(0.0, 0.051),
            ],
            [
                C64::new(0.21, 0.0),
                C64::new(-0.16, 0.41),
                C64::new(0.011, 0.0),
            ],
            [
                C64::new(0.0, 0.31),
                C64::new(0.11, 0.0),
                C64::new(0.26, -0.06),
            ],
        ];
        for i in 0..3 {
            for j in 0..3 {
                s[[i, j, 0]] = f0[i][j];
                s[[i, j, 1]] = f1[i][j];
            }
        }
        s
    }

    fn assert_close(actual: &Array3<C64>, expected: &Array3<C64>, tol: f64) {
        assert_eq!(actual.shape(), expected.shape(), "shape mismatch");
        for ((i, j, f), value) in actual.indexed_iter() {
            let want = expected[[i, j, f]];
            let diff = (*value - want).norm();
            let scale = want.norm().max(1.0);
            assert!(
                diff <= tol * scale,
                "element ({i},{j},{f}): got {value}, expected {want}, |diff| {diff} > {}",
                tol * scale
            );
        }
    }

    #[test]
    fn the_previous_option_c64_api_still_works() {
        // The pre-RM-112 entry points took `Option<C64>`; that still compiles
        // and means "scalar reference, 50 Ohm when omitted".
        let s = three_port_2f();
        let explicit = s2z(&s, Some(C64::new(50.0, 0.0))).unwrap();
        let omitted = s2z(&s, None::<C64>).unwrap();
        let scalar = s2z(&s, 50.0).unwrap();
        assert_close(&explicit, &scalar, 0.0);
        assert_close(&omitted, &scalar, 0.0);

        let y = s2y(&s, Some(C64::new(75.0, 0.0))).unwrap();
        assert_close(&y, &s2y(&s, 75.0).unwrap(), 0.0);
    }

    #[test]
    fn per_port_round_trip_three_ports() {
        let s = three_port_2f();
        let z0 = RefImpedance::from(vec![50.0, 75.0, 60.0]);
        let z = s2z(&s, z0.clone()).unwrap();
        let y = s2y(&s, z0.clone()).unwrap();
        assert_close(&z2s(&z, z0.clone()).unwrap(), &s, 1e-12);
        assert_close(&y2s(&y, z0.clone()).unwrap(), &s, 1e-12);
        // z <-> y via matrix inverse round trips too
        assert_close(&y2z(&z).unwrap(), &y, 1e-11);
        assert_close(&z2y(&y).unwrap(), &z, 1e-11);
        // renormalising to the unchanged reference is a no-op
        assert_close(&s_renorm(&s, z0.clone(), z0).unwrap(), &s, 1e-12);
    }

    #[test]
    fn per_port_matches_scalar_when_all_ports_are_equal() {
        let s = three_port_2f();
        let scalar = s2z(&s, 50.0).unwrap();
        let vector = s2z(&s, RefImpedance::from(vec![50.0, 50.0, 50.0])).unwrap();
        assert_close(&scalar, &vector, 1e-15);
        let scalar_y = s2y(&s, 50.0).unwrap();
        let vector_y = s2y(&s, RefImpedance::from(vec![50.0, 50.0, 50.0])).unwrap();
        assert_close(&scalar_y, &vector_y, 1e-15);
    }

    #[test]
    fn reference_impedance_changes_the_result_per_port() {
        // Only the diagonal of Zref and G change, so a per-port reference must
        // differ from a 50 Ohm scalar reference.
        let s = three_port_2f();
        let a = s2z(&s, 50.0).unwrap();
        let b = s2z(&s, RefImpedance::from(vec![50.0, 75.0, 60.0])).unwrap();
        let diff: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0, f64::max);
        assert!(
            diff > 1e-3,
            "per-port reference had no effect (max diff {diff})"
        );
    }

    #[test]
    fn s_to_z_matches_the_hand_computed_one_port_case() {
        // One port: z = ((1+s)/(1-s)) * re(z0) ... with a real 50 Ohm
        // reference and G = 1/sqrt(50), Zref = 50:
        //   z = sqrt(50) * ((1+s)/(1-s)) * 50 * (1/sqrt(50)) = 50 (1+s)/(1-s)
        let mut s = Array3::zeros((1, 1, 1));
        s[[0, 0, 0]] = C64::new(0.0, 0.5);
        let z = s2z(&s, 50.0).unwrap();
        let expected = C64::new(50.0, 0.0) * (C64::new(1.0, 0.0) + s[[0, 0, 0]])
            / (C64::new(1.0, 0.0) - s[[0, 0, 0]]);
        assert!((z[[0, 0, 0]] - expected).norm() < 1e-12);
    }

    #[test]
    fn two_port_abcd_round_trip() {
        let s = two_port(&[
            [C64::new(0.1, 0.2), C64::new(0.3, -0.1)],
            [C64::new(0.2, 0.05), C64::new(-0.15, 0.4)],
        ]);
        let a = s2a(&s, 50.0).unwrap();
        assert_close(&a2s(&a, 50.0).unwrap(), &s, 1e-12);
        // per-port reference with equal values must reduce to the scalar case
        assert_close(
            &s2a(&s, RefImpedance::from(vec![50.0, 50.0])).unwrap(),
            &a,
            1e-15,
        );
    }

    #[test]
    fn abcd_of_a_series_impedance_is_the_pozar_matrix() {
        // A series impedance Z has S11 = S22 = Z/(Z+2 Z0), S21 = S12 = 2 Z0/(Z+2 Z0)
        let z_series = C64::new(10.0, 30.0);
        let z0 = 50.0;
        let s11 = z_series / (z_series + C64::new(2.0 * z0, 0.0));
        let s21 = C64::new(2.0 * z0, 0.0) / (z_series + C64::new(2.0 * z0, 0.0));
        let mut s = Array3::zeros((2, 2, 1));
        s[[0, 0, 0]] = s11;
        s[[1, 1, 0]] = s11;
        s[[1, 0, 0]] = s21;
        s[[0, 1, 0]] = s21;
        let a = s2a(&s, z0).unwrap();
        assert!((a[[0, 0, 0]] - C64::new(1.0, 0.0)).norm() < 1e-12);
        assert!((a[[1, 1, 0]] - C64::new(1.0, 0.0)).norm() < 1e-12);
        assert!((a[[0, 1, 0]] - z_series).norm() < 1e-12);
        assert!(a[[1, 0, 0]].norm() < 1e-12);
    }

    #[test]
    fn asymmetric_reference_abcd_matches_a_network_analysis() {
        // Series Zs followed by shunt Yp, Z1 = 50, Z2 = 75.
        let zs = C64::new(20.0, 10.0);
        let yp = C64::new(0.01, 0.005);
        let mut a = Array3::zeros((2, 2, 1));
        a[[0, 0, 0]] = C64::new(1.0, 0.0) + zs * yp;
        a[[0, 1, 0]] = zs;
        a[[1, 0, 0]] = yp;
        a[[1, 1, 0]] = C64::new(1.0, 0.0);
        let z1 = C64::new(50.0, 0.0);
        let z2 = C64::new(75.0, 0.0);
        let s = a2s(&a, RefImpedance::from(vec![50.0, 75.0])).unwrap();
        // Circuit analysis: with port 2 terminated in Z2, the input impedance is
        // Zin = Zs + (1/Yp || Z2) and S11 = (Zin - Z1)/(Zin + Z1).
        let z_par = C64::new(1.0, 0.0) / (yp + C64::new(1.0, 0.0) / z2);
        let zin = zs + z_par;
        let s11 = (zin - z1) / (zin + z1);
        assert!(
            (s[[0, 0, 0]] - s11).norm() < 1e-12,
            "S11 {} vs {}",
            s[[0, 0, 0]],
            s11
        );
        // And S11/S22 are consistent with the per-port round trip.
        let back = s2a(&s, RefImpedance::from(vec![50.0, 75.0])).unwrap();
        assert_close(&back, &a, 1e-12);
    }

    #[test]
    fn hybrid_and_inverse_hybrid_round_trip() {
        let s = two_port(&[
            [C64::new(0.1, 0.2), C64::new(0.3, -0.1)],
            [C64::new(0.2, 0.05), C64::new(-0.15, 0.4)],
        ]);
        let h = s2h(&s, 50.0).unwrap();
        assert_close(&h2s(&h, 50.0).unwrap(), &s, 1e-11);
        let g = s2g(&s, 50.0).unwrap();
        assert_close(&g2s(&g, 50.0).unwrap(), &s, 1e-11);
        // per-port references flow through both paths
        let z0 = RefImpedance::from(vec![50.0, 75.0]);
        assert_close(
            &h2s(&s2h(&s, z0.clone()).unwrap(), z0.clone()).unwrap(),
            &s,
            1e-11,
        );
        assert_close(&g2s(&s2g(&s, z0.clone()).unwrap(), z0).unwrap(), &s, 1e-11);
    }

    #[test]
    fn hybrid_parameters_match_their_definitions() {
        // H: [V1; I2] = H [I1; V2]; with Z known, h11 = det(Z)/z22,
        // h12 = z12/z22, h21 = -z21/z22, h22 = 1/z22.
        let z = two_port(&[
            [C64::new(10.0, 5.0), C64::new(2.0, -1.0)],
            [C64::new(3.0, 0.5), C64::new(20.0, -4.0)],
        ]);
        let s = z2s(&z, 50.0).unwrap();
        let h = s2h(&s, 50.0).unwrap();
        let z22 = z[[1, 1, 0]];
        let det = z[[0, 0, 0]] * z22 - z[[0, 1, 0]] * z[[1, 0, 0]];
        assert!((h[[0, 0, 0]] - det / z22).norm() < 1e-10);
        assert!((h[[0, 1, 0]] - z[[0, 1, 0]] / z22).norm() < 1e-10);
        assert!((h[[1, 0, 0]] + z[[1, 0, 0]] / z22).norm() < 1e-10);
        assert!((h[[1, 1, 0]] - C64::new(1.0, 0.0) / z22).norm() < 1e-10);
    }

    #[test]
    fn inverse_hybrid_parameters_match_their_definitions() {
        // G: [I1; V2] = G [V1; I2]; with Y known,
        // g11 = det(Y)/y22, g12 = y12/y22, g21 = -y21/y22, g22 = 1/y22.
        let y = two_port(&[
            [C64::new(0.02, 0.001), C64::new(-0.005, 0.0)],
            [C64::new(-0.004, 0.0005), C64::new(0.03, -0.002)],
        ]);
        let s = y2s(&y, 50.0).unwrap();
        let g = s2g(&s, 50.0).unwrap();
        let y22 = y[[1, 1, 0]];
        let det = y[[0, 0, 0]] * y22 - y[[0, 1, 0]] * y[[1, 0, 0]];
        assert!((g[[0, 0, 0]] - det / y22).norm() < 1e-10);
        assert!((g[[0, 1, 0]] - y[[0, 1, 0]] / y22).norm() < 1e-10);
        assert!((g[[1, 0, 0]] + y[[1, 0, 0]] / y22).norm() < 1e-10);
        assert!((g[[1, 1, 0]] - C64::new(1.0, 0.0) / y22).norm() < 1e-10);
    }

    #[test]
    fn s_renorm_matches_an_explicit_two_step_conversion() {
        let s = three_port_2f();
        let direct = s_renorm(&s, 50.0, 75.0).unwrap();
        let manual = z2s(&s2z(&s, 50.0).unwrap(), 75.0).unwrap();
        assert_close(&direct, &manual, 1e-15);
        // Renormalising to the same impedance returns the input.
        assert_close(&s_renorm(&s, 50.0, 50.0).unwrap(), &s, 1e-12);
        // Per-port to per-port.
        let from = RefImpedance::from(vec![50.0, 75.0, 60.0]);
        let to = RefImpedance::from(vec![75.0, 50.0, 50.0]);
        let s_new = s_renorm(&s, from.clone(), to.clone()).unwrap();
        assert_close(&s_renorm(&s_new, to, from).unwrap(), &s, 1e-11);
    }

    #[test]
    fn matlab_style_matrix_helpers_reject_bad_shapes() {
        let mut a = Array2::<C64>::zeros((2, 3));
        a[[0, 0]] = C64::new(1.0, 0.0);
        assert!(cplx_inv(&a).unwrap_err().contains("must be square"));
        let b = array![
            [C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, 0.0)]
        ];
        assert!(cplx_inv(&b).unwrap_err().contains("singular"));
    }

    #[test]
    fn inversion_threshold_scales_with_the_matrix() {
        // A tiny but well conditioned matrix must invert (Octave does).
        let mut tiny = Array2::<C64>::zeros((2, 2));
        tiny[[0, 0]] = C64::new(1e-20, 0.0);
        tiny[[1, 1]] = C64::new(2e-20, 0.0);
        let inv = cplx_inv(&tiny).unwrap();
        assert!((inv[[0, 0]] - C64::new(1e20, 0.0)).norm() < 1e6);
        assert!((inv[[1, 1]] - C64::new(5e19, 0.0)).norm() < 1e6);

        // A singular matrix with large entries is still rejected.
        let mut singular = Array2::<C64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                singular[[i, j]] = C64::new(1e6, 0.0);
            }
        }
        assert!(cplx_inv(&singular).unwrap_err().contains("singular"));

        // NaN entries do not panic.
        let mut nan = Array2::<C64>::zeros((2, 2));
        nan[[0, 0]] = C64::new(f64::NAN, 0.0);
        nan[[1, 1]] = C64::new(1.0, 0.0);
        assert!(cplx_inv(&nan).is_err());
    }

    #[test]
    fn shape_and_reference_errors_are_clear() {
        let mut s = Array3::zeros((2, 3, 1));
        s[[0, 0, 0]] = C64::new(0.0, 0.0);
        assert!(s2z(&s, 50.0).unwrap_err().contains("square"));
        let three = three_port_2f();
        assert!(s2a(&three, 50.0).unwrap_err().contains("2x2xF"));
        assert!(a2s(&three, 50.0).unwrap_err().contains("2x2xF"));
        assert!(s2h(&three, 50.0).unwrap_err().contains("2x2xF"));
        assert!(s2z(&three, RefImpedance::from(vec![50.0, 75.0]))
            .unwrap_err()
            .contains("2 entries"));
        let empty = Array3::<C64>::zeros((0, 0, 1));
        assert!(s2z(&empty, 50.0).unwrap_err().contains("at least one port"));
    }

    #[test]
    fn one_frequency_degenerate_matrices_do_not_panic() {
        // S21 == 0 makes the ABCD division by zero: the M-file yields
        // Inf/NaN silently, so must we (no panic, no error).
        let mut s = Array3::zeros((2, 2, 1));
        s[[0, 0, 0]] = C64::new(0.5, 0.0);
        s[[1, 1, 0]] = C64::new(0.5, 0.0);
        let a = s2a(&s, 50.0).unwrap();
        assert!(!a[[0, 0, 0]].re.is_finite() || a[[0, 0, 0]] == C64::new(0.0, 0.0));
    }
}
