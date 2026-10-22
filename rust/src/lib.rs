//! CTB (Circuit Toolbox) ported to Rust.
//!
//! Network-parameter transforms and Touchstone read/write for RF circuit analysis.
//! Algorithms are ported from the CTB Matlab/Octave M-files (CTB/*.m).

use num_complex::Complex64;
use ndarray::{Array1, Array2, Array3};
use std::fs::File;
use std::io::{BufRead, BufReader};

type C64 = Complex64;

// ---------------------------------------------------------------------------
// Matrix helpers (Gauss-Jordan complex inversion, left/right division)
// ---------------------------------------------------------------------------

/// Invert a complex square matrix using Gauss-Jordan elimination with partial pivoting.
fn cplx_inv(m: &Array2<C64>) -> Result<Array2<C64>, String> {
    let n = m.nrows();
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = m[[i, j]];
        }
        aug[[i, n + i]] = C64::new(1.0, 0.0);
    }

    for col in 0..n {
        // Partial pivoting: find row with largest |value| in this column
        let mut pivot_row = col;
        let mut max_val = aug[[col, col]].norm();
        for row in (col + 1)..n {
            let val = aug[[row, col]].norm();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(format!("Matrix is singular at column {}", col));
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }

        // Normalize pivot row
        let pivot = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] = aug[[col, j]] / pivot;
        }

        // Eliminate all other rows
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..2 * n {
                    aug[[row, j]] = aug[[row, j]] - factor * aug[[col, j]];
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

/// Matrix left division: A \ B = inv(A) * B.
fn mldivide(a: &Array2<C64>, b: &Array2<C64>) -> Result<Array2<C64>, String> {
    let a_inv = cplx_inv(a)?;
    Ok(a_inv.dot(b))
}

/// Matrix right division: A / B = A * inv(B).
fn mrdivide(a: &Array2<C64>, b: &Array2<C64>) -> Result<Array2<C64>, String> {
    let b_inv = cplx_inv(b)?;
    Ok(a.dot(&b_inv))
}

/// Build Zref = diag(z0) and G = diag(1/sqrt(real(z0))) for an N-port.
fn make_zref_g(n: usize, z0: &[C64]) -> (Array2<C64>, Array2<C64>) {
    let mut zref = Array2::zeros((n, n));
    let mut g = Array2::zeros((n, n));
    for i in 0..n {
        zref[[i, i]] = z0[i];
        g[[i, i]] = C64::new(1.0 / z0[i].re.sqrt(), 0.0);
    }
    (zref, g)
}

/// Extract an NxN matrix for frequency index f from an NxNxF array.
fn get_freq_matrix(arr: &Array3<C64>, f: usize) -> Array2<C64> {
    let n = arr.shape()[0];
    let mut out = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = arr[[i, j, f]];
        }
    }
    out
}

/// Insert an NxN matrix into an NxNxF array at frequency index f.
fn set_freq_matrix(arr: &mut Array3<C64>, f: usize, val: &Array2<C64>) {
    let n = arr.shape()[0];
    for i in 0..n {
        for j in 0..n {
            arr[[i, j, f]] = val[[i, j]];
        }
    }
}

// ---------------------------------------------------------------------------
// S-parameter transforms
// ---------------------------------------------------------------------------

/// S to Z parameters.
/// z = G \ ((E - s) \ (s + E)) * Zref * G   (per frequency)
pub fn s2z(s: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    let n = s.shape()[0];
    let nf = s.shape()[2];

    let z0_val = ref_z.unwrap_or(C64::new(50.0, 0.0));
    let z0 = vec![z0_val; n];

    let (zref, g) = make_zref_g(n, &z0);
    let g_inv = cplx_inv(&g)?;
    let e: Array2<C64> = Array2::eye(n);

    let mut z = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let s_f = get_freq_matrix(s, f);
        let tmp = mldivide(&(&e - &s_f), &(&s_f + &e))?; // (E - s) \ (s + E)
        let tmp = tmp.dot(&zref);                        // * Zref
        let tmp = tmp.dot(&g);                           // * G
        let z_f = mldivide(&g_inv, &tmp)?;               // G \ tmp
        set_freq_matrix(&mut z, f, &z_f);
    }
    Ok(z)
}

/// Z to S parameters.
/// s = G * (z - Zref) / (z + Zref) / G   (per frequency)
pub fn z2s(z: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    let n = z.shape()[0];
    let nf = z.shape()[2];

    let z0_val = ref_z.unwrap_or(C64::new(50.0, 0.0));
    let z0 = vec![z0_val; n];

    let (zref, g) = make_zref_g(n, &z0);
    let _g_inv = cplx_inv(&g)?;
    let _e: Array2<C64> = Array2::eye(n);

    let mut s = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let z_f = get_freq_matrix(z, f);
        let tmp = mrdivide(&(&z_f - &zref), &(&z_f + &zref))?; // (z - Zref) / (z + Zref)
        let tmp = g.dot(&tmp);                                   // G * tmp
        let s_f = mrdivide(&tmp, &g)?;                           // tmp / G
        set_freq_matrix(&mut s, f, &s_f);
    }
    Ok(s)
}

/// S to Y parameters.
/// y = G \ (Zref \ ((s + E) \ (E - s))) * G   (per frequency)
pub fn s2y(s: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    let n = s.shape()[0];
    let nf = s.shape()[2];

    let z0_val = ref_z.unwrap_or(C64::new(50.0, 0.0));
    let z0 = vec![z0_val; n];

    let (zref, g) = make_zref_g(n, &z0);
    let g_inv = cplx_inv(&g)?;
    let _zref_inv = cplx_inv(&zref)?;
    let e: Array2<C64> = Array2::eye(n);

    let mut y = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let s_f = get_freq_matrix(s, f);
        let tmp = mldivide(&(&s_f + &e), &(&e - &s_f))?; // (s + E) \ (E - s)
        let tmp = mldivide(&zref, &tmp)?;                 // Zref \ tmp
        let tmp = tmp.dot(&g);                            // * G
        let y_f = mldivide(&g_inv, &tmp)?;                // G \ tmp
        set_freq_matrix(&mut y, f, &y_f);
    }
    Ok(y)
}

/// Y to S parameters.
/// s = G * ((E - Zref*y) / (E + Zref*y)) / G   (per frequency)
pub fn y2s(y: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    let n = y.shape()[0];
    let nf = y.shape()[2];

    let z0_val = ref_z.unwrap_or(C64::new(50.0, 0.0));
    let z0 = vec![z0_val; n];

    let (zref, g) = make_zref_g(n, &z0);
    let _g_inv = cplx_inv(&g)?;
    let e: Array2<C64> = Array2::eye(n);

    let mut s = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let y_f = get_freq_matrix(y, f);
        let tmp = mrdivide(&(&e - &zref.dot(&y_f)), &(&e + &zref.dot(&y_f)))?; // (E - Zref*y) / (E + Zref*y)
        let tmp = g.dot(&tmp);                                                  // G * tmp
        let s_f = mrdivide(&tmp, &g)?;                                          // tmp / G
        set_freq_matrix(&mut s, f, &s_f);
    }
    Ok(s)
}

/// S to ABCD parameters (2x2 only).
/// Element-wise formulas from Pozar "Microwave Engineering".
pub fn s2a(s: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    if s.shape()[0] != 2 || s.shape()[1] != 2 {
        return Err("s2a requires a 2x2xF array".to_string());
    }
    let nf = s.shape()[2];
    let z0 = ref_z.unwrap_or(C64::new(50.0, 0.0));

    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let s11 = s[[0, 0, f]];
        let s12 = s[[0, 1, f]];
        let s21 = s[[1, 0, f]];
        let s22 = s[[1, 1, f]];

        let denom = C64::new(2.0, 0.0) * s21;
        a[[0, 0, f]] = ((C64::new(1.0, 0.0) + s11) * (C64::new(1.0, 0.0) - s22) + s12 * s21) / denom;
        a[[0, 1, f]] = z0 * ((C64::new(1.0, 0.0) + s11) * (C64::new(1.0, 0.0) + s22) - s12 * s21) / denom;
        a[[1, 0, f]] = (C64::new(1.0, 0.0) / z0) * ((C64::new(1.0, 0.0) - s11) * (C64::new(1.0, 0.0) - s22) - s12 * s21) / denom;
        a[[1, 1, f]] = ((C64::new(1.0, 0.0) - s11) * (C64::new(1.0, 0.0) + s22) + s12 * s21) / denom;
    }
    Ok(a)
}

/// ABCD to S parameters (2x2 only).
/// Element-wise formulas from Pozar "Microwave Engineering".
pub fn a2s(a: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    if a.shape()[0] != 2 || a.shape()[1] != 2 {
        return Err("a2s requires a 2x2xF array".to_string());
    }
    let nf = a.shape()[2];
    let z0 = ref_z.unwrap_or(C64::new(50.0, 0.0));

    let mut s = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let aa = a[[0, 0, f]];
        let bb = a[[0, 1, f]];
        let cc = a[[1, 0, f]];
        let dd = a[[1, 1, f]];

        let denom = aa + bb / z0 + cc * z0 + dd;
        s[[0, 0, f]] = (aa + bb / z0 - cc * z0 - dd) / denom;
        s[[0, 1, f]] = C64::new(2.0, 0.0) * (aa * dd - bb * cc) / denom;
        s[[1, 0, f]] = C64::new(2.0, 0.0) / denom;
        s[[1, 1, f]] = (-aa + bb / z0 - cc * z0 + dd) / denom;
    }
    Ok(s)
}

/// S to H parameters (2x2 only).
/// Converts S -> Z -> H.
pub fn s2h(s: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    if s.shape()[0] != 2 || s.shape()[1] != 2 {
        return Err("s2h requires a 2x2xF array".to_string());
    }
    let z = s2z(s, ref_z)?;
    z_to_h(&z)
}

/// H to S parameters (2x2 only).
/// Converts H -> Z -> S.
pub fn h2s(h: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    if h.shape()[0] != 2 || h.shape()[1] != 2 {
        return Err("h2s requires a 2x2xF array".to_string());
    }
    let z = h_to_z(h)?;
    z2s(&z, ref_z)
}

/// S to G parameters (2x2 only).
/// Converts S -> Y -> G.
pub fn s2g(s: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    if s.shape()[0] != 2 || s.shape()[1] != 2 {
        return Err("s2g requires a 2x2xF array".to_string());
    }
    let y = s2y(s, ref_z)?;
    y_to_g(&y)
}

/// G to S parameters (2x2 only).
/// Converts G -> Y -> S.
pub fn g2s(g: &Array3<C64>, ref_z: Option<C64>) -> Result<Array3<C64>, String> {
    if g.shape()[0] != 2 || g.shape()[1] != 2 {
        return Err("g2s requires a 2x2xF array".to_string());
    }
    let y = g_to_y(g)?;
    y2s(&y, ref_z)
}

/// Scattering matrix renormalization: s_new = s2z(s, Z) then z2s(z, Z_new).
pub fn s_renorm(s: &Array3<C64>, z: C64, z_new: C64) -> Result<Array3<C64>, String> {
    let z_params = s2z(s, Some(z))?;
    z2s(&z_params, Some(z_new))
}

// ---------------------------------------------------------------------------
// Z <-> H and Y <-> G conversions (2x2, per frequency)
// ---------------------------------------------------------------------------

fn z_to_h(z: &Array3<C64>) -> Result<Array3<C64>, String> {
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

fn h_to_z(h: &Array3<C64>) -> Result<Array3<C64>, String> {
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
        z[[1, 1, f]] = h11 / h22;
    }
    Ok(z)
}

fn y_to_g(y: &Array3<C64>) -> Result<Array3<C64>, String> {
    let nf = y.shape()[2];
    let mut g = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let y11 = y[[0, 0, f]];
        let y12 = y[[0, 1, f]];
        let y21 = y[[1, 0, f]];
        let y22 = y[[1, 1, f]];
        let det_y = y11 * y22 - y12 * y21;
        g[[0, 0, f]] = det_y / y22;
        g[[0, 1, f]] = -y12 / y22;
        g[[1, 0, f]] = y21 / y22;
        g[[1, 1, f]] = C64::new(1.0, 0.0) / y22;
    }
    Ok(g)
}

fn g_to_y(g: &Array3<C64>) -> Result<Array3<C64>, String> {
    let nf = g.shape()[2];
    let mut y = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        let g11 = g[[0, 0, f]];
        let g12 = g[[0, 1, f]];
        let g21 = g[[1, 0, f]];
        let g22 = g[[1, 1, f]];
        let det_g = g11 * g22 - g12 * g21;
        y[[0, 0, f]] = det_g / g22;
        y[[0, 1, f]] = -g12 / g22;
        y[[1, 0, f]] = g21 / g22;
        y[[1, 1, f]] = g11 / g22;
    }
    Ok(y)
}

/// Z to Y (matrix inverse per frequency).
pub fn z2y(z: &Array3<C64>) -> Result<Array3<C64>, String> {
    let n = z.shape()[0];
    let nf = z.shape()[2];
    let mut y = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let z_f = get_freq_matrix(z, f);
        let y_f = cplx_inv(&z_f)?;
        set_freq_matrix(&mut y, f, &y_f);
    }
    Ok(y)
}

/// Y to Z (matrix inverse per frequency).
pub fn y2z(y: &Array3<C64>) -> Result<Array3<C64>, String> {
    let n = y.shape()[0];
    let nf = y.shape()[2];
    let mut z = Array3::zeros((n, n, nf));
    for f in 0..nf {
        let y_f = get_freq_matrix(y, f);
        let z_f = cplx_inv(&y_f)?;
        set_freq_matrix(&mut z, f, &z_f);
    }
    Ok(z)
}

// ---------------------------------------------------------------------------
// ABCD operations
// ---------------------------------------------------------------------------

/// Pi network ABCD matrix.
/// a(1,1) = 1 + Y2/Y3, a(1,2) = 1/Y3, a(2,1) = Y1 + Y2 + Y1*Y2/Y3, a(2,2) = 1 + Y1/Y3
pub fn a_pi(y1: &Array1<C64>, y2: &Array1<C64>, y3: &Array1<C64>) -> Array3<C64> {
    let nf = y1.len();
    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        a[[0, 0, f]] = C64::new(1.0, 0.0) + y2[f] / y3[f];
        a[[0, 1, f]] = C64::new(1.0, 0.0) / y3[f];
        a[[1, 0, f]] = y1[f] + y2[f] + y1[f] * y2[f] / y3[f];
        a[[1, 1, f]] = C64::new(1.0, 0.0) + y1[f] / y3[f];
    }
    a
}

/// T network ABCD matrix.
/// a(1,1) = 1 + Z1/Z2, a(1,2) = Z1 + Z2 + Z1*Z2/Z3, a(2,1) = 1/Z3, a(2,2) = 1 + Z2/Z3
pub fn a_t(z1: &Array1<C64>, z2: &Array1<C64>, z3: &Array1<C64>) -> Array3<C64> {
    let nf = z1.len();
    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        a[[0, 0, f]] = C64::new(1.0, 0.0) + z1[f] / z2[f];
        a[[0, 1, f]] = z1[f] + z2[f] + z1[f] * z2[f] / z3[f];
        a[[1, 0, f]] = C64::new(1.0, 0.0) / z3[f];
        a[[1, 1, f]] = C64::new(1.0, 0.0) + z2[f] / z3[f];
    }
    a
}

/// Multiply multiple ABCD matrices (per frequency).
/// a = a1 * a2 * ... * aN
pub fn a_mul(arrays: &[&Array3<C64>]) -> Result<Array3<C64>, String> {
    if arrays.is_empty() {
        return Err("a_mul requires at least one argument".to_string());
    }
    let mut result = arrays[0].clone();
    for n in 1..arrays.len() {
        let a2 = arrays[n];
        let nf = result.shape()[2];
        let n = result.shape()[0];
        for f in 0..nf {
            let r_f = get_freq_matrix(&result, f);
            let a2_f = get_freq_matrix(a2, f);
            let prod = r_f.dot(&a2_f);
            set_freq_matrix(&mut result, f, &prod);
        }
        // Verify dimensions match
        if n != a2.shape()[0] {
            return Err(format!("Dimension mismatch in a_mul at argument {}", n + 1));
        }
    }
    Ok(result)
}

/// Series element ABCD matrix.
/// a(1,1) = 1, a(1,2) = Z, a(2,1) = 0, a(2,2) = 1
pub fn a_series(z: &Array1<C64>) -> Array3<C64> {
    let nf = z.len();
    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        a[[0, 0, f]] = C64::new(1.0, 0.0);
        a[[0, 1, f]] = z[f];
        a[[1, 0, f]] = C64::new(0.0, 0.0);
        a[[1, 1, f]] = C64::new(1.0, 0.0);
    }
    a
}

/// Shunt element ABCD matrix.
/// a(1,1) = 1, a(1,2) = 0, a(2,1) = Y, a(2,2) = 1
pub fn a_shunt(y: &Array1<C64>) -> Array3<C64> {
    let nf = y.len();
    let mut a = Array3::zeros((2, 2, nf));
    for f in 0..nf {
        a[[0, 0, f]] = C64::new(1.0, 0.0);
        a[[0, 1, f]] = C64::new(0.0, 0.0);
        a[[1, 0, f]] = y[f];
        a[[1, 1, f]] = C64::new(1.0, 0.0);
    }
    a
}

// ---------------------------------------------------------------------------
// Touchstone I/O
// ---------------------------------------------------------------------------

/// Touchstone file data.
pub struct TouchstoneData {
    pub param_type: char,   // 's', 'z', or 'y'
    pub freq: Vec<f64>,     // frequencies in Hz
    pub data: Array3<C64>,  // (num_ports x num_ports x num_freq)
    pub r#ref: C64,         // reference impedance
    pub comment: Vec<String>,
}

/// Read a Touchstone (.sNp) file.
pub fn read_touchstone(filename: &str) -> Result<TouchstoneData, String> {
    let file = File::open(filename).map_err(|e| format!("Failed to open {}: {}", filename, e))?;
    let reader = BufReader::new(file);

    let mut comment: Vec<String> = Vec::new();
    let mut param_type: Option<char> = None;
    let mut ref_impedance = C64::new(50.0, 0.0);
    let mut freq_mul: f64 = 1.0;
    let mut format_ri: Option<i32> = None; // 0=DB, 1=MA, 2=RI
    let num_ports: usize;
    let mut freq: Vec<f64> = Vec::new();
    let mut data_vec: Vec<C64> = Vec::new();

    // Determine number of ports from extension
    let lower = filename.to_lowercase();
    let last_dot = lower.rfind('.').ok_or_else(|| "filename has no extension".to_string())?;
    let extension = &lower[last_dot + 1..];
    if !(extension.starts_with('s') && extension.ends_with('p')) {
        return Err("filename must have an extension like .s2p".to_string());
    }
    let port_str = &extension[1..extension.len() - 1];
    num_ports = port_str.parse::<usize>().map_err(|_| "invalid port count in extension".to_string())?;

    let np = num_ports;
    let expected_values = 2 * np * np + 1; // freq + real/imag pairs

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        let tline = line.trim();

        if tline.is_empty() || tline.starts_with('!') {
            if !tline.is_empty() {
                comment.push(tline.to_string());
            }
            continue;
        }

        if tline.starts_with('#') {
            // Parameter line
            let mut tokens = tline.split_whitespace();
            tokens.next(); // skip '#'
            let unit = tokens.next().ok_or_else(|| "missing unit".to_string())?;
            freq_mul = match unit.to_lowercase().as_str() {
                "hz" => 1.0,
                "khz" => 1e3,
                "mhz" => 1e6,
                "ghz" => 1e9,
                _ => return Err(format!("wrong unit: {}", unit)),
            };
            let type_tok = tokens.next().ok_or_else(|| "missing parameter type".to_string())?;
            param_type = Some(type_tok.to_lowercase().chars().next().unwrap_or('s'));
            let fmt_tok = tokens.next().ok_or_else(|| "missing format".to_string())?;
            format_ri = Some(match fmt_tok.to_lowercase().as_str() {
                "ri" => 2,
                "ma" => 1,
                "db" => 0,
                _ => return Err(format!("wrong complex format: {}", fmt_tok)),
            });
            let _r = tokens.next(); // expect 'R'
            let ref_tok = tokens.next().ok_or_else(|| "missing reference impedance".to_string())?;
            ref_impedance = C64::new(ref_tok.parse::<f64>().map_err(|_| "invalid reference impedance".to_string())?, 0.0);
            continue;
        }

        // Data line: parse all whitespace-separated tokens
        let tokens: Vec<&str> = tline.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        // Collect values until we have enough for one data point
        // We need to handle multi-line data records
        // Simple approach: parse all tokens as numbers
        let mut vals: Vec<f64> = Vec::new();
        for tok in &tokens {
            if let Ok(v) = tok.parse::<f64>() {
                vals.push(v);
            }
        }

        // Process complete records
        let mut idx = 0;
        while idx + expected_values <= vals.len() {
            let f = vals[idx] * freq_mul;
            idx += 1;
            freq.push(f);

            let mut temp_data = Array2::zeros((np, np));
            for source in 0..np {
                for dest in 0..np {
                    let pos = if np == 2 {
                        // Special two-port format: S11, S21, S12, S22 order
                        ((source - 0) * 2 + (dest - 0)) * 2
                    } else {
                        ((source) * np + (dest)) * 2
                    };
                    // Adjust: pos is 1-based in Matlab, convert to 0-based
                    let pos0 = pos - 2; // because Matlab pos starts at 2 for first data value
                    let re = vals[idx + pos0];
                    let im = vals[idx + pos0 + 1];
                    let val = match format_ri.unwrap() {
                        2 => C64::new(re, im), // RI
                        1 => C64::from_polar(re, im.to_radians()), // MA
                        0 => C64::from_polar(10.0f64.powf(re / 20.0), im.to_radians()), // DB
                        _ => C64::new(re, im),
                    };
                    // Matlab uses temp_data(dest, source) - note the swap
                    temp_data[[dest, source]] = val;
                }
            }
            for i in 0..np {
                for j in 0..np {
                    data_vec.push(temp_data[[i, j]]);
                }
            }
            idx += 2 * np * np;
        }
    }

    let data = Array3::from_shape_vec((np, np, freq.len()), data_vec)
        .map_err(|e| format!("Failed to create data array: {}", e))?;

    Ok(TouchstoneData {
        param_type: param_type.unwrap_or('s'),
        freq,
        data,
        r#ref: ref_impedance,
        comment,
    })
}

/// Write a Touchstone file.
pub fn write_touchstone(
    param_type: char,
    freq: &[f64],
    data: &Array3<C64>,
    filename: &str,
    ref_z: Option<C64>,
    comment: Option<&str>,
) -> Result<(), String> {
    let np = data.shape()[0];
    if data.shape()[1] != np {
        return Err("number of rows and columns do not agree".to_string());
    }

    let (par_type, ref_val) = match param_type.to_lowercase().next().unwrap_or('s') {
        'z' => ('Z', ref_z.unwrap_or(C64::new(1.0, 0.0))),
        'y' => ('Y', ref_z.unwrap_or(C64::new(1.0, 0.0))),
        's' => ('S', ref_z.unwrap_or(C64::new(50.0, 0.0))),
        _ => return Err("only z-, y- and s-parameters supported".to_string()),
    };

    let comment_str = comment.unwrap_or("");

    let mut file = File::create(filename).map_err(|e| format!("Failed to create {}: {}", filename, e))?;

    use std::io::Write;
    writeln!(file, "#   Hz   {}  RI   R   {}", par_type, ref_val.re)
        .map_err(|e| format!("Write error: {}", e))?;
    writeln!(file, "! {}", comment_str).map_err(|e| format!("Write error: {}", e))?;

    if np == 1 || np >= 3 {
        write_data_1(&mut file, freq, data)?;
    } else if np == 2 {
        write_data_2(&mut file, freq, data)?;
    } else {
        return Err("unhandled dimension".to_string());
    }

    Ok(())
}

fn write_data_1(
    file: &mut File,
    freq: &[f64],
    data: &Array3<C64>,
) -> Result<(), String> {
    use std::io::Write;
    let np = data.shape()[0];
    let nf = freq.len();

    for f in 0..nf {
        write!(file, "{} ", freq[f]).map_err(|e| format!("Write error: {}", e))?;
        let mut column = 0;
        while column < np {
            let z_limit = (4usize).min(np - column);
            for _ in 0..z_limit {
                for row in 0..np {
                    let val = data[[row, column, f]];
                    write!(file, "{} {}   ", val.re, val.im)
                        .map_err(|e| format!("Write error: {}", e))?;
                }
                column += 1;
            }
            writeln!(file).map_err(|e| format!("Write error: {}", e))?;
            writeln!(file).map_err(|e| format!("Write error: {}", e))?;
        }
        writeln!(file).map_err(|e| format!("Write error: {}", e))?;
    }
    Ok(())
}

fn write_data_2(
    file: &mut File,
    freq: &[f64],
    data: &Array3<C64>,
) -> Result<(), String> {
    use std::io::Write;
    let nf = freq.len();
    for f in 0..nf {
        let d11 = data[[0, 0, f]];
        let d21 = data[[1, 0, f]];
        let d12 = data[[0, 1, f]];
        let d22 = data[[1, 1, f]];
        writeln!(
            file,
            "{}   {} {}   {} {}   {} {}   {} {}",
            freq[f],
            d11.re, d11.im,
            d21.re, d21.im,
            d12.re, d12.im,
            d22.re, d22.im,
        )
        .map_err(|e| format!("Write error: {}", e))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Network manipulation
// ---------------------------------------------------------------------------

/// Network state.
pub struct Network {
    pub z0: C64,
    pub num_freq: usize,
    pub num_ports: usize,
    pub f: Vec<f64>,
    pub s: Array3<C64>,
    pub orig_s: Array3<C64>,
    pub z: Array3<C64>,
    pub y: Array3<C64>,
    pub abcd: Vec<Array3<C64>>,
}

/// Initialize a network from frequency vector and S-parameters.
pub fn init_network(freq: &[f64], s_para: &Array3<C64>) -> Result<Network, String> {
    let num_freq = freq.len();
    let num_ports = s_para.shape()[0];

    if s_para.shape()[1] != num_ports {
        return Err("n unequal m".to_string());
    }
    if s_para.shape()[2] != num_freq {
        return Err("number of frequencies do not match".to_string());
    }

    let z0 = C64::new(50.0, 0.0);
    let s = s_para.clone();
    let orig_s = s_para.clone();
    let z = s2z(&s, Some(z0))?;
    let y = s2y(&s, Some(z0))?;

    // Identity ABCD for each port
    let mut abcd_template = Array3::zeros((2, 2, num_freq));
    for f in 0..num_freq {
        abcd_template[[0, 0, f]] = C64::new(1.0, 0.0);
        abcd_template[[1, 1, f]] = C64::new(1.0, 0.0);
    }
    let abcd: Vec<Array3<C64>> = (0..num_ports).map(|_| abcd_template.clone()).collect();

    Ok(Network {
        z0,
        num_freq,
        num_ports,
        f: freq.to_vec(),
        s,
        orig_s,
        z,
        y,
        abcd,
    })
}

/// Add a linear network element to a port.
pub fn add_element_2port(
    net: &mut Network,
    port: usize,
    element_type: &str,
    value: C64,
) -> Result<(), String> {
    if port < 1 || port > net.num_ports {
        return Err("invalid port number".to_string());
    }
    let port_idx = port - 1;

    let mut z_val: Option<Array1<C64>> = None;
    let mut y_val: Option<Array1<C64>> = None;

    let two_pi = 2.0 * std::f64::consts::PI;
    let _f_arr = Array1::from_vec(net.f.clone());

    match element_type.to_lowercase().as_str() {
        "c_se" => {
            // Series capacitor: Z = 1 / (2j*pi*f*C)
            let mut z = Array1::zeros(net.num_freq);
            for f in 0..net.num_freq {
                z[f] = C64::new(0.0, 0.0) - C64::new(0.0, 1.0) / (two_pi * net.f[f] * value.re);
            }
            z_val = Some(z);
        }
        "c_sh" => {
            // Shunt capacitor: Y = 2j*pi*f*C
            let mut y = Array1::zeros(net.num_freq);
            for f in 0..net.num_freq {
                y[f] = C64::new(0.0, 1.0) * (two_pi * net.f[f] * value.re);
            }
            y_val = Some(y);
        }
        "l_se" => {
            // Series inductor: Z = 2j*pi*f*L
            let mut z = Array1::zeros(net.num_freq);
            for f in 0..net.num_freq {
                z[f] = C64::new(0.0, 1.0) * (two_pi * net.f[f] * value.re);
            }
            z_val = Some(z);
        }
        "l_sh" => {
            // Shunt inductor: Y = 1 / (2j*pi*f*L)
            let mut y = Array1::zeros(net.num_freq);
            for f in 0..net.num_freq {
                y[f] = C64::new(0.0, 0.0) - C64::new(0.0, 1.0) / (two_pi * net.f[f] * value.re);
            }
            y_val = Some(y);
        }
        "r_se" => {
            // Series resistor: Z = R
            let mut z = Array1::zeros(net.num_freq);
            for f in 0..net.num_freq {
                z[f] = C64::new(value.re, 0.0);
            }
            z_val = Some(z);
        }
        "r_sh" => {
            // Shunt resistor: Y = 1/R
            let mut y = Array1::zeros(net.num_freq);
            for f in 0..net.num_freq {
                y[f] = C64::new(1.0 / value.re, 0.0);
            }
            y_val = Some(y);
        }
        "z_se" => {
            // Series impedance: Z = value (same for all frequencies)
            let mut z = Array1::zeros(net.num_freq);
            for f in 0..net.num_freq {
                z[f] = value;
            }
            z_val = Some(z);
        }
        "y_sh" => {
            // Shunt admittance: Y = value (same for all frequencies)
            let mut y = Array1::zeros(net.num_freq);
            for f in 0..net.num_freq {
                y[f] = value;
            }
            y_val = Some(y);
        }
        _ => return Err("unknown element type".to_string()),
    }

    if let Some(y) = y_val {
        let a_sh = a_shunt(&y);
        let abcd_mul = a_mul(&[&net.abcd[port_idx], &a_sh])?;
        net.abcd[port_idx] = abcd_mul;

        // Reshape Y to (1,1,nf) and add to diagonal
        for f in 0..net.num_freq {
            net.y[[port_idx, port_idx, f]] += y[f];
        }
        net.s = y2s(&net.y, Some(net.z0))?;
        net.z = y2z(&net.y)?;
    } else if let Some(z) = z_val {
        let a_se = a_series(&z);
        let abcd_mul = a_mul(&[&net.abcd[port_idx], &a_se])?;
        net.abcd[port_idx] = abcd_mul;

        for f in 0..net.num_freq {
            net.z[[port_idx, port_idx, f]] += z[f];
        }
        net.s = z2s(&net.z, Some(net.z0))?;
        net.y = z2y(&net.z)?;
    } else {
        return Err("internal error".to_string());
    }

    Ok(())
}

/// Apply current to a port.
/// Returns (I_port, I_orig_port) as (num_ports x num_freq) matrices.
pub fn apply_current_2port(
    net: &Network,
    port: usize,
    current: &Array1<C64>,
) -> Result<(Array2<C64>, Array2<C64>), String> {
    if port < 1 || port > net.num_ports {
        return Err("invalid port number".to_string());
    }
    let port_idx = port - 1;

    let z0 = if net.z0.re == 50.0 && net.z0.im == 0.0 {
        // Scalar Z0 - expand to per-port
        vec![net.z0; net.num_ports]
    } else {
        vec![net.z0; net.num_ports]
    };

    // z_term = z + diag(Z0)
    let mut z_term = net.z.clone();
    for n in 0..net.num_ports {
        for f in 0..net.num_freq {
            z_term[[n, n, f]] += z0[n];
        }
    }

    // Remove row/col for the excited port
    let port_other: Vec<usize> = (0..net.num_ports).filter(|&p| p != port_idx).collect();

    let mut i_port = Array2::zeros((net.num_ports, net.num_freq));
    for f in 0..net.num_freq {
        i_port[[port_idx, f]] = current[f];
    }

    // Build reduced matrix
    let n_red = net.num_ports - 1;
    if n_red > 0 {
        for f in 0..net.num_freq {
            let mut z_mat = Array2::zeros((n_red, n_red));
            for (i, &pi) in port_other.iter().enumerate() {
                for (j, &pj) in port_other.iter().enumerate() {
                    z_mat[[i, j]] = z_term[[pi, pj, f]];
                }
            }

            // z_port_vec = -z_term(port_other, port, f) as column vector
            let mut b_mat = Array2::zeros((n_red, 1));
            for (i, &pi) in port_other.iter().enumerate() {
                b_mat[[i, 0]] = -z_term[[pi, port_idx, f]];
            }

            // I_port(port_other, f) = z_mat \ b_mat * current(f)
            let sol = mldivide(&z_mat, &b_mat)?;
            for i in 0..n_red {
                i_port[[port_other[i], f]] = sol[[i, 0]] * current[f];
            }
        }
    }

    // I_orig_port
    let mut i_orig_port = Array2::zeros((net.num_ports, net.num_freq));
    for n in 0..net.num_ports {
        for f in 0..net.num_freq {
            let c = net.abcd[n][[1, 0, f]];
            let d = net.abcd[n][[1, 1, f]];
            i_orig_port[[n, f]] = (z0[n] * c + d) * i_port[[n, f]];
        }
    }

    Ok((i_port, i_orig_port))
}

/// Apply RF power to a port.
/// Returns (I_port, I_orig_port).
pub fn apply_rf_power_2port(
    net: &Network,
    port: usize,
    power: f64,
) -> Result<(Array2<C64>, Array2<C64>), String> {
    if port < 1 || port > net.num_ports {
        return Err("invalid port number".to_string());
    }
    let port_idx = port - 1;

    let z0_port = net.z0.re;

    // I_tot = sqrt(2*abs(power)/Z0(port)) * (1 - S(port,port,:)) * exp(1j*angle(power))
    let mut i_tot = Array1::zeros(net.num_freq);
    let amp = (2.0 * power.abs() / z0_port).sqrt();
    let phase_factor = if power < 0.0 { C64::new(-1.0, 0.0) } else { C64::new(1.0, 0.0) };

    for f in 0..net.num_freq {
        let s_pp = net.s[[port_idx, port_idx, f]];
        i_tot[f] = amp * (C64::new(1.0, 0.0) - s_pp) * phase_factor;
    }

    apply_current_2port(net, port, &i_tot)
}

/// Set port termination impedance.
pub fn set_port_termination(net: &mut Network, port: usize, impedance: C64) -> Result<(), String> {
    if port < 1 || port > net.num_ports {
        return Err("invalid port number".to_string());
    }

    if net.z0 == impedance {
        return Ok(()); // nothing needs to be done
    }

    net.z0 = impedance;
    net.s = z2s(&net.z, Some(net.z0))?;

    Ok(())
}
