//! Differential tests of the CTB Rust port against the Octave M-files.
//!
//! Every fixture in `tests/data/` was produced by `tests/octave/gen_reference.m`
//! running the unmodified CTB M-files under GNU Octave 11.1.0. Regenerate with
//!
//! ```text
//! octave --no-gui --quiet CTB/rust/tests/octave/gen_reference.m
//! ```
//!
//! Tolerances
//! ----------
//! The transforms and the network operations solve dense complex linear systems;
//! this crate uses Gauss-Jordan elimination with partial pivoting while Octave
//! uses an LU factorisation, so the results can only agree to rounding error of
//! the elimination (no bitwise equality is possible). Every comparison therefore
//! uses the element-wise bound
//!
//! ```text
//! |actual - expected| <= tol * max(1, |expected|)
//! ```
//!
//! with `tol = 1e-12` for the direct transforms (well conditioned 2-/3-port
//! matrices, `cond < 1e3`) and `tol = 1e-11` for the network states, which are
//! the result of several chained inversions. `a_mul`, `a_series`, `a_shunt` and
//! `a_t` are element-wise and must agree to `1e-15`.
//!
//! The measured worst-case deviation of every case is printed with
//! `cargo test -- --nocapture`.

use ctb_rs::{
    a2s, a_mul, a_series, a_shunt, a_t, add_element_2port, add_element_2port_multi,
    add_elements_2port, apply_current_2port, apply_current_2port_multi, apply_rf_power_2port,
    init_network, read_touchstone, s2a, s2y, s2z, s_renorm, set_port_termination, write_touchstone,
    y2s, z2s, RefImpedance,
};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use std::path::PathBuf;

type C64 = Complex64;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
}

/// Load a fixture written by `ctb_ref_write_mat3`.
fn load(name: &str) -> Array3<C64> {
    let path = data_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("{} has no header", path.display()));
    let dims: Vec<usize> = header
        .split_whitespace()
        .map(|token| token.parse().expect("dimension"))
        .collect();
    assert_eq!(dims.len(), 3, "{}: bad header '{header}'", path.display());

    let mut array = Array3::zeros((dims[0], dims[1], dims[2]));
    for f in 0..dims[2] {
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                let line = lines
                    .next()
                    .unwrap_or_else(|| panic!("{} is truncated", path.display()));
                let mut tokens = line.split_whitespace();
                let re: f64 = tokens.next().expect("real part").parse().expect("real");
                let im: f64 = tokens.next().expect("imag part").parse().expect("imag");
                array[[i, j, f]] = C64::new(re, im);
            }
        }
    }
    assert!(
        lines.next().is_none(),
        "{} has extra values",
        path.display()
    );
    array
}

/// Load a fixture holding a real vector stored as a `1 x n x 1` matrix.
fn load_vector(name: &str) -> Vec<f64> {
    let array = load(name);
    assert_eq!(array.shape()[0], 1);
    assert_eq!(array.shape()[2], 1);
    array
        .indexed_iter()
        .map(|((_, _, _), value)| {
            assert_eq!(value.im, 0.0, "expected a real vector in {name}");
            value.re
        })
        .collect()
}

/// Compare against a fixture and return the worst element-wise deviation.
fn compare(name: &str, actual: &Array3<C64>, expected: &Array3<C64>, tol: f64) -> f64 {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{name}: shape {:?} != {:?}",
        actual.shape(),
        expected.shape()
    );
    let mut worst = 0.0;
    let mut worst_at = (0, 0, 0);
    for ((i, j, f), value) in actual.indexed_iter() {
        let want = expected[[i, j, f]];
        let deviation = (*value - want).norm() / want.norm().max(1.0);
        if deviation > worst {
            worst = deviation;
            worst_at = (i, j, f);
        }
    }
    println!("{name:<34} worst {worst:.3e} at {worst_at:?} (tol {tol:.0e})");
    assert!(
        worst <= tol,
        "{name}: worst deviation {worst:.3e} at {worst_at:?} exceeds {tol:.1e}\n  \
         actual = {}\n  expected = {}",
        actual[worst_at],
        expected[worst_at]
    );
    worst
}

fn three_port_freqs() -> Vec<f64> {
    vec![1e6, 2e6, 3e6]
}

// ---------------------------------------------------------------------------
// RM-112 — per-port reference impedances
// ---------------------------------------------------------------------------

#[test]
fn per_port_reference_impedance_transforms_match_octave() {
    let s = load("s3_input.txt");
    let z0 = RefImpedance::from(vec![50.0, 75.0, 60.0]);

    let z = s2z(&s, z0.clone()).unwrap();
    compare(
        "s2z 3-port [50;75;60]",
        &z,
        &load("z3_perport_50_75_60.txt"),
        1e-12,
    );

    let y = s2y(&s, z0.clone()).unwrap();
    compare(
        "s2y 3-port [50;75;60]",
        &y,
        &load("y3_perport_50_75_60.txt"),
        1e-12,
    );

    compare(
        "z2s(z, [50;75;60])",
        &z2s(&z, z0.clone()).unwrap(),
        &load("s3_roundtrip_z.txt"),
        1e-12,
    );
    compare(
        "y2s(y, [50;75;60])",
        &y2s(&y, z0.clone()).unwrap(),
        &load("s3_roundtrip_y.txt"),
        1e-12,
    );
    compare(
        "z2y(z)",
        &ctb_rs::z2y(&z).unwrap(),
        &load("y3_from_z3.txt"),
        1e-12,
    );

    compare(
        "s_renorm [50;75;60]->[75;50;50]",
        &s_renorm(
            &s,
            RefImpedance::from(vec![50.0, 75.0, 60.0]),
            RefImpedance::from(vec![75.0, 50.0, 50.0]),
        )
        .unwrap(),
        &load("s3_renorm_perport.txt"),
        1e-12,
    );
    compare(
        "s_renorm 50->75 (scalar)",
        &s_renorm(&s, 50.0, 75.0).unwrap(),
        &load("s3_renorm_scalar.txt"),
        1e-12,
    );

    // The scalar paths are pinned as well.
    compare(
        "s2z 3-port scalar 50",
        &s2z(&s, 50.0).unwrap(),
        &load("z3_scalar_50.txt"),
        1e-12,
    );
    compare(
        "s2y 3-port scalar 50",
        &s2y(&s, 50.0).unwrap(),
        &load("y3_scalar_50.txt"),
        1e-12,
    );

    // Round trip through the port order: the renormalised matrix must return to
    // the input when it is renormalised back (checked against the fixture
    // above), and a per-port vector with equal entries must equal the scalar
    // reference exactly.
    let equal = s2z(&s, RefImpedance::from(vec![50.0, 50.0, 50.0])).unwrap();
    let scalar = s2z(&s, 50.0).unwrap();
    compare("s2z [50;50;50] == 50", &equal, &scalar, 1e-15);
}

// ---------------------------------------------------------------------------
// RM-113 — 2-port ABCD and the ABCD building blocks
// ---------------------------------------------------------------------------

#[test]
fn two_port_abcd_transforms_match_octave() {
    let s = load("s2_input.txt");
    compare(
        "s2a 2-port ref 50",
        &s2a(&s, 50.0).unwrap(),
        &load("a2_from_s2_ref50.txt"),
        1e-12,
    );
    compare(
        "s2a 2-port ref 75",
        &s2a(&s, 75.0).unwrap(),
        &load("a2_from_s2_ref75.txt"),
        1e-12,
    );

    let a = load("a2_input.txt");
    compare(
        "a2s 2-port ref 50",
        &a2s(&a, 50.0).unwrap(),
        &load("s2_from_a2_ref50.txt"),
        1e-12,
    );
    compare(
        "a2s 2-port ref 75",
        &a2s(&a, 75.0).unwrap(),
        &load("s2_from_a2_ref75.txt"),
        1e-12,
    );
}

#[test]
fn abcd_blocks_match_octave() {
    // a_series / a_shunt / a_mul
    let series_z = Array1::from_vec(vec![
        C64::new(10.0, 20.0),
        C64::new(15.0, -5.0),
        C64::new(20.0, 0.0),
    ]);
    let shunt_y = Array1::from_vec(vec![
        C64::new(0.010, 0.002),
        C64::new(0.020, -0.001),
        C64::new(0.005, 0.003),
    ]);
    let a_series_fixture = load("a_series.txt");
    let a_shunt_fixture = load("a_shunt.txt");
    compare("a_series", &a_series(&series_z), &a_series_fixture, 0.0);
    compare("a_shunt", &a_shunt(&shunt_y), &a_shunt_fixture, 0.0);
    compare(
        "a_mul(a_series, a_shunt)",
        &a_mul(&[&a_series(&series_z), &a_shunt(&shunt_y)]).unwrap(),
        &load("a_mul.txt"),
        1e-15,
    );

    // a_t (native values, including the A(1,1) defect)
    let t_z1 = Array1::from_vec(vec![
        C64::new(10.0, 5.0),
        C64::new(20.0, -5.0),
        C64::new(30.0, 1.0),
    ]);
    let t_z2 = Array1::from_vec(vec![
        C64::new(5.0, 1.0),
        C64::new(1.0, 2.0),
        C64::new(2.0, -1.0),
    ]);
    let t_z3 = Array1::from_vec(vec![
        C64::new(50.0, 0.0),
        C64::new(30.0, -10.0),
        C64::new(40.0, 5.0),
    ]);
    compare(
        "a_t",
        &a_t(&t_z1, &t_z2, &t_z3).unwrap(),
        &load("a_t.txt"),
        1e-15,
    );
}

// ---------------------------------------------------------------------------
// RM-115 — network operations
// ---------------------------------------------------------------------------

#[test]
fn network_operations_match_octave() {
    let frequencies = three_port_freqs();
    let s = load("s3_input.txt");

    let mut net = init_network(&frequencies, &s).unwrap();
    compare("InitNetwork s", &net.s, &load("net_init_s.txt"), 1e-12);
    compare("InitNetwork z", &net.z, &load("net_init_z.txt"), 1e-11);
    compare("InitNetwork y", &net.y, &load("net_init_y.txt"), 1e-11);
    for n in 0..3 {
        compare(
            &format!("InitNetwork ABCD{{{}}}", n + 1),
            &net.abcd[n],
            &load(&format!("net_init_abcd_{}.txt", n + 1)),
            0.0,
        );
    }

    // Same element sequence as the generator.
    add_element_2port(&mut net, 2, "C_se", C64::new(1e-12, 0.0)).unwrap();
    add_element_2port_multi(&mut net, &[1, 2, 3], "R_se", &[C64::new(50.0, 0.0)]).unwrap();
    add_element_2port(&mut net, 3, "L_sh", C64::new(1e-9, 0.0)).unwrap();
    add_elements_2port(
        &mut net,
        2,
        &[
            ("C_sh", C64::new(2e-12, 0.0)),
            ("R_sh", C64::new(75.0, 0.0)),
        ],
    )
    .unwrap();
    compare("AddElement2Port s", &net.s, &load("net_add_s.txt"), 1e-11);
    compare("AddElement2Port z", &net.z, &load("net_add_z.txt"), 1e-11);
    compare("AddElement2Port y", &net.y, &load("net_add_y.txt"), 1e-11);
    for n in 0..3 {
        compare(
            &format!("AddElement2Port ABCD{{{}}}", n + 1),
            &net.abcd[n],
            &load(&format!("net_add_abcd_{}.txt", n + 1)),
            1e-15,
        );
    }

    // Currents and powers are applied to the network as it is after the element
    // chain, before the termination change.
    let one = Array1::from_vec(vec![C64::new(1e-3, 0.0)]);
    let (i_port, i_orig) = apply_current_2port(&net, 1, &one).unwrap();
    compare(
        "ApplyCurrent2Port I_port",
        &to_array3(&i_port),
        &load("current_single_i.txt"),
        1e-11,
    );
    compare(
        "ApplyCurrent2Port I_orig",
        &to_array3(&i_orig),
        &load("current_single_iorig.txt"),
        1e-11,
    );

    let currents = [
        Array1::from_vec(vec![C64::new(1e-3, 0.0)]),
        Array1::from_vec(vec![C64::new(2e-3, 0.0)]),
    ];
    let (i_multi, i_multi_orig) = apply_current_2port_multi(&net, &[1, 3], &currents).unwrap();
    compare(
        "ApplyCurrent2Port [1 3] I",
        &to_array3(&i_multi),
        &load("current_multi_i.txt"),
        1e-11,
    );
    compare(
        "ApplyCurrent2Port [1 3] I_orig",
        &to_array3(&i_multi_orig),
        &load("current_multi_iorig.txt"),
        1e-11,
    );

    let (p_port, p_orig) = apply_rf_power_2port(&net, 2, 1e-3).unwrap();
    compare(
        "ApplyRFPower2Port I_port",
        &to_array3(&p_port),
        &load("power_i.txt"),
        1e-11,
    );
    compare(
        "ApplyRFPower2Port I_orig",
        &to_array3(&p_orig),
        &load("power_iorig.txt"),
        1e-11,
    );

    set_port_termination(&mut net, 2, C64::new(75.0, 0.0)).unwrap();
    compare(
        "SetPortTermination s",
        &net.s,
        &load("net_term_s.txt"),
        1e-11,
    );
    compare(
        "SetPortTermination z",
        &net.z,
        &load("net_term_z.txt"),
        1e-15,
    );
    // Preserved native defect: net.y is not refreshed.
    compare(
        "SetPortTermination y (stale)",
        &net.y,
        &load("net_term_y.txt"),
        1e-11,
    );
    let z0 = net.z0_per_port().unwrap();
    let expected_z0 = load_vector("net_term_z0.txt");
    assert_eq!(z0.len(), expected_z0.len());
    for (actual, expected) in z0.iter().zip(expected_z0.iter()) {
        assert!(
            (actual.re - expected).abs() < 1e-15 && actual.im == 0.0,
            "Z0 {actual} != {expected}"
        );
    }
    // The M-file expands the scalar Z0 to a vector on the first change.
    assert!(matches!(net.z0, RefImpedance::PerPort(_)));
}

/// `(ports x frequencies)` matrices are stored as `ports x frequencies x 1`.
fn to_array3(matrix: &Array2<C64>) -> Array3<C64> {
    let mut array = Array3::zeros((matrix.nrows(), matrix.ncols(), 1));
    for ((i, j), value) in matrix.indexed_iter() {
        array[[i, j, 0]] = *value;
    }
    array
}

// ---------------------------------------------------------------------------
// RM-114 — Touchstone parity
// ---------------------------------------------------------------------------

struct TouchstoneCase {
    file: &'static str,
    param_type: char,
    reference: C64,
    comment: &'static str,
}

const TOUCHSTONE_CASES: &[TouchstoneCase] = &[
    TouchstoneCase {
        file: "native_2port.s2p",
        param_type: 's',
        reference: C64::new(50.0, 0.0),
        comment: "ctb differential 2-port",
    },
    TouchstoneCase {
        file: "native_3port.s3p",
        param_type: 's',
        reference: C64::new(50.0, 0.0),
        comment: "ctb differential 3-port",
    },
    TouchstoneCase {
        file: "native_1port_z.s1p",
        param_type: 'z',
        reference: C64::new(75.0, 0.0),
        comment: "one-port z",
    },
    TouchstoneCase {
        file: "native_4port.s4p",
        param_type: 's',
        reference: C64::new(50.0, 0.0),
        comment: "four port",
    },
    TouchstoneCase {
        file: "native_5port.s5p",
        param_type: 's',
        reference: C64::new(50.0, 0.0),
        comment: "five port",
    },
    TouchstoneCase {
        file: "native_complex_ref.s2p",
        param_type: 's',
        reference: C64::new(50.0, 5.0),
        comment: "complex reference",
    },
];

fn read_meta(name: &str) -> (char, u32, String, Vec<C64>, Vec<String>) {
    let path = data_dir().join(format!("{name}.meta.txt"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut param_type = '?';
    let mut version = 0;
    let mut order = String::new();
    let mut reference = Vec::new();
    let mut comments = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("type ") {
            param_type = rest.trim().chars().next().unwrap();
        } else if let Some(rest) = line.strip_prefix("version ") {
            version = rest.trim().parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("order ") {
            order = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("ref") {
            let values: Vec<f64> = rest
                .split_whitespace()
                .map(|token| token.parse().expect("reference impedance"))
                .collect();
            assert_eq!(values.len() % 2, 0, "reference impedances are re/im pairs");
            reference = values
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| C64::new(pair[0], pair[1]))
                .collect();
        } else if let Some(rest) = line.strip_prefix("comments ") {
            let count: usize = rest.trim().parse().unwrap();
            for _ in 0..count {
                comments.push(lines.next().expect("comment line").to_string());
            }
        }
    }
    (param_type, version, order, reference, comments)
}

#[test]
fn native_touchstone_files_are_read_like_octave() {
    for case in TOUCHSTONE_CASES {
        check_against_octave(case.file);
    }
}

/// Hand written v1 files the native writer never produces (dB/MA formats, other
/// units, comments, blank lines). `read_touchstone.m` is the reference.
#[test]
fn hand_written_v1_files_are_read_like_the_native_reader() {
    for file in [
        "hand_db_2port.s2p",
        "hand_ma_1port.s1p",
        "hand_ghz_comments.s2p",
    ] {
        check_against_octave(file);
    }
}

fn check_against_octave(file: &str) {
    let path = data_dir().join(file);
    let parsed = read_touchstone(path.to_str().unwrap()).unwrap_or_else(|e| panic!("{file}: {e}"));
    let (param_type, version, order, reference, comments) = read_meta(file);

    assert_eq!(parsed.param_type, param_type, "{file}");
    assert_eq!(parsed.version, version, "{file}");
    assert_eq!(parsed.two_port_data_order.as_str(), order, "{file}");
    assert_eq!(parsed.comment, comments, "{file}");

    let frequencies = load_vector(&format!("{file}.freq.txt"));
    assert_eq!(parsed.freq.len(), frequencies.len(), "{file}");
    for (actual, expected) in parsed.freq.iter().zip(frequencies.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "{file}: frequency {actual} != {expected}"
        );
    }

    let per_port = parsed.ref_per_port().unwrap();
    assert_eq!(per_port.len(), reference.len(), "{file}");
    for (actual, expected) in per_port.iter().zip(reference.iter()) {
        assert!(
            (actual - expected).norm() < 1e-12 * expected.norm().max(1.0),
            "{file}: reference {actual} != {expected}"
        );
    }

    let expected = load(&format!("{file}.data.txt"));
    compare(&format!("read {file}"), &parsed.data, &expected, 1e-15);
}

#[test]
fn written_touchstone_files_are_byte_identical_to_octave() {
    for case in TOUCHSTONE_CASES {
        let native_path = data_dir().join(case.file);
        let native = std::fs::read_to_string(&native_path).unwrap();
        let data = load(&format!("{}.input.txt", case.file));
        let frequencies = load_vector(&format!("{}.freq.txt", case.file));

        let ext = case.file.rsplit('.').next().unwrap();
        let stem = case.file.replace('.', "-");
        let out = std::env::temp_dir().join(format!(
            "ctb-rs-diff-{}-{}.{}",
            std::process::id(),
            stem,
            ext
        ));
        write_touchstone(
            case.param_type,
            &frequencies,
            &data,
            out.to_str().unwrap(),
            Some(RefImpedance::from(case.reference)),
            Some(case.comment),
        )
        .unwrap();
        let ours = std::fs::read_to_string(&out).unwrap();
        let _ = std::fs::remove_file(&out);
        assert_eq!(
            ours, native,
            "{}: Rust writer is not byte identical to write_touchstone.m",
            case.file
        );
        println!("{:<34} byte identical ({} bytes)", case.file, ours.len());
    }
}

#[test]
fn written_files_read_back_like_octave() {
    // Round trip through both readers: write with Rust, read with Rust, and
    // compare with what Octave reads from the native file.
    for case in TOUCHSTONE_CASES {
        let data = load(&format!("{}.input.txt", case.file));
        let frequencies = load_vector(&format!("{}.freq.txt", case.file));
        let expected = load(&format!("{}.data.txt", case.file));

        let out = std::env::temp_dir().join(format!(
            "ctb-rs-roundtrip-{}-{}",
            std::process::id(),
            case.file
        ));
        write_touchstone(
            case.param_type,
            &frequencies,
            &data,
            out.to_str().unwrap(),
            Some(RefImpedance::from(case.reference)),
            Some(case.comment),
        )
        .unwrap();
        let parsed = read_touchstone(out.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&out);
        compare(
            &format!("round trip {}", case.file),
            &parsed.data,
            &expected,
            1e-15,
        );
    }
}
