//! Integration tests for the `ctb` command line (RM-116 delivery form).

use std::path::{Path, PathBuf};
use std::process::Command;

use ctb_rs::touchstone::{read_touchstone, write_touchstone};
use ctb_rs::transforms::{s2z, s_renorm};
use ndarray::Array3;
use num_complex::Complex64;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_ctb")
}

fn temp_dir(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("ctb-cli-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&directory).ok();
    std::fs::create_dir_all(&directory).expect("temp dir");
    directory
}

fn sample_s() -> (Vec<f64>, Array3<Complex64>) {
    let freq = vec![1.0e9, 2.0e9];
    let mut s =
        Array3::from_shape_vec((2, 2, 2), vec![Complex64::new(0.0, 0.0); 8]).expect("shape");
    s[[0, 0, 0]] = Complex64::new(0.1, 0.2);
    s[[0, 1, 0]] = Complex64::new(0.7, -0.1);
    s[[1, 0, 0]] = Complex64::new(0.6, 0.05);
    s[[1, 1, 0]] = Complex64::new(-0.2, 0.3);
    s[[0, 0, 1]] = Complex64::new(0.05, -0.1);
    s[[0, 1, 1]] = Complex64::new(0.8, 0.0);
    s[[1, 0, 1]] = Complex64::new(0.75, 0.1);
    s[[1, 1, 1]] = Complex64::new(0.2, -0.4);
    (freq, s)
}

fn write_sample(directory: &Path) -> String {
    let (freq, s) = sample_s();
    let path = directory.join("input.s2p");
    write_touchstone(
        's',
        &freq,
        &s,
        path.to_str().expect("path"),
        None,
        Some("cli fixture"),
    )
    .expect("write fixture");
    path.display().to_string()
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("run the ctb binary")
}

#[test]
fn info_reports_the_file_contents() {
    let directory = temp_dir("info");
    let input = write_sample(&directory);
    let output = run(&["info", &input]);
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("parameter: s"), "{text}");
    assert!(text.contains("ports: 2"), "{text}");
    assert!(text.contains("frequencies: 2"), "{text}");
    assert!(text.contains("reference: 50"), "{text}");
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn convert_writes_the_requested_parameter_type() {
    let directory = temp_dir("convert");
    let input = write_sample(&directory);
    let output_path = directory.join("output.s2p");
    let output = run(&[
        "convert",
        "--to",
        "z",
        "--ref",
        "50,75",
        input.as_str(),
        output_path.to_str().expect("path"),
    ]);
    assert!(output.status.success(), "{output:?}");
    let converted = read_touchstone(output_path.to_str().expect("path")).expect("read z");
    assert_eq!(converted.param_type, 'z');
    assert_eq!(converted.num_ports(), 2);
    let (freq, s) = sample_s();
    assert_eq!(converted.freq, freq);
    let expected = s2z(&s, vec![50.0, 75.0]).expect("reference transform");
    for index in 0..expected.len() {
        let (i, j, f) = (
            index / expected.shape()[1] / expected.shape()[2],
            (index / expected.shape()[2]) % expected.shape()[1],
            index % expected.shape()[2],
        );
        let difference = (converted.data[[i, j, f]] - expected[[i, j, f]]).norm();
        // the native `.6e` Touchstone format keeps 7 significant digits
        let tolerance = 1e-6 * expected[[i, j, f]].norm().max(1.0);
        assert!(difference < tolerance, "cell {i},{j},{f}: {difference}");
    }
    // and back to S with the same per-port reference
    let back_path = directory.join("back.s2p");
    let output = run(&[
        "convert",
        "--to",
        "s",
        "--ref",
        "50,75",
        output_path.to_str().expect("path"),
        back_path.to_str().expect("path"),
    ]);
    assert!(output.status.success(), "{output:?}");
    let back = read_touchstone(back_path.to_str().expect("path")).expect("read s");
    for index in 0..s.len() {
        let (i, j, f) = (
            index / s.shape()[1] / s.shape()[2],
            (index / s.shape()[2]) % s.shape()[1],
            index % s.shape()[2],
        );
        let tolerance = 1e-6 * s[[i, j, f]].norm().max(1.0);
        assert!((back.data[[i, j, f]] - s[[i, j, f]]).norm() < tolerance);
    }
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn renorm_matches_the_library() {
    let directory = temp_dir("renorm");
    let input = write_sample(&directory);
    let output_path = directory.join("renorm.s2p");
    let output = run(&[
        "renorm",
        "--to",
        "75",
        input.as_str(),
        output_path.to_str().expect("path"),
    ]);
    assert!(output.status.success(), "{output:?}");
    let renormed = read_touchstone(output_path.to_str().expect("path")).expect("read");
    let (_, s) = sample_s();
    let expected = s_renorm(&s, 50.0, 75.0).expect("renormalise");
    for index in 0..expected.len() {
        let (i, j, f) = (
            index / expected.shape()[1] / expected.shape()[2],
            (index / expected.shape()[2]) % expected.shape()[1],
            index % expected.shape()[2],
        );
        let tolerance = 1e-6 * expected[[i, j, f]].norm().max(1.0);
        assert!(
            (renormed.data[[i, j, f]] - expected[[i, j, f]]).norm() < tolerance,
            "cell {i},{j},{f}"
        );
    }
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn invalid_invocations_fail_with_a_message() {
    let directory = temp_dir("errors");
    let input = write_sample(&directory);
    for args in [
        vec!["unknown"],
        vec!["convert", "--to", "q", input.as_str(), "out.s2p"],
        vec!["convert", "--to", "z", "/missing/file.s2p", "out.s2p"],
        vec!["renorm", "--to", "-50", input.as_str(), "out.s2p"],
        vec!["info"],
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "{args:?} must fail");
        let text = String::from_utf8_lossy(&output.stderr);
        assert!(!text.trim().is_empty(), "{args:?} has no error message");
    }
    let version = run(&["version"]);
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout).starts_with("ctb "));
    let _ = std::fs::remove_dir_all(&directory);
}
