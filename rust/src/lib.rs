//! CTB (Circuit Toolbox) ported to Rust.
//!
//! Network-parameter transforms and Touchstone read/write for RF circuit
//! analysis. Algorithms are ported from the CTB Matlab/Octave M-files
//! (`CTB/*.m`).
//!
//! # Layout
//!
//! * [`ref_impedance`] — scalar and per-port reference impedances.
//! * [`transforms`] — S/Z/Y/ABCD/H/G conversions and `s_renorm`.
//! * [`abcd`] — ABCD building blocks (`a_pi`, `a_t`, `a_series`, `a_shunt`,
//!   `a_mul`).
//! * [`touchstone`] — Touchstone v1/v2 reader and writer.
//! * [`network`] — `InitNetwork`/`AddElement2Port`/`ApplyCurrent2Port`/
//!   `ApplyRFPower2Port`/`SetPortTermination`.
//!
//! # Numerics
//!
//! Every transform keeps the native element-wise/left-right-division
//! formulation; all matrices are `Complex64` and shaped `(ports, ports,
//! frequencies)` exactly like the M-files.
//!
//! The M-files accept a scalar reference impedance or a per-port vector; that
//! is modelled by [`RefImpedance`]. The previous `Option<C64>` entry points keep
//! working because `Option<C64>` converts into [`RefImpedance`].
//!
//! # Example
//!
//! ```
//! use ctb_rs::{s2z, z2s, RefImpedance};
//! use ndarray::Array3;
//! use num_complex::Complex64;
//!
//! let mut s = Array3::<Complex64>::zeros((2, 2, 1));
//! s[[0, 1, 0]] = Complex64::new(0.5, 0.0); // S21
//! s[[1, 0, 0]] = Complex64::new(0.5, 0.0); // S12
//!
//! // Scalar reference impedance (the M-file default is 50 Ohm).
//! let z = s2z(&s, 50.0).unwrap();
//! let back = z2s(&z, 50.0).unwrap();
//! assert!((back[[0, 0, 0]] - s[[0, 0, 0]]).norm() < 1e-12);
//!
//! // Per-port reference impedances, as in `s2z(s, [50; 75])`.
//! let z0 = RefImpedance::from(vec![50.0, 75.0]);
//! let z = s2z(&s, z0.clone()).unwrap();
//! let back = z2s(&z, z0).unwrap();
//! assert!((back[[1, 0, 0]] - s[[1, 0, 0]]).norm() < 1e-12);
//! ```

use num_complex::Complex64;

/// Complex number type used throughout the crate (matches MATLAB doubles).
pub type C64 = Complex64;

pub mod abcd;
pub mod network;
pub mod ref_impedance;
pub mod touchstone;
pub mod transforms;

pub use abcd::{a_mul, a_pi, a_series, a_shunt, a_t};
pub use network::{
    add_element_2port, add_element_2port_multi, add_element_2port_values, add_elements_2port,
    apply_current_2port, apply_current_2port_multi, apply_rf_power_2port,
    apply_rf_power_2port_multi, init_network, init_network_with_ref, set_port_termination,
    set_port_termination_multi, Network,
};
pub use ref_impedance::RefImpedance;
pub use touchstone::{read_touchstone, write_touchstone, TouchstoneData, TwoPortDataOrder};
pub use transforms::{a2s, g2s, h2s, s2a, s2g, s2h, s2y, s2z, s_renorm, y2s, y2z, z2s, z2y};
