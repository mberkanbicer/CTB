//! Network operations: `InitNetwork`, `AddElement2Port`, `ApplyCurrent2Port`,
//! `ApplyRFPower2Port` and `SetPortTermination`.
//!
//! Ported from `InitNetwork.m`, `AddElement2Port.m`, `ApplyCurrent2Port.m`,
//! `ApplyRFPower2Port.m` and `SetPortTermination.m`. A network keeps the
//! scattering, impedance and admittance matrices plus one ABCD matrix per port
//! (the network attached to that port), exactly like the `net` struct of the
//! M-files.

use crate::abcd::{a_mul, a_series, a_shunt};
use crate::transforms::{s2y, s2z, y2s, y2z, z2s, z2y};
use crate::{RefImpedance, C64};
use ndarray::{Array1, Array2, Array3};

/// A multi-port network with lumped elements attached to its ports.
#[derive(Clone, Debug)]
pub struct Network {
    /// Reference impedance (scalar or per port), `net.Z0` in the M-files.
    pub z0: RefImpedance,
    /// Number of frequency points.
    pub num_freq: usize,
    /// Number of ports.
    pub num_ports: usize,
    /// Frequency vector in Hz.
    pub f: Vec<f64>,
    /// Current scattering parameters.
    pub s: Array3<C64>,
    /// Scattering parameters the network was initialised with (`net.orig.s`).
    pub orig_s: Array3<C64>,
    /// Current impedance parameters.
    pub z: Array3<C64>,
    /// Current admittance parameters.
    pub y: Array3<C64>,
    /// ABCD matrix of the element chain attached to each port.
    pub abcd: Vec<Array3<C64>>,
}

impl Network {
    /// Per-port reference impedance of the current state.
    pub fn z0_per_port(&self) -> Result<Vec<C64>, String> {
        self.z0.resolve(self.num_ports)
    }

    /// The reference impedance of one port (1 based).
    pub fn port_impedance(&self, port: usize) -> Result<C64, String> {
        if port < 1 || port > self.num_ports {
            return Err(format!(
                "invalid port number {port}, the network has {} ports",
                self.num_ports
            ));
        }
        Ok(self.z0_per_port()?[port - 1])
    }
}

fn check_port(net: &Network, port: usize) -> Result<usize, String> {
    if port < 1 || port > net.num_ports {
        return Err(format!(
            "invalid port number {port}, the network has {} ports",
            net.num_ports
        ));
    }
    Ok(port - 1)
}

fn identity_abcd(num_freq: usize) -> Array3<C64> {
    let mut abcd = Array3::zeros((2, 2, num_freq));
    for f in 0..num_freq {
        abcd[[0, 0, f]] = C64::new(1.0, 0.0);
        abcd[[1, 1, f]] = C64::new(1.0, 0.0);
    }
    abcd
}

/// Initialise a network from a frequency vector and S-parameters.
///
/// `net.Z0 = 50`, `net.z = s2z(s, 50)`, `net.y = s2y(s, 50)` and an identity
/// ABCD matrix per port, as in `InitNetwork.m`.
pub fn init_network(freq: &[f64], s_para: &Array3<C64>) -> Result<Network, String> {
    init_network_with_ref(freq, s_para, RefImpedance::DEFAULT)
}

/// Initialise a network with an explicit reference impedance.
///
/// `InitNetwork.m` always uses 50 Ohm. Reading a Touchstone file with a scalar
/// or per-port `R`/`[Reference]` and initialising the network with that value is
/// the natural continuation, so the value can be passed here.
pub fn init_network_with_ref(
    freq: &[f64],
    s_para: &Array3<C64>,
    reference: impl Into<RefImpedance>,
) -> Result<Network, String> {
    let num_freq = freq.len();
    let num_ports = s_para.shape()[0];
    if s_para.shape()[1] != num_ports {
        return Err("n unequal m".to_string());
    }
    if s_para.shape()[2] != num_freq {
        return Err("number of frequencies do not match".to_string());
    }
    if num_ports == 0 {
        return Err("a network needs at least one port".to_string());
    }
    let reference = reference.into();
    reference.resolve(num_ports)?;

    let s = s_para.clone();
    let z = s2z(&s, &reference)?;
    let y = s2y(&s, &reference)?;
    let abcd = (0..num_ports).map(|_| identity_abcd(num_freq)).collect();

    Ok(Network {
        z0: reference,
        num_freq,
        num_ports,
        f: freq.to_vec(),
        orig_s: s.clone(),
        s,
        z,
        y,
        abcd,
    })
}

/// Resolve the value of a lumped element for every frequency.
///
/// `values` holds either one value (broadcast over all frequencies, like a
/// scalar M-file argument) or one value per frequency.
fn element_response(net: &Network, element_type: &str, values: &[C64]) -> Result<Element, String> {
    if values.is_empty() {
        return Err("no element value given".to_string());
    }
    if values.len() != 1 && values.len() != net.num_freq {
        return Err(format!(
            "{} element values for {} frequencies; give one value or one per frequency",
            values.len(),
            net.num_freq
        ));
    }
    let value_at = |f: usize| {
        if values.len() == 1 {
            values[0]
        } else {
            values[f]
        }
    };

    let lower = element_type.to_ascii_lowercase();
    if lower == "z_se" || lower == "y_sh" {
        let mut response = Array1::zeros(net.num_freq);
        for f in 0..net.num_freq {
            response[f] = value_at(f);
        }
        return Ok(if lower == "z_se" {
            Element::Series(response)
        } else {
            Element::Shunt(response)
        });
    }
    if !matches!(
        lower.as_str(),
        "c_se" | "c_sh" | "l_se" | "l_sh" | "r_se" | "r_sh"
    ) {
        return Err(format!("unknown element type '{element_type}'"));
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut response = Array1::zeros(net.num_freq);
    for f in 0..net.num_freq {
        let omega = C64::new(two_pi * net.f[f], 0.0);
        let value = value_at(f);
        response[f] = match lower.as_str() {
            // Z = 1/(2j*pi*f*C)
            "c_se" => C64::new(1.0, 0.0) / (C64::new(0.0, 1.0) * omega * value),
            // Y = 2j*pi*f*C
            "c_sh" => C64::new(0.0, 1.0) * omega * value,
            // Z = 2j*pi*f*L
            "l_se" => C64::new(0.0, 1.0) * omega * value,
            // Y = 1/(2j*pi*f*L)
            "l_sh" => C64::new(1.0, 0.0) / (C64::new(0.0, 1.0) * omega * value),
            // Z = R
            "r_se" => value,
            // Y = 1/R
            _ => C64::new(1.0, 0.0) / value,
        };
    }
    Ok(if lower.ends_with("_se") {
        Element::Series(response)
    } else {
        Element::Shunt(response)
    })
}

enum Element {
    Series(Array1<C64>),
    Shunt(Array1<C64>),
}

/// Add a linear network element to one port (`AddElement2Port.m`).
///
/// `element_type` is one of `C_se`, `C_sh`, `L_se`, `L_sh`, `R_se`, `R_sh`,
/// `Z_se`, `Y_sh` (case insensitive). `value` may be complex.
///
/// The M-file also accepts a frequency vector as the value (its examples build
/// `Z_se`/`Y_sh` that way); use [`add_element_2port_values`] for that.
pub fn add_element_2port(
    net: &mut Network,
    port: usize,
    element_type: &str,
    value: C64,
) -> Result<(), String> {
    add_element_2port_values(net, port, element_type, &[value])
}

/// Add a linear network element with one value per frequency
/// (`AddElement2Port.m` with a frequency-vector `value`).
pub fn add_element_2port_values(
    net: &mut Network,
    port: usize,
    element_type: &str,
    values: &[C64],
) -> Result<(), String> {
    let index = check_port(net, port)?;
    let element = element_response(net, element_type, values)?;
    match element {
        Element::Shunt(y) => {
            let abcd = a_mul(&[&net.abcd[index], &a_shunt(&y)])?;
            net.abcd[index] = abcd;
            for f in 0..net.num_freq {
                net.y[[index, index, f]] += y[f];
            }
            net.s = y2s(&net.y, &net.z0)?;
            net.z = y2z(&net.y)?;
        }
        Element::Series(z) => {
            let abcd = a_mul(&[&net.abcd[index], &a_series(&z)])?;
            net.abcd[index] = abcd;
            for f in 0..net.num_freq {
                net.z[[index, index, f]] += z[f];
            }
            net.s = z2s(&net.z, &net.z0)?;
            net.y = z2y(&net.z)?;
        }
    }
    Ok(())
}

/// Add the same element type to several ports (`AddElement2Port.m` with a port
/// vector).
///
/// `values` is broadcast when it holds a single entry; otherwise it must have
/// one entry per port.
pub fn add_element_2port_multi(
    net: &mut Network,
    ports: &[usize],
    element_type: &str,
    values: &[C64],
) -> Result<(), String> {
    if ports.is_empty() {
        return Err("no ports given".to_string());
    }
    if values.len() != 1 && values.len() != ports.len() {
        return Err(format!(
            "{} values for {} ports; give one value or one per port",
            values.len(),
            ports.len()
        ));
    }
    for (index, port) in ports.iter().enumerate() {
        let value = if values.len() == 1 {
            values[0]
        } else {
            values[index]
        };
        add_element_2port(net, *port, element_type, value)?;
    }
    Ok(())
}

/// Apply a chain of elements of different types to one port
/// (`AddElement2Port.m` with a cell-array type).
pub fn add_elements_2port(
    net: &mut Network,
    port: usize,
    elements: &[(&str, C64)],
) -> Result<(), String> {
    check_port(net, port)?;
    if elements.is_empty() {
        return Err("no elements given".to_string());
    }
    for (element_type, value) in elements {
        add_element_2port(net, port, element_type, *value)?;
    }
    Ok(())
}

/// Apply a total current to one port (`ApplyCurrent2Port.m`).
///
/// Returns `(I_port, I_orig_port)`, both `(ports x frequencies)`.
///
/// `current` holds the total current per frequency; a single value is broadcast
/// like `current*ones(size(net.f))` in the M-file.
pub fn apply_current_2port(
    net: &Network,
    port: usize,
    current: &Array1<C64>,
) -> Result<(Array2<C64>, Array2<C64>), String> {
    let index = check_port(net, port)?;
    if current.len() != 1 && current.len() != net.num_freq {
        return Err(format!(
            "{} current values for {} frequencies; give one value or one per frequency",
            current.len(),
            net.num_freq
        ));
    }
    let current_at = |f: usize| -> C64 {
        if current.len() == 1 {
            current[0]
        } else {
            current[f]
        }
    };

    let z0 = net.z0_per_port()?;

    // z_term = z + diag(Z0)
    let mut z_term = net.z.clone();
    for n in 0..net.num_ports {
        for f in 0..net.num_freq {
            z_term[[n, n, f]] += z0[n];
        }
    }

    let port_other: Vec<usize> = (0..net.num_ports).filter(|&p| p != index).collect();

    let mut i_port = Array2::zeros((net.num_ports, net.num_freq));
    for f in 0..net.num_freq {
        i_port[[index, f]] = current_at(f);
    }

    let n_red = port_other.len();
    if n_red > 0 {
        for f in 0..net.num_freq {
            let mut z_mat = Array2::zeros((n_red, n_red));
            for (i, &pi) in port_other.iter().enumerate() {
                for (j, &pj) in port_other.iter().enumerate() {
                    z_mat[[i, j]] = z_term[[pi, pj, f]];
                }
            }
            let mut rhs = Array2::zeros((n_red, 1));
            for (i, &pi) in port_other.iter().enumerate() {
                rhs[[i, 0]] = -z_term[[pi, index, f]];
            }
            let solution = crate::transforms::solve(&z_mat, &rhs)?;
            for i in 0..n_red {
                i_port[[port_other[i], f]] = solution[[i, 0]] * current_at(f);
            }
        }
    }

    // I_orig_port(n,:) = (Z0(n)*C + D) .* I_port(n,:)
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

/// Apply a current to several ports and sum the resulting currents
/// (`ApplyCurrent2Port.m` with a port vector).
pub fn apply_current_2port_multi(
    net: &Network,
    ports: &[usize],
    currents: &[Array1<C64>],
) -> Result<(Array2<C64>, Array2<C64>), String> {
    if ports.is_empty() {
        return Err("no ports given".to_string());
    }
    if currents.len() != 1 && currents.len() != ports.len() {
        return Err(format!(
            "{} current vectors for {} ports; give one or one per port",
            currents.len(),
            ports.len()
        ));
    }
    let mut i_port = Array2::zeros((net.num_ports, net.num_freq));
    let mut i_orig = Array2::zeros((net.num_ports, net.num_freq));
    for (index, port) in ports.iter().enumerate() {
        let current = if currents.len() == 1 {
            &currents[0]
        } else {
            &currents[index]
        };
        let (a, b) = apply_current_2port(net, *port, current)?;
        i_port += &a;
        i_orig += &b;
    }
    Ok((i_port, i_orig))
}

/// Apply an RF power to one port (`ApplyRFPower2Port.m`).
///
/// `I_tot = sqrt(2*|P|/Z0(port)) * (1 - S(port,port)) * exp(1j*angle(P))`, then
/// [`apply_current_2port`].
pub fn apply_rf_power_2port(
    net: &Network,
    port: usize,
    power: impl Into<C64>,
) -> Result<(Array2<C64>, Array2<C64>), String> {
    let index = check_port(net, port)?;
    let power = power.into();
    let z0 = net.z0_per_port()?;
    let amplitude = (2.0 * power.norm() / z0[index].re).sqrt();
    let phase = C64::from_polar(1.0, power.arg());

    let mut current = Array1::zeros(net.num_freq);
    for f in 0..net.num_freq {
        current[f] = amplitude * (C64::new(1.0, 0.0) - net.s[[index, index, f]]) * phase;
    }
    apply_current_2port(net, port, &current)
}

/// Apply an RF power to several ports and sum the resulting currents
/// (`ApplyRFPower2Port.m` with a port vector).
pub fn apply_rf_power_2port_multi(
    net: &Network,
    ports: &[usize],
    powers: &[C64],
) -> Result<(Array2<C64>, Array2<C64>), String> {
    if ports.is_empty() {
        return Err("no ports given".to_string());
    }
    if powers.len() != 1 && powers.len() != ports.len() {
        return Err(format!(
            "{} power values for {} ports; give one or one per port",
            powers.len(),
            ports.len()
        ));
    }
    let mut i_port = Array2::zeros((net.num_ports, net.num_freq));
    let mut i_orig = Array2::zeros((net.num_ports, net.num_freq));
    for (index, port) in ports.iter().enumerate() {
        let power = if powers.len() == 1 {
            powers[0]
        } else {
            powers[index]
        };
        let (a, b) = apply_rf_power_2port(net, *port, power)?;
        i_port += &a;
        i_orig += &b;
    }
    Ok((i_port, i_orig))
}

/// Set the termination impedance of one port (`SetPortTermination.m`).
///
/// The first non-matching termination expands a scalar `Z0` to a per-port
/// vector, and the scattering matrix is renormalised from the current impedance
/// parameters.
///
/// Native quirk kept on purpose: `SetPortTermination.m` updates `net.s` but
/// leaves `net.y` as it was, so `net.y` no longer corresponds to `net.s` after a
/// renormalisation. Only `net.z` is used to rebuild `net.s`, so the returned
/// currents are unaffected; a caller reading `net.y` directly (or adding a shunt
/// element afterwards) sees the stale matrix, exactly like in Octave.
pub fn set_port_termination(net: &mut Network, port: usize, impedance: C64) -> Result<(), String> {
    let index = check_port(net, port)?;

    match &net.z0 {
        RefImpedance::Scalar(current) => {
            if *current == impedance {
                return Ok(());
            }
            net.z0 = RefImpedance::PerPort(vec![*current; net.num_ports]);
        }
        RefImpedance::PerPort(_) => {}
    }

    if let RefImpedance::PerPort(values) = &mut net.z0 {
        if values[index] == impedance {
            return Ok(());
        }
        values[index] = impedance;
    }

    net.s = z2s(&net.z, &net.z0)?;
    Ok(())
}

/// Set the termination impedance of several ports
/// (`SetPortTermination.m` with a port vector).
pub fn set_port_termination_multi(
    net: &mut Network,
    ports: &[usize],
    impedances: &[C64],
) -> Result<(), String> {
    if ports.is_empty() {
        return Err("no ports given".to_string());
    }
    if impedances.len() != 1 && impedances.len() != ports.len() {
        return Err(format!(
            "{} impedances for {} ports; give one value or one per port",
            impedances.len(),
            ports.len()
        ));
    }
    for (index, port) in ports.iter().enumerate() {
        let impedance = if impedances.len() == 1 {
            impedances[0]
        } else {
            impedances[index]
        };
        set_port_termination(net, *port, impedance)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    fn series_network() -> Network {
        let freq = [1e6, 2e6, 3e6];
        let mut s = Array3::zeros((2, 2, 3));
        for (f, _) in freq.iter().enumerate() {
            s[[0, 0, f]] = c(0.1 * (f as f64 + 1.0), 0.05);
            s[[0, 1, f]] = c(0.7, -0.1 * f as f64);
            s[[1, 0, f]] = c(0.7, -0.1 * f as f64);
            s[[1, 1, f]] = c(-0.2, 0.1 * f as f64);
        }
        init_network(&freq, &s).unwrap()
    }

    #[test]
    fn init_network_sets_the_native_state() {
        let freq = [1e6, 2e6];
        let mut s = Array3::zeros((2, 2, 2));
        s[[0, 0, 0]] = c(0.1, 0.2);
        s[[1, 0, 0]] = c(0.3, 0.0);
        s[[0, 1, 0]] = c(0.3, 0.0);
        s[[1, 1, 0]] = c(0.4, -0.1);
        s[[0, 0, 1]] = c(0.11, 0.21);
        s[[1, 0, 1]] = c(0.31, 0.0);
        s[[0, 1, 1]] = c(0.31, 0.0);
        s[[1, 1, 1]] = c(0.41, -0.11);
        let net = init_network(&freq, &s).unwrap();
        assert_eq!(net.z0, RefImpedance::Scalar(c(50.0, 0.0)));
        assert_eq!(net.num_ports, 2);
        assert_eq!(net.num_freq, 2);
        assert_eq!(net.f, vec![1e6, 2e6]);
        assert_eq!(net.orig_s, net.s);
        for port in 0..2 {
            for f in 0..2 {
                assert_eq!(net.abcd[port][[0, 0, f]], c(1.0, 0.0));
                assert_eq!(net.abcd[port][[1, 1, f]], c(1.0, 0.0));
                assert_eq!(net.abcd[port][[0, 1, f]], c(0.0, 0.0));
                assert_eq!(net.abcd[port][[1, 0, f]], c(0.0, 0.0));
            }
        }
    }

    #[test]
    fn init_network_validates_its_arguments() {
        let mut s = Array3::zeros((2, 3, 2));
        s[[0, 0, 0]] = c(0.0, 0.0);
        assert!(init_network(&[1.0, 2.0], &s)
            .unwrap_err()
            .contains("n unequal m"));
        let s = Array3::zeros((2, 2, 3));
        assert!(init_network(&[1.0, 2.0], &s)
            .unwrap_err()
            .contains("number of frequencies"));
        let s = Array3::<C64>::zeros((0, 0, 1));
        assert!(init_network(&[1.0], &s)
            .unwrap_err()
            .contains("at least one port"));
        let s = Array3::zeros((2, 2, 1));
        assert!(
            init_network_with_ref(&[1.0], &s, RefImpedance::from(vec![50.0]))
                .unwrap_err()
                .contains("1 entries")
        );
    }

    /// A matched 6 dB attenuator (S11 = S22 = 0, S21 = S12 = 0.5) is a
    /// well conditioned two-port with a known input impedance of 50 Ohm.
    fn matched_attenuator() -> Network {
        let mut s = Array3::zeros((2, 2, 1));
        s[[0, 1, 0]] = c(0.5, 0.0);
        s[[1, 0, 0]] = c(0.5, 0.0);
        init_network(&[1e6], &s).unwrap()
    }

    #[test]
    fn series_resistor_on_a_matched_attenuator() {
        // Port 1 sees 50 (source) + 50 (series) + 50 (matched attenuator) = 100,
        // so S11 = 1/3; the divider gives S21 = 1/3.
        let mut net = matched_attenuator();
        add_element_2port(&mut net, 1, "R_se", c(50.0, 0.0)).unwrap();
        assert!((net.s[[0, 0, 0]] - c(1.0 / 3.0, 0.0)).norm() < 1e-12);
        assert!((net.s[[1, 0, 0]] - c(1.0 / 3.0, 0.0)).norm() < 1e-13);
        // The ABCD of port 1 now holds the series element.
        assert!((net.abcd[0][[0, 1, 0]] - c(50.0, 0.0)).norm() < 1e-12);
        // Port 2 was not touched.
        assert_eq!(net.abcd[1][[0, 1, 0]], c(0.0, 0.0));
        // z/y stay consistent with s.
        let z = z2s(&net.z, 50.0).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((net.s[[i, j, 0]] - z[[i, j, 0]]).norm() < 1e-15);
            }
        }
    }

    #[test]
    fn shunt_resistor_on_a_matched_attenuator() {
        // With 50 Ohm in parallel to the 50 Ohm input impedance of port 2 the
        // port sees 25 Ohm: S22 = (25-50)/(25+50) = -1/3.
        let mut net = matched_attenuator();
        let y_before = net.y.clone();
        add_element_2port(&mut net, 2, "R_sh", c(50.0, 0.0)).unwrap();
        assert!((net.s[[1, 1, 0]] - c(-1.0 / 3.0, 0.0)).norm() < 1e-12);
        // Only the diagonal entry of the excited port of Y grew by 1/R.
        assert!((net.y[[1, 1, 0]] - y_before[[1, 1, 0]] - c(1.0 / 50.0, 0.0)).norm() < 1e-15);
        assert_eq!(net.y[[0, 1, 0]], y_before[[0, 1, 0]]);
        assert_eq!(net.y[[0, 0, 0]], y_before[[0, 0, 0]]);
        assert_eq!(net.y[[1, 0, 0]], y_before[[1, 0, 0]]);
    }

    #[test]
    fn frequency_dependent_element_values_match_the_scalar_form() {
        // Z_se with the explicit 1/(2 pi f C) vector must equal C_se with C.
        let net = series_network();
        let c = 1e-12;
        let two_pi = 2.0 * std::f64::consts::PI;
        let z: Vec<C64> = net
            .f
            .iter()
            .map(|f| C64::new(1.0, 0.0) / (C64::new(0.0, 1.0) * C64::new(two_pi * f, 0.0) * c))
            .collect();

        let mut a = net.clone();
        add_element_2port(&mut a, 1, "C_se", C64::new(c, 0.0)).unwrap();
        let mut b = net.clone();
        add_element_2port_values(&mut b, 1, "Z_se", &z).unwrap();
        for f in 0..a.num_freq {
            for i in 0..a.num_ports {
                for j in 0..a.num_ports {
                    assert!((a.s[[i, j, f]] - b.s[[i, j, f]]).norm() < 1e-15);
                }
            }
            assert!((a.abcd[0][[0, 1, f]] - b.abcd[0][[0, 1, f]]).norm() < 1e-18);
        }

        // A single value is broadcast; an empty or mismatched vector is an error.
        let mut c1 = net.clone();
        add_element_2port_values(&mut c1, 1, "Z_se", &[C64::new(25.0, 5.0)]).unwrap();
        let mut c2 = net.clone();
        add_element_2port(&mut c2, 1, "Z_se", C64::new(25.0, 5.0)).unwrap();
        for f in 0..c1.num_freq {
            for i in 0..c1.num_ports {
                for j in 0..c1.num_ports {
                    assert!((c1.s[[i, j, f]] - c2.s[[i, j, f]]).norm() < 1e-15);
                }
            }
        }
        let mut d = net.clone();
        assert!(add_element_2port_values(&mut d, 1, "Z_se", &[])
            .unwrap_err()
            .contains("no element value"));
        assert!(
            add_element_2port_values(&mut d, 1, "Z_se", &[C64::new(1.0, 0.0); 2])
                .unwrap_err()
                .contains("2 element values")
        );
    }

    #[test]
    fn element_types_are_computed_from_the_frequency_vector() {
        let net0 = series_network();
        let f = 1e6;
        let omega = 2.0 * std::f64::consts::PI * f;
        type ElementCase = (&'static str, C64, fn(C64, f64) -> C64, bool);
        let cases: [ElementCase; 8] = [
            (
                "C_se",
                c(1e-12, 0.0),
                |v, w| c(1.0, 0.0) / (c(0.0, 1.0) * c(w, 0.0) * v),
                true,
            ),
            (
                "C_sh",
                c(1e-12, 0.0),
                |v, w| c(0.0, 1.0) * c(w, 0.0) * v,
                false,
            ),
            (
                "L_se",
                c(1e-9, 0.0),
                |v, w| c(0.0, 1.0) * c(w, 0.0) * v,
                true,
            ),
            (
                "L_sh",
                c(1e-9, 0.0),
                |v, w| c(1.0, 0.0) / (c(0.0, 1.0) * c(w, 0.0) * v),
                false,
            ),
            ("R_se", c(25.0, 0.0), |v, _| v, true),
            ("R_sh", c(25.0, 0.0), |v, _| c(1.0, 0.0) / v, false),
            ("Z_se", c(3.0, 4.0), |v, _| v, true),
            ("Y_sh", c(0.02, -0.01), |v, _| v, false),
        ];
        for (kind, value, expected, is_series) in cases {
            let mut net = net0.clone();
            add_element_2port(&mut net, 1, kind, value).unwrap();
            let want = expected(value, omega);
            let got = if is_series {
                net.abcd[0][[0, 1, 0]]
            } else {
                net.abcd[0][[1, 0, 0]]
            };
            assert!(
                (got - want).norm() < 1e-18 * want.norm().max(1.0),
                "{kind}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn invalid_ports_and_element_types_are_errors() {
        let mut net = series_network();
        assert!(add_element_2port(&mut net, 0, "R_se", c(50.0, 0.0))
            .unwrap_err()
            .contains("invalid port number"));
        assert!(add_element_2port(&mut net, 3, "R_se", c(50.0, 0.0))
            .unwrap_err()
            .contains("invalid port number"));
        assert!(add_element_2port(&mut net, 1, "X_se", c(50.0, 0.0))
            .unwrap_err()
            .contains("unknown element type"));
        assert!(add_element_2port_multi(&mut net, &[1, 2], "R_se", &[c(1.0, 0.0)]).is_ok());
        assert!(
            add_element_2port_multi(&mut net, &[1, 2], "R_se", &[c(1.0, 0.0); 3])
                .unwrap_err()
                .contains("3 values for 2 ports")
        );
        assert!(add_elements_2port(&mut net, 1, &[])
            .unwrap_err()
            .contains("no elements"));
    }

    #[test]
    fn multi_port_elements_match_repeated_single_calls() {
        let mut a = series_network();
        add_element_2port_multi(&mut a, &[1, 2], "C_se", &[c(1e-12, 0.0), c(2e-12, 0.0)]).unwrap();
        let mut b = series_network();
        add_element_2port(&mut b, 1, "C_se", c(1e-12, 0.0)).unwrap();
        add_element_2port(&mut b, 2, "C_se", c(2e-12, 0.0)).unwrap();
        for f in 0..a.num_freq {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((a.s[[i, j, f]] - b.s[[i, j, f]]).norm() < 1e-15);
                }
            }
        }
        // Broadcast value.
        let mut c1 = series_network();
        add_element_2port_multi(&mut c1, &[1, 2], "R_sh", &[c(50.0, 0.0)]).unwrap();
        let mut c2 = series_network();
        add_element_2port(&mut c2, 1, "R_sh", c(50.0, 0.0)).unwrap();
        add_element_2port(&mut c2, 2, "R_sh", c(50.0, 0.0)).unwrap();
        for f in 0..c1.num_freq {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((c1.s[[i, j, f]] - c2.s[[i, j, f]]).norm() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn element_chains_are_applied_in_order() {
        let mut a = series_network();
        add_elements_2port(
            &mut a,
            1,
            &[("C_se", c(1e-12, 0.0)), ("R_sh", c(75.0, 0.0))],
        )
        .unwrap();
        let mut b = series_network();
        add_element_2port(&mut b, 1, "C_se", c(1e-12, 0.0)).unwrap();
        add_element_2port(&mut b, 1, "R_sh", c(75.0, 0.0)).unwrap();
        for f in 0..a.num_freq {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((a.s[[i, j, f]] - b.s[[i, j, f]]).norm() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn current_application_solves_the_terminated_impedance_system() {
        // For a single excited port the reduced system is one equation:
        //   I_other = -Zterm[other, port] / Zterm[other, other] * I_port
        let net = series_network();
        let current = Array1::from_vec(vec![c(1e-3, 0.0)]);
        let (i_port, i_orig) = apply_current_2port(&net, 1, &current).unwrap();
        assert!((i_port[[0, 0]] - c(1e-3, 0.0)).norm() < 1e-18);
        let z21 = net.z[[1, 0, 0]];
        let z22 = net.z[[1, 1, 0]] + c(50.0, 0.0);
        let expected = -z21 / z22 * c(1e-3, 0.0);
        assert!(
            (i_port[[1, 0]] - expected).norm() < 1e-18,
            "{i1} vs {expected}",
            i1 = i_port[[1, 0]]
        );
        // Identity ABCD per port: I_orig == I_port.
        assert!((i_orig[[0, 0]] - i_port[[0, 0]]).norm() < 1e-15);
        assert!((i_orig[[1, 0]] - i_port[[1, 0]]).norm() < 1e-15);
        // Exciting port 2 instead swaps the roles.
        let (i_port2, _) = apply_current_2port(&net, 2, &current).unwrap();
        let expected2 = -net.z[[0, 1, 0]] / (net.z[[0, 0, 0]] + c(50.0, 0.0)) * c(1e-3, 0.0);
        assert!((i_port2[[0, 0]] - expected2).norm() < 1e-18);
    }

    #[test]
    fn current_application_broadcasts_a_scalar_and_rejects_wrong_lengths() {
        let net = series_network();
        let single = Array1::from_vec(vec![c(2e-3, 0.0)]);
        let per_freq = Array1::from_vec(vec![c(2e-3, 0.0), c(2e-3, 0.0), c(2e-3, 0.0)]);
        let (a, _) = apply_current_2port(&net, 1, &single).unwrap();
        let (b, _) = apply_current_2port(&net, 1, &per_freq).unwrap();
        for f in 0..net.num_freq {
            for n in 0..net.num_ports {
                assert!((a[[n, f]] - b[[n, f]]).norm() < 1e-15);
            }
        }
        let wrong = Array1::from_vec(vec![c(1.0, 0.0); 2]);
        assert!(apply_current_2port(&net, 1, &wrong)
            .unwrap_err()
            .contains("2 current values"));
        assert!(apply_current_2port(&net, 0, &single)
            .unwrap_err()
            .contains("invalid port number"));
    }

    #[test]
    fn multi_port_currents_are_summed() {
        let net = series_network();
        let one = Array1::from_vec(vec![c(1e-3, 0.0)]);
        let two = Array1::from_vec(vec![c(2e-3, 0.0)]);
        let (a, ao) =
            apply_current_2port_multi(&net, &[1, 2], &[one.clone(), two.clone()]).unwrap();
        let (b1, b1o) = apply_current_2port(&net, 1, &one).unwrap();
        let (b2, b2o) = apply_current_2port(&net, 2, &two).unwrap();
        for n in 0..net.num_ports {
            for f in 0..net.num_freq {
                assert!((a[[n, f]] - (b1[[n, f]] + b2[[n, f]])).norm() < 1e-18);
                assert!((ao[[n, f]] - (b1o[[n, f]] + b2o[[n, f]])).norm() < 1e-18);
            }
        }
        let empty: [Array1<C64>; 0] = [];
        assert!(apply_current_2port_multi(&net, &[1, 2], &empty)
            .unwrap_err()
            .contains("0 current vectors"));
        assert!(apply_current_2port_multi(&net, &[1], &[one.clone(), two])
            .unwrap_err()
            .contains("2 current vectors"));
    }

    #[test]
    fn rf_power_is_a_current_with_the_native_amplitude() {
        let net = series_network();
        let power = 1e-3;
        let z0 = net.port_impedance(1).unwrap().re;
        let (i_port, _) = apply_rf_power_2port(&net, 1, power).unwrap();
        let expected: Vec<C64> = (0..net.num_freq)
            .map(|f| c((2.0 * power / z0).sqrt(), 0.0) * (c(1.0, 0.0) - net.s[[0, 0, f]]))
            .collect();
        for (f, want) in expected.iter().enumerate() {
            assert!((i_port[[0, f]] - want).norm() < 1e-18);
        }
        // A negative power applies a 180 degree phase shift.
        let (i_neg, _) = apply_rf_power_2port(&net, 1, -power).unwrap();
        for f in 0..net.num_freq {
            assert!((i_neg[[0, f]] + i_port[[0, f]]).norm() < 1e-18);
        }
        // A complex power keeps its magnitude and shifts the phase.
        let complex = c(1e-3, 1e-3);
        let (i_complex, _) = apply_rf_power_2port(&net, 1, complex).unwrap();
        let amplitude_complex = (2.0 * complex.norm() / z0).sqrt();
        for f in 0..net.num_freq {
            let want = amplitude_complex
                * (c(1.0, 0.0) - net.s[[0, 0, f]])
                * C64::from_polar(1.0, complex.arg());
            assert!((i_complex[[0, f]] - want).norm() < 1e-18);
        }
    }

    #[test]
    fn multi_port_power_matches_the_sum_of_single_applications() {
        let net = series_network();
        let (a, _) = apply_rf_power_2port_multi(&net, &[1, 2], &[c(1e-3, 0.0); 2]).unwrap();
        let (b1, _) = apply_rf_power_2port(&net, 1, 1e-3).unwrap();
        let (b2, _) = apply_rf_power_2port(&net, 2, 1e-3).unwrap();
        for n in 0..net.num_ports {
            for f in 0..net.num_freq {
                assert!((a[[n, f]] - (b1[[n, f]] + b2[[n, f]])).norm() < 1e-18);
            }
        }
    }

    #[test]
    fn set_port_termination_renormalises_like_the_m_file() {
        let mut net = series_network();
        let before = net.s.clone();
        let z_old = net.z.clone();
        set_port_termination(&mut net, 2, c(75.0, 0.0)).unwrap();
        assert_eq!(
            net.z0,
            RefImpedance::PerPort(vec![c(50.0, 0.0), c(75.0, 0.0)])
        );
        // s = z2s(z, [50 75]) - z is unchanged by a renormalisation.
        for f in 0..net.num_freq {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((net.z[[i, j, f]] - z_old[[i, j, f]]).norm() < 1e-18);
                }
            }
        }
        let expected = z2s(&net.z, RefImpedance::from(vec![50.0, 75.0])).unwrap();
        for f in 0..net.num_freq {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((net.s[[i, j, f]] - expected[[i, j, f]]).norm() < 1e-15);
                }
            }
        }
        // Setting the same value again changes nothing.
        let snapshot = net.s.clone();
        set_port_termination(&mut net, 2, c(75.0, 0.0)).unwrap();
        assert_eq!(net.s, snapshot);
        // Setting it back to the original value restores the original S matrix.
        set_port_termination(&mut net, 2, c(50.0, 0.0)).unwrap();
        for f in 0..net.num_freq {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((net.s[[i, j, f]] - before[[i, j, f]]).norm() < 1e-12);
                }
            }
        }
        // The native quirk: net.y is not refreshed by SetPortTermination.
        let expected_y = s2y(&before, 50.0).unwrap();
        for f in 0..net.num_freq {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((net.y[[i, j, f]] - expected_y[[i, j, f]]).norm() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn set_port_termination_on_a_scalar_matching_value_is_a_no_op() {
        let mut net = series_network();
        set_port_termination(&mut net, 1, c(50.0, 0.0)).unwrap();
        // Still scalar: the M-file returns before expanding.
        assert_eq!(net.z0, RefImpedance::Scalar(c(50.0, 0.0)));
    }

    #[test]
    fn set_port_termination_multi_and_validation() {
        let mut net = series_network();
        set_port_termination_multi(&mut net, &[1, 2], &[c(60.0, 0.0), c(70.0, 0.0)]).unwrap();
        assert_eq!(
            net.z0,
            RefImpedance::PerPort(vec![c(60.0, 0.0), c(70.0, 0.0)])
        );
        assert!(set_port_termination(&mut net, 3, c(50.0, 0.0))
            .unwrap_err()
            .contains("invalid port number"));
        assert!(set_port_termination(&mut net, 1, c(-50.0, 0.0))
            .unwrap_err()
            .contains("positive real part"));
        assert!(
            set_port_termination_multi(&mut net, &[1, 2], &[c(1.0, 0.0); 3])
                .unwrap_err()
                .contains("3 impedances")
        );
        assert!(set_port_termination_multi(&mut net, &[], &[])
            .unwrap_err()
            .contains("no ports"));
    }

    #[test]
    fn per_port_termination_is_used_by_the_current_application() {
        let net0 = series_network();
        let mut net = net0.clone();
        set_port_termination(&mut net, 2, c(75.0, 0.0)).unwrap();
        let current = Array1::from_vec(vec![c(1e-3, 0.0)]);
        let (a, _) = apply_current_2port(&net, 1, &current).unwrap();
        // Port 2 is terminated in 75 instead of 50: the currents must differ.
        let (b, _) = apply_current_2port(&net0, 1, &current).unwrap();
        assert!((a[[1, 0]] - b[[1, 0]]).norm() > 1e-6);
    }
}
