//! Reference impedances for N-port network parameters.
//!
//! The CTB M-files take `ref` either as a scalar or as a per-port vector
//! (`[50;75;60]`). `s2z.m`, `s2y.m`, `z2s.m` and `y2s.m` expand a scalar with
//! `Z0*ones(N,1)` and then use `Z0(n)` per port; `s_renorm.m` forwards both
//! arguments unchanged, so per-port vectors work there as well. `s2a.m` and
//! `a2s.m` only support a scalar (see [`crate::transforms::s2a`] for the
//! generalisation used here).

use crate::C64;
use ndarray::Array1;

/// Reference impedance of an N-port network.
///
/// * [`RefImpedance::Scalar`] — one impedance for every port (the M-file
///   default is 50 Ohm).
/// * [`RefImpedance::PerPort`] — one impedance per port, in port order.
#[derive(Clone, Debug, PartialEq)]
pub enum RefImpedance {
    /// A single reference impedance for all ports.
    Scalar(C64),
    /// One reference impedance per port.
    PerPort(Vec<C64>),
}

impl Default for RefImpedance {
    fn default() -> Self {
        Self::Scalar(Self::DEFAULT)
    }
}

impl RefImpedance {
    /// Scalar reference impedance the M-files use when `ref` is omitted.
    pub const DEFAULT: C64 = C64::new(50.0, 0.0);

    /// Build a scalar reference impedance.
    pub fn scalar(z: impl Into<C64>) -> Self {
        Self::Scalar(z.into())
    }

    /// Build a per-port reference impedance.
    pub fn per_port(values: Vec<C64>) -> Self {
        Self::PerPort(values)
    }

    /// True for [`RefImpedance::Scalar`].
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }

    /// The value of a scalar reference impedance.
    pub fn as_scalar(&self) -> Option<C64> {
        match self {
            Self::Scalar(z) => Some(*z),
            Self::PerPort(_) => None,
        }
    }

    /// The shared value when the reference impedance is uniform.
    ///
    /// Returns `Some` for a scalar and for a per-port vector whose entries are
    /// all equal.
    pub fn uniform(&self) -> Option<C64> {
        match self {
            Self::Scalar(z) => Some(*z),
            Self::PerPort(values) => {
                let first = *values.first()?;
                if values.iter().all(|z| *z == first) {
                    Some(first)
                } else {
                    None
                }
            }
        }
    }

    /// Number of stored entries (1 for a scalar).
    pub fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::PerPort(values) => values.len(),
        }
    }

    /// Always false; a reference impedance always has at least one entry.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The impedance of port `index` (zero based).
    pub fn port(&self, index: usize) -> Result<C64, String> {
        let value = match self {
            Self::Scalar(z) => *z,
            Self::PerPort(values) => *values.get(index).ok_or_else(|| {
                format!(
                    "reference impedance has {} entries, port {} was requested",
                    values.len(),
                    index + 1
                )
            })?,
        };
        Ok(value)
    }

    /// Expand to one impedance per port and validate every entry.
    ///
    /// A scalar expands to `n_ports` copies, exactly like `Z0*ones(N,1)` in the
    /// M-files. A per-port vector must have exactly `n_ports` entries.
    ///
    /// The M-files take `G(n,n) = 1/sqrt(real(Z0(n)))` without checking the
    /// value, so a zero or negative real part silently produces `Inf`/`NaN`
    /// results. This conversion rejects such entries instead.
    pub fn resolve(&self, n_ports: usize) -> Result<Vec<C64>, String> {
        if n_ports == 0 {
            return Err("network has no ports".to_string());
        }
        let values = match self {
            Self::Scalar(z) => vec![*z; n_ports],
            Self::PerPort(values) => {
                if values.len() != n_ports {
                    return Err(format!(
                        "reference impedance has {} entries but the network has {} ports",
                        values.len(),
                        n_ports
                    ));
                }
                values.clone()
            }
        };
        for (index, z) in values.iter().enumerate() {
            validate(*z).map_err(|e| format!("port {}: {}", index + 1, e))?;
        }
        Ok(values)
    }
}

fn validate(z: C64) -> Result<(), String> {
    if !z.re.is_finite() || !z.im.is_finite() {
        return Err(format!("reference impedance must be finite, got {z}"));
    }
    if z.re <= 0.0 {
        return Err(format!(
            "reference impedance must have a positive real part, got {z}"
        ));
    }
    Ok(())
}

impl From<C64> for RefImpedance {
    fn from(z: C64) -> Self {
        Self::Scalar(z)
    }
}

impl From<f64> for RefImpedance {
    fn from(z: f64) -> Self {
        Self::Scalar(C64::new(z, 0.0))
    }
}

impl From<&RefImpedance> for RefImpedance {
    fn from(value: &RefImpedance) -> Self {
        value.clone()
    }
}

impl From<Vec<C64>> for RefImpedance {
    fn from(values: Vec<C64>) -> Self {
        Self::PerPort(values)
    }
}

impl From<Vec<f64>> for RefImpedance {
    fn from(values: Vec<f64>) -> Self {
        Self::PerPort(values.into_iter().map(|z| C64::new(z, 0.0)).collect())
    }
}

impl From<&[C64]> for RefImpedance {
    fn from(values: &[C64]) -> Self {
        Self::PerPort(values.to_vec())
    }
}

impl From<&[f64]> for RefImpedance {
    fn from(values: &[f64]) -> Self {
        Self::PerPort(values.iter().map(|z| C64::new(*z, 0.0)).collect())
    }
}

impl From<Array1<C64>> for RefImpedance {
    fn from(values: Array1<C64>) -> Self {
        Self::PerPort(values.to_vec())
    }
}

impl From<Array1<f64>> for RefImpedance {
    fn from(values: Array1<f64>) -> Self {
        Self::PerPort(values.iter().map(|z| C64::new(*z, 0.0)).collect())
    }
}

/// Keep the historical `Option<C64>` API working: `Some(z)` is a scalar
/// reference, `None` is the 50 Ohm default the M-files use when `ref` is not
/// given.
impl From<Option<C64>> for RefImpedance {
    fn from(value: Option<C64>) -> Self {
        Self::Scalar(value.unwrap_or(Self::DEFAULT))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_resolves_to_every_port() {
        let z = RefImpedance::from(50.0).resolve(3).unwrap();
        assert_eq!(z, vec![C64::new(50.0, 0.0); 3]);
        assert!(RefImpedance::from(50.0).is_scalar());
        assert_eq!(RefImpedance::from(50.0).len(), 1);
    }

    #[test]
    fn per_port_vector_is_kept_in_port_order() {
        let z = RefImpedance::from(vec![50.0, 75.0, 60.0])
            .resolve(3)
            .unwrap();
        assert_eq!(
            z,
            vec![
                C64::new(50.0, 0.0),
                C64::new(75.0, 0.0),
                C64::new(60.0, 0.0)
            ]
        );
        assert!(!RefImpedance::from(vec![50.0, 75.0, 60.0]).is_scalar());
        assert_eq!(
            RefImpedance::from(vec![50.0, 75.0]).port(1).unwrap(),
            C64::new(75.0, 0.0)
        );
    }

    #[test]
    fn per_port_length_must_match_the_port_count() {
        let err = RefImpedance::from(vec![50.0, 75.0]).resolve(3).unwrap_err();
        assert!(err.contains("2 entries"), "unexpected error: {err}");
        assert!(err.contains("3 ports"), "unexpected error: {err}");
    }

    #[test]
    fn non_positive_or_non_finite_impedances_are_rejected() {
        assert!(RefImpedance::from(0.0)
            .resolve(1)
            .unwrap_err()
            .contains("positive real part"));
        assert!(RefImpedance::from(-50.0)
            .resolve(1)
            .unwrap_err()
            .contains("positive real part"));
        assert!(RefImpedance::from(f64::NAN)
            .resolve(1)
            .unwrap_err()
            .contains("finite"));
        assert!(RefImpedance::from(vec![50.0, -1.0])
            .resolve(2)
            .unwrap_err()
            .contains("port 2"));
    }

    #[test]
    fn option_none_is_the_native_fifty_ohm_default() {
        assert_eq!(
            RefImpedance::from(None::<C64>),
            RefImpedance::Scalar(C64::new(50.0, 0.0))
        );
        assert_eq!(RefImpedance::default(), RefImpedance::from(50.0));
    }

    #[test]
    fn uniform_detects_equal_vectors() {
        assert_eq!(
            RefImpedance::from(vec![75.0, 75.0]).uniform(),
            Some(C64::new(75.0, 0.0))
        );
        assert_eq!(RefImpedance::from(vec![50.0, 75.0]).uniform(), None);
        assert_eq!(
            RefImpedance::from(50.0).uniform(),
            Some(C64::new(50.0, 0.0))
        );
    }

    #[test]
    fn complex_reference_impedances_are_allowed() {
        let z = C64::new(50.0, 5.0);
        assert_eq!(RefImpedance::from(z).resolve(2).unwrap(), vec![z, z]);
    }
}
