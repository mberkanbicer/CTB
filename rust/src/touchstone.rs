//! Touchstone (.sNp) reader and writer.
//!
//! The reader/writer pairs with `read_touchstone.m` / `write_touchstone.m`.
//!
//! # v1 (native) behaviour
//!
//! * The number of ports comes from the file extension (`.s2p` -> 2 ports).
//! * Lines starting with `!` are collected as comments (the leading `!` is kept,
//!   exactly like the M-file).
//! * One option line (`# <Hz|kHz|MHz|GHz> <S|Y|Z|G|H> <RI|MA|DB> R <value>`)
//!   sets the units, parameter type, data format and reference impedance.
//! * A data record is `frequency` plus `2*ports*ports` values and may be spread
//!   over several lines.
//! * Two-port records are ordered `S11, S21, S12, S22`; all other port counts
//!   are column major (`S11, S21, ..., Sn1, S12, ...`).
//!
//! # v2 keywords
//!
//! `[Version]`, `[Number of Ports]`, `[Two-Port Data Order]` (`21`/`12`),
//! `[Number of Frequencies]`, `[Number of Data Points]`, `[Reference]`
//! (per port), `[Network Data]`, `[Matrix Format] Full`, `[End]`.
//!
//! `[Mixed-Mode Order]` is rejected because mixed-mode data would be silently
//! misinterpreted as single-ended data. A `[Noise Data]` section ends the
//! network data (noise parameters are not modelled by CTB and are ignored).
//!
//! # Native defects that are preserved
//!
//! `write_touchstone.m` emits the values of a port-count >= 3 file row by row
//! (`data(row, column)` with `row` outer), while every Touchstone reader -
//! including `read_touchstone.m` - expects the column-major order. A 3-port file
//! written by the M-file therefore reads back transposed. The v1 writer here
//! reproduces the native byte stream exactly, so the same file is produced and
//! the same (mis)reading happens. v2 output (used for per-port reference
//! impedances, which v1 cannot express) is written in the standard column-major
//! order so that 3-port per-port-reference round trips are correct.

use crate::{RefImpedance, C64};
use ndarray::{Array2, Array3};
use std::fmt::Write as _;
use std::fs;

const FMT_DB: u8 = 0;
const FMT_MA: u8 = 1;
const FMT_RI: u8 = 2;

/// Value order of the data records of a 2-port file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TwoPortDataOrder {
    /// `S11, S21, S12, S22` (Touchstone default, `.s2p` v1 order).
    #[default]
    Order21,
    /// `S11, S12, S21, S22` (Touchstone v2 `[Two-Port Data Order] 12`).
    Order12,
}

impl TwoPortDataOrder {
    /// Keyword spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Order21 => "21",
            Self::Order12 => "12",
        }
    }
}

/// Contents of a Touchstone file.
#[derive(Clone, Debug)]
pub struct TouchstoneData {
    /// File format version: 1 (plain v1) or 2 (contains v2 keywords).
    pub version: u32,
    /// Parameter type, lowercase: `s`, `y`, `z`, `g` or `h`.
    pub param_type: char,
    /// Frequencies in Hz (already scaled by the option line unit).
    pub freq: Vec<f64>,
    /// `(ports x ports x frequencies)` parameter matrix.
    pub data: Array3<C64>,
    /// Reference impedance: scalar (`R <value>`) or per port (`[Reference]`).
    pub r#ref: RefImpedance,
    /// Data record order found in the file.
    pub two_port_data_order: TwoPortDataOrder,
    /// Comment lines, including the leading `!` (as in `read_touchstone.m`).
    pub comment: Vec<String>,
}

impl TouchstoneData {
    /// Number of ports of the file.
    pub fn num_ports(&self) -> usize {
        self.data.shape()[0]
    }

    /// Number of frequency points.
    pub fn num_freq(&self) -> usize {
        self.freq.len()
    }

    /// The reference impedance when it is the same on every port.
    pub fn ref_scalar(&self) -> Option<C64> {
        self.r#ref.uniform()
    }

    /// Per-port reference impedance (scalar references are expanded).
    pub fn ref_per_port(&self) -> Result<Vec<C64>, String> {
        self.r#ref.resolve(self.num_ports())
    }
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/// C `printf("%e")`: `[-]d.dddddde±dd`.
pub(crate) fn fmt_e(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let text = format!("{:.6e}", value);
    match text.split_once('e') {
        Some((mantissa, exponent)) => {
            let exponent: i32 = exponent.parse().unwrap_or(0);
            format!(
                "{}e{}{:02}",
                mantissa,
                if exponent < 0 { '-' } else { '+' },
                exponent.abs()
            )
        }
        None => text,
    }
}

/// C `printf("% e")`: like [`fmt_e`] but with a leading space for non-negative
/// values.
fn fmt_e_signed(value: f64) -> String {
    let text = fmt_e(value);
    if text.starts_with('-') {
        text
    } else {
        format!(" {text}")
    }
}

/// MATLAB `num2str` for one real component of a reference impedance field.
///
/// Shortest round-trip decimal (integral values print without a `.0`), which
/// matches `num2str` for the values that occur in practice (`50`, `75`, `1`,
/// `50.5`). `num2str` switches to exponent notation for `|x| < 1e-4`, this
/// function does not; the value is identical, only the spelling differs.
fn fmt_num(value: f64) -> String {
    format!("{value}")
}

/// MATLAB `num2str` for a (possibly complex) reference impedance.
fn fmt_impedance(value: C64) -> String {
    if value.im == 0.0 {
        fmt_num(value.re)
    } else if value.im > 0.0 {
        format!("{}+{}i", fmt_num(value.re), fmt_num(value.im))
    } else {
        format!("{}-{}i", fmt_num(value.re), fmt_num(-value.im))
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

struct OptionLine {
    freq_mul: f64,
    param_type: char,
    format: u8,
    reference: C64,
}

fn parse_number(token: &str, line_no: usize) -> Result<f64, String> {
    token
        .parse::<f64>()
        .map_err(|_| format!("line {line_no}: '{token}' is not a number"))
}

/// Parse a reference impedance token.
///
/// `str2double` in Octave (and therefore `read_touchstone.m`) also accepts the
/// complex forms `a+bi`, `a-bi` and `bi`. `R 50+5i` is a valid native
/// reference impedance, so it is accepted here as well.
fn parse_impedance(token: &str, line_no: usize) -> Result<C64, String> {
    if let Ok(value) = token.parse::<f64>() {
        return Ok(C64::new(value, 0.0));
    }
    let lower = token.to_ascii_lowercase();
    let body = lower
        .strip_suffix('i')
        .or_else(|| lower.strip_suffix('j'))
        .ok_or_else(|| format!("line {line_no}: '{token}' is not a reference impedance"))?;
    // The last sign that is not part of an exponent split, e.g. "1e-3+2e-4".
    let mut split = None;
    let bytes = body.as_bytes();
    for index in 1..body.len() {
        let ch = bytes[index] as char;
        if (ch == '+' || ch == '-') && !matches!(bytes[index - 1] as char, 'e' | 'E') {
            split = Some(index);
        }
    }
    match split {
        Some(index) => {
            let re = parse_number(&body[..index], line_no)?;
            let im = parse_number(&body[index..], line_no)?;
            Ok(C64::new(re, im))
        }
        None => {
            let im = parse_number(body, line_no)?;
            Ok(C64::new(0.0, im))
        }
    }
}

fn parse_option_line(line: &str, line_no: usize) -> Result<OptionLine, String> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 5 {
        return Err(format!(
            "line {line_no}: incomplete option line '{line}', expected \
             '# <Hz|kHz|MHz|GHz> <S|Y|Z|G|H> <RI|MA|DB> R <value>'"
        ));
    }
    let freq_mul = match tokens[1].to_ascii_lowercase().as_str() {
        "hz" => 1.0,
        "khz" => 1e3,
        "mhz" => 1e6,
        "ghz" => 1e9,
        other => {
            return Err(format!(
                "line {line_no}: wrong frequency unit '{other}', expected Hz, kHz, MHz or GHz"
            ))
        }
    };
    let param_type = match tokens[2].to_ascii_lowercase().as_str() {
        "s" => 's',
        "y" => 'y',
        "z" => 'z',
        "g" => 'g',
        "h" => 'h',
        other => {
            return Err(format!(
                "line {line_no}: wrong parameter type '{other}', expected S, Y, Z, G or H"
            ))
        }
    };
    let format = match tokens[3].to_ascii_lowercase().as_str() {
        "ri" => FMT_RI,
        "ma" => FMT_MA,
        "db" => FMT_DB,
        other => {
            return Err(format!(
                "line {line_no}: wrong complex format '{other}', expected RI, MA or DB"
            ))
        }
    };
    if !tokens[4].eq_ignore_ascii_case("r") {
        return Err(format!(
            "line {line_no}: option line is missing the reference impedance ('R <value>'), found '{}'",
            tokens[4]
        ));
    }
    let value_token = tokens.get(5).ok_or_else(|| {
        format!("line {line_no}: the option line has no reference impedance value")
    })?;
    let reference = parse_impedance(value_token, line_no)?;
    Ok(OptionLine {
        freq_mul,
        param_type,
        format,
        reference,
    })
}

fn parse_keyword(line: &str, line_no: usize) -> Result<(String, String), String> {
    let end = line
        .find(']')
        .ok_or_else(|| format!("line {line_no}: malformed keyword '{line}' (missing ']')"))?;
    let key = line[1..end].trim().to_ascii_lowercase();
    let rest = line[end + 1..].trim().to_string();
    Ok((key, rest))
}

fn ports_from_extension(filename: &str) -> Result<usize, String> {
    let lower = filename.to_lowercase();
    let name = lower.rsplit(['/', '\\']).next().unwrap_or(&lower);
    let dot = name
        .rfind('.')
        .ok_or_else(|| format!("filename has no extension: {filename}"))?;
    let extension = &name[dot + 1..];
    if extension.len() < 3 || !extension.starts_with('s') || !extension.ends_with('p') {
        return Err(format!(
            "filename must have an extension like .s2p, got '.{extension}'"
        ));
    }
    let digits = &extension[1..extension.len() - 1];
    let ports: usize = digits.parse().map_err(|_| {
        format!("'.{extension}' is not a Touchstone port-count extension like .s2p or .s3p")
    })?;
    if ports == 0 {
        return Err(format!("'.{extension}' declares zero ports"));
    }
    Ok(ports)
}

/// Read a Touchstone v1 or v2 file.
pub fn read_touchstone(filename: &str) -> Result<TouchstoneData, String> {
    let text =
        fs::read_to_string(filename).map_err(|e| format!("Failed to open {filename}: {e}"))?;
    parse_touchstone(filename, &text)
}

fn parse_touchstone(filename: &str, text: &str) -> Result<TouchstoneData, String> {
    let extension_ports = ports_from_extension(filename)?;

    let mut comment: Vec<String> = Vec::new();
    let mut option: Option<OptionLine> = None;
    let mut version = 1u32;
    let mut declared_ports: Option<usize> = None;
    let mut declared_frequencies: Option<usize> = None;
    let mut declared_points: Option<usize> = None;
    let mut declared_ref: Option<Vec<C64>> = None;
    let mut order = TwoPortDataOrder::Order21;
    let mut records: Vec<(f64, Array2<C64>)> = Vec::new();
    let mut pending: Vec<f64> = Vec::new();
    let mut pending_line = 0usize;

    let mut first_line = true;
    for (index, raw) in text.lines().enumerate() {
        let line_no = index + 1;
        let mut raw = raw;
        if first_line {
            // Tolerate a UTF-8 BOM instead of failing on it.
            raw = raw.trim_start_matches('\u{feff}');
            first_line = false;
        }
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('!') {
            if !pending.is_empty() {
                return Err(format!(
                    "line {line_no}: data record started on line {pending_line} is incomplete"
                ));
            }
            comment.push(format!("!{}", rest));
            continue;
        }
        if line.starts_with('#') {
            if !pending.is_empty() {
                return Err(format!(
                    "line {line_no}: data record started on line {pending_line} is incomplete"
                ));
            }
            let parsed = parse_option_line(line, line_no)?;
            // `read_touchstone.m` warns about additional parameter lines and
            // then keeps using the new one; do the same.
            option = Some(parsed);
            continue;
        }
        if line.starts_with('[') {
            if !pending.is_empty() {
                return Err(format!(
                    "line {line_no}: data record started on line {pending_line} is incomplete"
                ));
            }
            let (key, rest) = parse_keyword(line, line_no)?;
            version = version.max(2);
            match key.as_str() {
                "version" => {
                    let value =
                        parse_number(rest.split_whitespace().next().unwrap_or(""), line_no)?;
                    if !value.is_finite() || value < 1.0 {
                        return Err(format!(
                            "line {line_no}: unsupported Touchstone version {value}"
                        ));
                    }
                    if value < 2.0 {
                        version = 1;
                    }
                }
                "number of ports" => {
                    let ports = parse_usize(&rest, line_no, "number of ports")?;
                    if ports == 0 {
                        return Err(format!(
                            "line {line_no}: [Number of Ports] must be positive"
                        ));
                    }
                    declared_ports = Some(ports);
                }
                "two-port data order" => {
                    order = match rest.split_whitespace().next().unwrap_or("") {
                        "21" => TwoPortDataOrder::Order21,
                        "12" => TwoPortDataOrder::Order12,
                        other => {
                            return Err(format!(
                                "line {line_no}: unsupported [Two-Port Data Order] '{other}', \
                                 expected 21 or 12"
                            ))
                        }
                    };
                }
                "number of frequencies" => {
                    declared_frequencies =
                        Some(parse_usize(&rest, line_no, "number of frequencies")?);
                }
                "number of data points" => {
                    declared_points = Some(parse_usize(&rest, line_no, "number of data points")?);
                }
                "reference" => {
                    let mut values = Vec::new();
                    for token in rest.split_whitespace() {
                        values.push(parse_impedance(token, line_no)?);
                    }
                    if values.is_empty() {
                        return Err(format!("line {line_no}: [Reference] is empty"));
                    }
                    declared_ref = Some(values);
                }
                "matrix format" => {
                    let value = rest.split_whitespace().next().unwrap_or("");
                    if !value.eq_ignore_ascii_case("full") {
                        return Err(format!(
                            "line {line_no}: unsupported [Matrix Format] '{value}', only 'Full' is supported"
                        ));
                    }
                }
                "network data" | "interpolation" => {}
                "mixed-mode order" => {
                    return Err(format!(
                        "line {line_no}: mixed-mode Touchstone data is not supported"
                    ))
                }
                "noise data" | "number of noise frequencies" | "noise parameters" => {
                    // Noise parameters are not modelled by CTB; the network data
                    // section is complete at this point.
                    break;
                }
                "end" => break,
                other => {
                    return Err(format!(
                        "line {line_no}: unsupported Touchstone keyword '[{other}]'"
                    ));
                }
            }
            continue;
        }

        // Data line: numbers, one or more records, possibly continued below.
        if option.is_none() {
            return Err(format!(
                "line {line_no}: data before the '# ... R <value>' option line"
            ));
        }
        let opt = option.as_ref().unwrap();
        if pending.is_empty() {
            pending_line = line_no;
        }
        for token in line.split_whitespace() {
            if token.starts_with('!') {
                // Tolerate a trailing comment on a data line.
                break;
            }
            pending.push(parse_number(token, line_no)?);
        }

        let record_len = 1 + 2 * extension_ports * extension_ports;
        while pending.len() >= record_len {
            let record: Vec<f64> = pending.drain(..record_len).collect();
            let matrix = record_to_matrix(&record, extension_ports, order, opt.format);
            records.push((record[0] * opt.freq_mul, matrix));
        }
    }

    if !pending.is_empty() {
        return Err(format!(
            "line {pending_line}: incomplete data record, {} of {} values",
            pending.len(),
            1 + 2 * extension_ports * extension_ports
        ));
    }

    let option = option.ok_or_else(|| {
        format!("{filename}: missing the '# <unit> <type> <format> R <value>' option line")
    })?;

    if let Some(ports) = declared_ports {
        if ports != extension_ports {
            return Err(format!(
                "{filename}: [Number of Ports] {ports} does not match the '.s{extension_ports}p' extension"
            ));
        }
    }
    if records.is_empty() {
        return Err(format!("{filename}: no data records found"));
    }
    if let Some(count) = declared_points {
        if count != records.len() {
            return Err(format!(
                "{filename}: [Number of Data Points] {count} but {} records were read",
                records.len()
            ));
        }
    }
    if let Some(count) = declared_frequencies {
        if count != records.len() {
            return Err(format!(
                "{filename}: [Number of Frequencies] {count} but {} records were read",
                records.len()
            ));
        }
    }

    let ports = extension_ports;
    let reference = match declared_ref {
        Some(values) => {
            if values.len() > ports {
                return Err(format!(
                    "{filename}: [Reference] lists {} values for {ports} ports",
                    values.len()
                ));
            }
            let mut full = vec![option.reference; ports];
            full[..values.len()].copy_from_slice(&values);
            RefImpedance::PerPort(full)
        }
        None => RefImpedance::Scalar(option.reference),
    };
    // Validate the reference impedance the same way the transforms do.
    reference
        .resolve(ports)
        .map_err(|e| format!("{filename}: {e}"))?;

    let nf = records.len();
    let mut data = Array3::zeros((ports, ports, nf));
    for (f, (_, matrix)) in records.iter().enumerate() {
        for i in 0..ports {
            for j in 0..ports {
                data[[i, j, f]] = matrix[[i, j]];
            }
        }
    }
    let freq = records.into_iter().map(|(f, _)| f).collect();

    Ok(TouchstoneData {
        version,
        param_type: option.param_type,
        freq,
        data,
        r#ref: reference,
        two_port_data_order: order,
        comment,
    })
}

fn parse_usize(rest: &str, line_no: usize, what: &str) -> Result<usize, String> {
    let token = rest.split_whitespace().next().unwrap_or("");
    token
        .parse::<usize>()
        .map_err(|_| format!("line {line_no}: '{token}' is not a valid {what}"))
}

fn record_to_matrix(
    record: &[f64],
    ports: usize,
    order: TwoPortDataOrder,
    format: u8,
) -> Array2<C64> {
    let mut matrix = Array2::zeros((ports, ports));
    for k in 0..ports * ports {
        let (row, column) = if ports == 2 && order == TwoPortDataOrder::Order12 {
            (k / 2, k % 2)
        } else {
            (k % ports, k / ports)
        };
        let magnitude = record[1 + 2 * k];
        let phase = record[2 + 2 * k];
        let value = match format {
            FMT_RI => C64::new(magnitude, phase),
            FMT_MA => C64::from_polar(magnitude, phase / 180.0 * std::f64::consts::PI),
            _ => C64::from_polar(
                10.0f64.powf(magnitude / 20.0),
                phase / 180.0 * std::f64::consts::PI,
            ),
        };
        matrix[[row, column]] = value;
    }
    matrix
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a Touchstone file.
///
/// * `param_type` — `s`, `y`, `z`, `g` or `h` (the M-file only supports
///   `z`/`y`/`s`; `g`/`h` are accepted because the reader supports them).
/// * `freq` — frequencies in Hz.
/// * `data` — `(ports x ports x frequencies)`.
/// * `ref_z` — scalar or per-port reference impedance; `None` selects the
///   native default (1 Ohm for Z/Y, 50 Ohm otherwise).
/// * `comment` — written as `! <comment>`.
///
/// A scalar reference produces the native v1 layout (byte for byte, including
/// the row-major `>=3`-port defect described in the module docs). A per-port
/// reference produces a Touchstone v2 file, because v1 cannot express one.
pub fn write_touchstone(
    param_type: char,
    freq: &[f64],
    data: &Array3<C64>,
    filename: &str,
    ref_z: Option<RefImpedance>,
    comment: Option<&str>,
) -> Result<(), String> {
    let ports = data.shape()[0];
    if data.shape()[1] != ports {
        return Err("number of rows and columns do not agree".to_string());
    }
    if ports == 0 {
        return Err("a Touchstone file needs at least one port".to_string());
    }
    if freq.len() != data.shape()[2] {
        return Err(format!(
            "{} frequencies but {} data blocks",
            freq.len(),
            data.shape()[2]
        ));
    }

    let lower = param_type.to_ascii_lowercase();
    let par_type = match lower {
        's' => 'S',
        'y' => 'Y',
        'z' => 'Z',
        'g' => 'G',
        'h' => 'H',
        _ => {
            return Err(format!(
                "only s-, y-, z-, g- and h-parameters are supported, got '{param_type}'"
            ))
        }
    };
    let default_reference = if lower == 'z' || lower == 'y' {
        1.0
    } else {
        50.0
    };
    let reference = ref_z.unwrap_or_else(|| RefImpedance::from(default_reference));
    let per_port = reference.resolve(ports)?;

    let mut out = String::new();
    match reference {
        RefImpedance::Scalar(_) => {
            let _ = writeln!(
                out,
                "#   Hz   {}  RI   R   {}",
                par_type,
                fmt_impedance(per_port[0])
            );
            let _ = writeln!(out, "! {}", comment.unwrap_or(""));
            if ports == 2 {
                write_data_2(&mut out, freq, data);
            } else {
                write_data_1(&mut out, freq, data);
            }
        }
        RefImpedance::PerPort(_) => {
            let _ = writeln!(out, "[Version] 2.0");
            let _ = writeln!(
                out,
                "#   Hz   {}  RI   R   {}",
                par_type,
                fmt_impedance(per_port[0])
            );
            let _ = writeln!(out, "! {}", comment.unwrap_or(""));
            let _ = writeln!(out, "[Number of Ports] {}", ports);
            if ports == 2 {
                let _ = writeln!(out, "[Two-Port Data Order] 21");
            }
            let _ = writeln!(out, "[Number of Data Points] {}", freq.len());
            let _ = writeln!(out, "[Number of Frequencies] {}", freq.len());
            let mut refs = String::new();
            for (index, z) in per_port.iter().enumerate() {
                if index > 0 {
                    refs.push(' ');
                }
                let _ = write!(refs, "{}", fmt_impedance(*z));
            }
            let _ = writeln!(out, "[Reference] {refs}");
            let _ = writeln!(out, "[Network Data]");
            write_data_v2(&mut out, freq, data);
            let _ = writeln!(out, "[End]");
        }
    }

    fs::write(filename, out).map_err(|e| format!("Failed to write {filename}: {e}"))
}

/// Native `write_data_1`: `fprintf('%e ', f)` then 4 complex values per line,
/// row by row (the `>=3`-port order defect is preserved on purpose).
fn write_data_1(out: &mut String, freq: &[f64], data: &Array3<C64>) {
    let ports = data.shape()[0];
    for (f, frequency) in freq.iter().enumerate() {
        let _ = write!(out, "{} ", fmt_e(*frequency));
        for row in 0..ports {
            let mut column = 0;
            while column < ports {
                let chunk = 4.min(ports - column);
                for _ in 0..chunk {
                    let value = data[[row, column, f]];
                    let _ = write!(
                        out,
                        "{} {}   ",
                        fmt_e_signed(value.re),
                        fmt_e_signed(value.im)
                    );
                    column += 1;
                }
                let _ = write!(out, "\n             ");
            }
        }
        let _ = writeln!(out);
    }
}

/// Native `write_data_2`: one record per line in `S11, S21, S12, S22` order.
fn write_data_2(out: &mut String, freq: &[f64], data: &Array3<C64>) {
    for (f, frequency) in freq.iter().enumerate() {
        let d11 = data[[0, 0, f]];
        let d21 = data[[1, 0, f]];
        let d12 = data[[0, 1, f]];
        let d22 = data[[1, 1, f]];
        let _ = writeln!(
            out,
            "{}   {} {}   {} {}   {} {}   {} {}",
            fmt_e(*frequency),
            fmt_e_signed(d11.re),
            fmt_e_signed(d11.im),
            fmt_e_signed(d21.re),
            fmt_e_signed(d21.im),
            fmt_e_signed(d12.re),
            fmt_e_signed(d12.im),
            fmt_e_signed(d22.re),
            fmt_e_signed(d22.im)
        );
    }
}

/// Touchstone v2 data block: one record per line, column major for every port
/// count (`S11, S21, ..., Sn1, S12, ...`).
fn write_data_v2(out: &mut String, freq: &[f64], data: &Array3<C64>) {
    let ports = data.shape()[0];
    for (f, frequency) in freq.iter().enumerate() {
        let mut line = String::new();
        let _ = write!(line, "{}", fmt_e(*frequency));
        for column in 0..ports {
            for row in 0..ports {
                let value = data[[row, column, f]];
                let _ = write!(
                    line,
                    " {} {}",
                    fmt_e_signed(value.re),
                    fmt_e_signed(value.im)
                );
            }
        }
        let _ = writeln!(out, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempFile {
        path: String,
    }

    impl TempFile {
        /// Create `<temp>/ctb-rs-test-<pid>-<n>-<name>` with `contents`.
        fn with_name(name: &str, contents: &str) -> Self {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("ctb-rs-test-{}-{id}-{name}", std::process::id()))
                .to_string_lossy()
                .into_owned();
            if let Some(parent) = std::path::Path::new(&path).parent() {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(&path, contents).unwrap();
            Self { path }
        }

        fn with_suffix(suffix: &str, contents: &str) -> Self {
            Self::with_name(&format!("file{suffix}"), contents)
        }

        fn empty(suffix: &str) -> Self {
            Self::with_suffix(suffix, "")
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    const TWO_PORT: &str = "\
#   Hz   S  RI   R   50
! hello
1.000000e+06    1.000000e-01  2.000000e-01    2.000000e-01  0.000000e+00    3.000000e-01 -1.000000e-01   -1.500000e-01  4.000000e-01
2.000000e+06    1.100000e-01  2.100000e-01    2.100000e-01  0.000000e+00    3.100000e-01 -1.100000e-01   -1.600000e-01  4.100000e-01
";

    #[test]
    fn reads_a_two_port_v1_file() {
        let file = TempFile::with_suffix(".s2p", TWO_PORT);
        let data = read_touchstone(&file.path).unwrap();
        assert_eq!(data.version, 1);
        assert_eq!(data.param_type, 's');
        assert_eq!(data.num_ports(), 2);
        assert_eq!(data.num_freq(), 2);
        assert_eq!(data.freq, vec![1e6, 2e6]);
        assert_eq!(data.r#ref, RefImpedance::Scalar(c(50.0, 0.0)));
        assert_eq!(data.comment, vec!["! hello".to_string()]);
        assert_eq!(data.two_port_data_order, TwoPortDataOrder::Order21);
        assert_eq!(data.data[[0, 0, 0]], c(0.1, 0.2));
        assert_eq!(data.data[[1, 0, 0]], c(0.2, 0.0));
        assert_eq!(data.data[[0, 1, 0]], c(0.3, -0.1));
        assert_eq!(data.data[[1, 1, 0]], c(-0.15, 0.4));
        assert_eq!(data.data[[0, 0, 1]], c(0.11, 0.21));
        assert_eq!(data.data[[1, 1, 1]], c(-0.16, 0.41));
    }

    #[test]
    fn reads_one_port_db_and_milli_degrees_units() {
        // 1-port, dB/degree format, MHz unit, record continued on a second line.
        let text = "\
! a comment
# MHz S DB R 50
10   0.0  0.0
20   -6.020599913279624
    90.0
";
        let file = TempFile::with_suffix(".s1p", text);
        let data = read_touchstone(&file.path).unwrap();
        assert_eq!(data.version, 1);
        assert_eq!(data.freq, vec![10e6, 20e6]);
        assert!((data.data[[0, 0, 0]] - c(1.0, 0.0)).norm() < 1e-15);
        // -6.0206 dB = 0.5, 90 degrees = i
        assert!((data.data[[0, 0, 1]] - c(0.0, 0.5)).norm() < 1e-15);
        assert_eq!(data.comment, vec!["! a comment".to_string()]);
    }

    #[test]
    fn reads_magnitude_angle_format() {
        let text = "# kHz Y MA R 75\n1 0.5 180\n";
        let file = TempFile::with_suffix(".s1p", text);
        let data = read_touchstone(&file.path).unwrap();
        assert_eq!(data.param_type, 'y');
        assert_eq!(data.freq, vec![1e3]);
        assert!((data.data[[0, 0, 0]] - c(-0.5, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn reads_v2_per_port_reference_and_keywords() {
        let text = "\
[Version] 2.0
# GHz S RI R 50
[Number of Ports] 2
[Two-Port Data Order] 12
[Number of Frequencies] 1
[Number of Data Points] 1
[Reference] 50 75
[Network Data]
1  0.1 0.0  0.2 0.0  0.3 0.0  0.4 0.0
[End]
";
        let file = TempFile::with_suffix(".s2p", text);
        let data = read_touchstone(&file.path).unwrap();
        assert_eq!(data.version, 2);
        assert_eq!(data.two_port_data_order, TwoPortDataOrder::Order12);
        assert_eq!(data.freq, vec![1e9]);
        // Order 12: S11, S12, S21, S22
        assert_eq!(data.data[[0, 0, 0]], c(0.1, 0.0));
        assert_eq!(data.data[[0, 1, 0]], c(0.2, 0.0));
        assert_eq!(data.data[[1, 0, 0]], c(0.3, 0.0));
        assert_eq!(data.data[[1, 1, 0]], c(0.4, 0.0));
        assert_eq!(
            data.r#ref,
            RefImpedance::PerPort(vec![c(50.0, 0.0), c(75.0, 0.0)])
        );
        assert_eq!(data.ref_scalar(), None);
        assert_eq!(
            data.ref_per_port().unwrap(),
            vec![c(50.0, 0.0), c(75.0, 0.0)]
        );
    }

    #[test]
    fn v2_reference_list_shorter_than_the_port_count_fills_from_the_option_line() {
        let text = "# Hz S RI R 40\n[Number of Ports] 3\n[Reference] 10 20\n1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";
        let file = TempFile::with_suffix(".s3p", text);
        let data = read_touchstone(&file.path).unwrap();
        assert_eq!(
            data.ref_per_port().unwrap(),
            vec![c(10.0, 0.0), c(20.0, 0.0), c(40.0, 0.0)]
        );
    }

    #[test]
    fn reads_multiple_records_on_one_line() {
        // 1-port records hold 3 values; the lines below hold one, two, one and
        // four records respectively.
        let text = "# Hz S RI R 50\n1 10 0\n2 20 0 3 30 0\n4 40 0\n5 50 0 6 60 0 7 70 0 8 80 0\n";
        let file = TempFile::with_suffix(".s1p", text);
        let data = read_touchstone(&file.path).unwrap();
        assert_eq!(data.freq, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        for (f, expected) in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
            .iter()
            .enumerate()
        {
            assert_eq!(data.data[[0, 0, f]], c(*expected, 0.0), "record {f}");
        }
    }

    #[test]
    fn rejects_malformed_input_without_panicking() {
        let cases: &[(&str, &str, &str)] = &[
            ("no_extension", "", "no extension"),
            ("bad.txt", "# Hz S RI R 50\n1 0 0 0 0\n", "like .s2p"),
            ("x.s0p", "", "zero ports"),
            ("x.s2p", "1 0 0 0\n", "data before"),
            (
                "x.s2p",
                "# Hz X RI R 50\n1 0 0 0 0 0 0 0 0\n",
                "wrong parameter type",
            ),
            ("x.s2p", "# Hz S XX R 50\n1 0 0\n", "wrong complex format"),
            ("x.s2p", "# Xz S RI R 50\n1 0 0\n", "wrong frequency unit"),
            ("x.s2p", "# Hz S RI 50\n1 0 0\n", "missing the reference"),
            ("x.s2p", "# Hz S RI R\n1 0 0\n", "reference impedance value"),
            ("x.s2p", "# Hz S RI\n1 0 0\n", "incomplete option line"),
            (
                "x.s2p",
                "# Hz S RI R fifty\n1 0 0\n",
                "not a reference impedance",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n1 0 0 abc 0 0 0 0 0\n",
                "not a number",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n1 0 0 0 0 0 0 0\n",
                "incomplete data record",
            ),
            ("x.s2p", "# Hz S RI R 50\n", "no data records"),
            (
                "x.s2p",
                "# Hz S RI R 50\n[Mixed-Mode Order] 21\n1 0 0 0 0 0 0 0 0\n",
                "mixed-mode",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n[Number of Ports] 3\n1 0 0 0 0 0 0 0 0\n",
                "does not match",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n[Number of Data Points] 5\n1 0 0 0 0 0 0 0 0\n",
                "Number of Data Points",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n[Two-Port Data Order] 13\n1 0 0 0 0 0 0 0 0\n",
                "Two-Port Data Order",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n[Number of Ports\n1 0 0 0 0 0 0 0 0\n",
                "missing ']'",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n[Moon Phase] 3\n1 0 0 0 0 0 0 0 0\n",
                "unsupported Touchstone keyword",
            ),
            (
                "x.s2p",
                "# Hz S RI R -50\n1 0 0 0 0 0 0 0 0\n",
                "positive real part",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n[Reference] 50 75 60\n1 0 0 0 0 0 0 0 0\n",
                "lists 3 values",
            ),
            (
                "x.s3p",
                "# Hz S RI R 50\n1 0 0 0 0 0 0 0 0\n! truncated\n",
                "is incomplete",
            ),
            (
                "x.s2p",
                "# Hz S RI R 50\n[Matrix Format] Lower\n1 0 0 0 0 0 0 0 0\n",
                "Matrix Format",
            ),
        ];
        for (name, contents, needle) in cases {
            let file = TempFile::with_name(name, contents);
            let err = read_touchstone(&file.path).unwrap_err();
            assert!(
                err.to_lowercase().contains(&needle.to_lowercase()),
                "expected '{needle}' in error '{err}' for {contents:?}"
            );
        }
    }

    #[test]
    fn write_and_read_back_one_two_and_three_port_files() {
        for ports in [1usize, 2, 3] {
            let mut data = Array3::zeros((ports, ports, 2));
            for f in 0..2 {
                for i in 0..ports {
                    for j in 0..ports {
                        data[[i, j, f]] =
                            c(0.1 * (i + 1) as f64, -0.05 * (j + 1) as f64 + f as f64);
                    }
                }
            }
            let freq = [1e6, 2e6];
            let file = TempFile::empty(&format!(".s{ports}p"));
            write_touchstone('s', &freq, &data, &file.path, None, Some("round trip")).unwrap();
            let back = read_touchstone(&file.path).unwrap();
            assert_eq!(back.num_ports(), ports);
            assert_eq!(back.freq, vec![1e6, 2e6]);
            assert_eq!(back.comment, vec!["! round trip".to_string()]);
            if ports == 2 {
                for f in 0..2 {
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!(
                                (back.data[[i, j, f]] - data[[i, j, f]]).norm() < 1e-15,
                                "2-port round trip ({i},{j},{f})"
                            );
                        }
                    }
                }
            } else if ports == 1 {
                for f in 0..2 {
                    assert!((back.data[[0, 0, f]] - data[[0, 0, f]]).norm() < 1e-15);
                }
            } else {
                // Preserved native defect: the v1 writer is row major for >= 3
                // ports while every reader is column major, so the record comes
                // back transposed.
                for f in 0..2 {
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!(
                                (back.data[[i, j, f]] - data[[j, i, f]]).norm() < 1e-15,
                                "3-port native-order round trip ({i},{j},{f})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn per_port_reference_writes_v2_and_round_trips_three_ports() {
        let ports = 3;
        let mut data = Array3::zeros((ports, ports, 2));
        for f in 0..2 {
            for i in 0..ports {
                for j in 0..ports {
                    data[[i, j, f]] = c(0.2 * (i as f64) - 0.1 * (j as f64), 0.3 * f as f64 + 0.01);
                }
            }
        }
        let freq = [1e9, 2e9];
        let file = TempFile::empty(".s3p");
        write_touchstone(
            's',
            &freq,
            &data,
            &file.path,
            Some(RefImpedance::from(vec![50.0, 75.0, 60.0])),
            None,
        )
        .unwrap();
        let text = fs::read_to_string(&file.path).unwrap();
        assert!(text.starts_with("[Version] 2.0\n"));
        assert!(text.contains("[Number of Ports] 3\n"));
        assert!(text.contains("[Reference] 50 75 60\n"));
        assert!(text.contains("[Network Data]\n"));
        assert!(text.trim_end().ends_with("[End]"));

        let back = read_touchstone(&file.path).unwrap();
        assert_eq!(back.version, 2);
        assert_eq!(
            back.ref_per_port().unwrap(),
            vec![c(50.0, 0.0), c(75.0, 0.0), c(60.0, 0.0)]
        );
        for f in 0..2 {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((back.data[[i, j, f]] - data[[i, j, f]]).norm() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn v1_header_matches_the_native_writer() {
        let mut data = Array3::zeros((1, 1, 1));
        data[[0, 0, 0]] = c(0.5, -0.5);
        let file = TempFile::empty(".s1p");
        write_touchstone('z', &[1e9], &data, &file.path, None, Some("one")).unwrap();
        let text = fs::read_to_string(&file.path).unwrap();
        assert_eq!(
            text,
            "#   Hz   Z  RI   R   1\n! one\n1.000000e+09  5.000000e-01 -5.000000e-01   \n             \n"
        );
    }

    #[test]
    fn v1_two_port_layout_matches_the_native_writer() {
        let mut data = Array3::zeros((2, 2, 1));
        data[[0, 0, 0]] = c(0.1, 0.2);
        data[[1, 0, 0]] = c(0.2, 0.0);
        data[[0, 1, 0]] = c(0.3, -0.1);
        data[[1, 1, 0]] = c(-0.15, 0.4);
        let file = TempFile::empty(".s2p");
        write_touchstone('s', &[1e6], &data, &file.path, None, Some("hello")).unwrap();
        let text = fs::read_to_string(&file.path).unwrap();
        assert_eq!(
            text,
            "#   Hz   S  RI   R   50\n! hello\n\
             1.000000e+06    1.000000e-01  2.000000e-01    2.000000e-01  0.000000e+00    \
             3.000000e-01 -1.000000e-01   -1.500000e-01  4.000000e-01\n"
        );
    }

    #[test]
    fn writer_rejects_bad_input() {
        let mut data = Array3::zeros((2, 3, 1));
        data[[0, 0, 0]] = c(1.0, 0.0);
        let file = TempFile::empty(".s2p");
        assert!(write_touchstone('s', &[1.0], &data, &file.path, None, None)
            .unwrap_err()
            .contains("rows and columns"));
        let data = Array3::<C64>::zeros((2, 2, 2));
        assert!(write_touchstone('s', &[1.0], &data, &file.path, None, None)
            .unwrap_err()
            .contains("frequencies"));
        let data = Array3::<C64>::zeros((2, 2, 1));
        assert!(write_touchstone('q', &[1.0], &data, &file.path, None, None)
            .unwrap_err()
            .contains("only s-, y-, z-"));
        assert!(write_touchstone(
            's',
            &[1.0],
            &data,
            &file.path,
            Some(RefImpedance::from(vec![50.0, 75.0, 60.0])),
            None
        )
        .unwrap_err()
        .contains("3 entries"));
    }

    #[test]
    fn complex_reference_impedance_round_trips_like_num2str() {
        // read_touchstone.m accepts "R 50+5i" (str2double handles the complex
        // form), so it must be readable and writable here as well.
        let text = "# MHz S RI R 50+5i\n1 0.1 0.0 0.2 0.0 0.2 0.0 0.1 0.0\n";
        let file = TempFile::with_suffix(".s2p", text);
        let parsed = read_touchstone(&file.path).unwrap();
        assert_eq!(parsed.r#ref, RefImpedance::Scalar(c(50.0, 5.0)));

        let data = Array3::<C64>::zeros((2, 2, 1));
        let out = TempFile::empty(".s2p");
        write_touchstone(
            's',
            &[1.0],
            &data,
            &out.path,
            Some(RefImpedance::from(c(50.0, 5.0))),
            None,
        )
        .unwrap();
        let written = fs::read_to_string(&out.path).unwrap();
        assert!(
            written.starts_with("#   Hz   S  RI   R   50+5i\n"),
            "{written}"
        );
        assert_eq!(
            read_touchstone(&out.path).unwrap().r#ref,
            RefImpedance::Scalar(c(50.0, 5.0))
        );

        // Negative imaginary parts use num2str's "a-bi" spelling.
        write_touchstone(
            's',
            &[1.0],
            &data,
            &out.path,
            Some(RefImpedance::from(c(50.0, -2.5))),
            None,
        )
        .unwrap();
        assert!(fs::read_to_string(&out.path)
            .unwrap()
            .starts_with("#   Hz   S  RI   R   50-2.5i\n"));

        // Exponent forms are parsed too ("1e-3+2e-4i").
        let text = "# Hz S RI R 1e-3+2e-4i\n1 0.1 0.0 0.2 0.0 0.2 0.0 0.1 0.0\n";
        let file = TempFile::with_suffix(".s2p", text);
        let parsed = read_touchstone(&file.path).unwrap();
        assert_eq!(parsed.r#ref, RefImpedance::Scalar(c(1e-3, 2e-4)));
    }

    #[test]
    fn fmt_e_matches_c_printf() {
        assert_eq!(fmt_e(1.0), "1.000000e+00");
        assert_eq!(fmt_e(1e6), "1.000000e+06");
        assert_eq!(fmt_e(-1.5e-1), "-1.500000e-01");
        assert_eq!(fmt_e(0.0), "0.000000e+00");
        assert_eq!(fmt_e(-0.0), "-0.000000e+00");
        assert_eq!(fmt_e(f64::INFINITY), "inf");
        assert_eq!(fmt_e(f64::NEG_INFINITY), "-inf");
        assert_eq!(fmt_e(f64::NAN), "nan");
        assert_eq!(fmt_e_signed(0.1), " 1.000000e-01");
        assert_eq!(fmt_e_signed(-0.1), "-1.000000e-01");
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let err = read_touchstone("/definitely/not/here/x.s2p").unwrap_err();
        assert!(err.contains("Failed to open"), "unexpected error: {err}");
    }

    #[test]
    fn a_parameter_only_file_is_rejected() {
        let file = TempFile::with_suffix(".s2p", "# Hz S RI R 50\n! nothing else\n");
        assert!(read_touchstone(&file.path)
            .unwrap_err()
            .contains("no data records"));
    }
}
