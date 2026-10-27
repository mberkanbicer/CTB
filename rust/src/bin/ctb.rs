//! `ctb` command line interface.
//!
//! Delivery form for MATLAB/Octave users (RM-116): the native toolbox is a set
//! of M-files; this binary exposes the Rust transforms and Touchstone I/O so a
//! thin M-file wrapper can call it with `system()` and read the produced
//! Touchstone file with the native `read_touchstone.m`.
//!
//! Commands:
//!   ctb info <file.sNp>
//!   ctb convert --to <s|y|z|a|h|g> [--ref R[,R...]] <in.sNp> <out.sNp>
//!   ctb renorm --to R[,R...] [--from R[,R...]] <in.sNp> <out.sNp>

use std::process::ExitCode;

use ctb_rs::ref_impedance::RefImpedance;
use ctb_rs::touchstone::{read_touchstone, write_touchstone};
use ctb_rs::transforms::{g2s, h2s, s2g, s2h, s2y, s2z, s_renorm, y2s, z2s};

const USAGE: &str = "\
Usage: ctb <command> [options] <files>

Commands:
  info <file.sNp>                       Print ports, frequencies and reference
  convert --to <type> [--ref R[,R...]] <in.sNp> <out.sNp>
                                        Convert to S, Y, Z, A, H or G parameters
  renorm --to R[,R...] [--from R[,R...]] <in.sNp> <out.sNp>
                                        Renormalise an S-parameter file
  help                                  Print this text
  version                               Print the version

Reference impedances may be scalar (50) or per-port (50,75,60) in ohms.
The output file must use the Touchstone .sNp extension; the parameter type is
stored in the option line, as in the native writer.
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ctb: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(format!("missing command\n\n{USAGE}"));
    }
    let command = args.remove(0);
    match command.as_str() {
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("ctb {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "info" => {
            let file = single_arg(&args, "info")?;
            let data = read_touchstone(&file)?;
            println!("file: {file}");
            println!("version: {}", data.version);
            println!("parameter: {}", data.param_type);
            println!("ports: {}", data.num_ports());
            println!("frequencies: {}", data.num_freq());
            match data.r#ref.uniform() {
                Some(value) => println!("reference: {} ohm", value.re),
                None => {
                    let values = data
                        .r#ref
                        .resolve(data.num_ports())?
                        .iter()
                        .map(|value| format!("{}", value.re))
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!("reference: [{values}] ohm");
                }
            }
            if let (Some(first), Some(last)) = (data.freq.first(), data.freq.last()) {
                println!("frequency range: {first:e} .. {last:e} Hz");
            }
            Ok(())
        }
        "convert" => {
            let mut target = None;
            let mut reference = None;
            let mut files = Vec::new();
            let mut index = 0;
            while index < args.len() {
                match args[index].as_str() {
                    "--to" => {
                        target = Some(take_value(&args, &mut index, "--to")?);
                    }
                    "--ref" => {
                        reference = Some(parse_ref(&take_value(&args, &mut index, "--ref")?)?);
                    }
                    other if other.starts_with('-') => {
                        return Err(format!("unknown option '{other}'"));
                    }
                    other => files.push(other.to_owned()),
                }
                index += 1;
            }
            let target = target.ok_or("convert needs --to <type>")?;
            if files.len() != 2 {
                return Err("convert needs an input and an output file".into());
            }
            let input = read_touchstone(&files[0])?;
            let ref_z = reference.unwrap_or_else(|| input.r#ref.clone());
            let s = to_s(input.param_type, &input.data, &ref_z)?;
            let target = target.to_ascii_lowercase();
            let data =
                match target.as_str() {
                    "s" => s,
                    "y" => s2y(&s, &ref_z)?,
                    "z" => s2z(&s, &ref_z)?,
                    "a" => return Err(
                        "Touchstone cannot store ABCD parameters; use the library or the M-file"
                            .into(),
                    ),
                    "h" => s2h(&s, &ref_z)?,
                    "g" => s2g(&s, &ref_z)?,
                    other => return Err(format!("unsupported target type '{other}'")),
                };
            let type_char = target.chars().next().expect("checked target");
            write_touchstone(
                type_char,
                &input.freq,
                &data,
                &files[1],
                Some(ref_z),
                Some("converted by ctb"),
            )
        }
        "renorm" => {
            let mut target = None;
            let mut source = None;
            let mut files = Vec::new();
            let mut index = 0;
            while index < args.len() {
                match args[index].as_str() {
                    "--to" => target = Some(parse_ref(&take_value(&args, &mut index, "--to")?)?),
                    "--from" => {
                        source = Some(parse_ref(&take_value(&args, &mut index, "--from")?)?)
                    }
                    other if other.starts_with('-') => {
                        return Err(format!("unknown option '{other}'"));
                    }
                    other => files.push(other.to_owned()),
                }
                index += 1;
            }
            let target = target.ok_or("renorm needs --to <R[,R...]>")?;
            if files.len() != 2 {
                return Err("renorm needs an input and an output file".into());
            }
            let input = read_touchstone(&files[0])?;
            let source = source.unwrap_or_else(|| input.r#ref.clone());
            let s = to_s(input.param_type, &input.data, &source)?;
            let data = s_renorm(&s, &source, &target)?;
            write_touchstone(
                's',
                &input.freq,
                &data,
                &files[1],
                Some(target),
                Some("renormalised by ctb"),
            )
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    }
}

fn single_arg(args: &[String], command: &str) -> Result<String, String> {
    if args.len() != 1 {
        return Err(format!("{command} needs exactly one file"));
    }
    Ok(args[0].clone())
}

fn take_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{option} needs a value"))
}

/// Parse `50` or `50,75,60` (real ohms) into a reference impedance.
fn parse_ref(text: &str) -> Result<RefImpedance, String> {
    let values = text
        .split(',')
        .map(|value| {
            let value = value.trim();
            if value.is_empty() {
                return Err(format!("empty reference impedance in '{text}'"));
            }
            let parsed: f64 = value
                .parse()
                .map_err(|_| format!("not a number: '{value}'"))?;
            if !parsed.is_finite() || parsed <= 0.0 {
                return Err(format!("reference impedance must be positive: '{value}'"));
            }
            Ok(parsed)
        })
        .collect::<Result<Vec<f64>, String>>()?;
    if values.is_empty() {
        return Err("missing reference impedance".into());
    }
    if values.len() == 1 {
        Ok(RefImpedance::from(values[0]))
    } else {
        Ok(RefImpedance::from(values))
    }
}

/// Convert any supported Touchstone parameter type to S.
fn to_s(
    param_type: char,
    data: &ndarray::Array3<ctb_rs::C64>,
    ref_z: &RefImpedance,
) -> Result<ndarray::Array3<ctb_rs::C64>, String> {
    match param_type {
        's' => Ok(data.clone()),
        'z' => z2s(data, ref_z.clone()),
        'y' => y2s(data, ref_z.clone()),
        'h' => h2s(data, ref_z.clone()),
        'g' => g2s(data, ref_z.clone()),
        other => Err(format!("unsupported source parameter type '{other}'")),
    }
}
