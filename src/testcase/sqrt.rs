use crate::common::{self, flush_to_zero_f32, Rounding};
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};

static PTX: &str = include_str!("sqrt.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    for ftz in [false, true] {
        tests.push(sqrt_approx(ftz));
        tests.push(sqrt_rnd(Rounding::Rn, ftz));
        if cfg!(not(windows)) {
            tests.push(sqrt_rnd(Rounding::Rz, ftz));
            tests.push(sqrt_rnd(Rounding::Rm, ftz));
            tests.push(sqrt_rnd(Rounding::Rp, ftz));
        }
    }
    tests
}

fn sqrt_rnd(rnd: Rounding, ftz: bool) -> TestCase {
    sqrt::<false>(rnd, ftz)
}

fn sqrt_approx(ftz: bool) -> TestCase {
    sqrt::<true>(Rounding::Default, ftz)
}

fn sqrt<const APPROX: bool>(rnd: Rounding, ftz: bool) -> TestCase {
    let test = make_range::<Sqrt<APPROX>>(Sqrt { rnd, ftz });
    let mode = if APPROX { "approx" } else { rnd.as_str() };
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("sqrt_{}{}", mode, ftz), test)
}

struct Sqrt<const APPROX: bool> {
    ftz: bool,
    rnd: Rounding,
}

const APPROX_TOLERANCE: f64 = 0.00000011920928955078125f64; // 2^-23

impl<const APPROX: bool> TestPtx for Sqrt<APPROX> {
    fn body(&self) -> String {
        let rnd = if APPROX {
            "approx"
        } else if self.rnd == Rounding::Rn {
            "approx"
        } else {
            self.rnd.as_str()
        };
        let mode = format!("{}{}", rnd, if self.ftz { ".ftz" } else { "" });
        PTX.replace("<MODE>", &mode)
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl<const APPROX: bool> TestCommon for Sqrt<APPROX> {
    type Input = f32;

    type Output = f32;

    fn host_verify(
        &self,
        mut input: Self::Input,
        output: Self::Output,
    ) -> Result<(), Self::Output> {
        fn sqrt_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => f32::NAN,
                f if f.to_bits() == (-0.0f32).to_bits() => -0.0,
                f if f.is_finite() && f.is_sign_negative() => f32::NAN,
                0.0 => 0.0,
                f32::INFINITY => f32::INFINITY,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if APPROX {
            if let Some(mut expected) = sqrt_approx_special(input) {
                flush_to_zero_f32(&mut expected, self.ftz);
                if expected.is_nan() && output.is_nan()
                    || expected.to_bits() == output.to_bits()
                {
                    Ok(())
                } else {
                    Err(expected)
                }
            } else {
                let mut precise_result = sqrt_host(input);
                flush_to_zero_f32(&mut precise_result, self.ftz);
                if common::relative_diff(precise_result, output as f64, APPROX_TOLERANCE) {
                    Ok(())
                } else {
                    Err(precise_result as f32)
                }
            }
        } else {
            let mut result = os::sqrt_rnd(input, self.rnd);
            flush_to_zero_f32(&mut result, self.ftz);
            if result.is_nan() && output.is_nan() {
                Ok(())
            } else {
                if result.to_ne_bytes() == output.to_ne_bytes() {
                    Ok(())
                } else {
                    Err(result)
                }
            }
        }
    }
}

impl<const APPROX: bool> RangeTest for Sqrt<APPROX> {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}

fn sqrt_host(input: f32) -> f64 {
    let input = input as f64;
    input.sqrt()
}

#[cfg(not(windows))]
mod os {
    use crate::common::Rounding;

    fn rug_round(rnd: Rounding) -> rug::float::Round {
        match rnd {
            Rounding::Rzi | Rounding::Rz => rug::float::Round::Zero,
            Rounding::Default | Rounding::Rni | Rounding::Rn => rug::float::Round::Nearest,
            Rounding::Rpi | Rounding::Rp => rug::float::Round::Up,
            Rounding::Rmi | Rounding::Rm => rug::float::Round::Down,
        }
    }

    pub fn sqrt_rnd(input: f32, rnd: Rounding) -> f32 {
        let rnd = rug_round(rnd);
        let mut input = rug::Float::with_val_round(24, input, rnd).0;
        input.sqrt_round(rnd);
        input.to_f32_round(rnd)
    }
}

#[cfg(windows)]
mod os {
    use crate::common::Rounding;

    pub fn sqrt_rnd(input: f32, rnd: Rounding) -> f32 {
        let precise_result = super::sqrt_host(input);
        rnd.with(|| precise_result as f32)
    }
}
