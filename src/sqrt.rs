use crate::common::{self, flush_to_zero_f32, Rounding};
use crate::cuda::Cuda;
use crate::test::{self, RangeTest, TestCase, TestCommon};
use std::mem;

pub static PTX: &str = include_str!("sqrt.ptx");

pub(crate) fn all_tests() -> Vec<TestCase> {
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

pub(super) fn sqrt_rnd(rnd: Rounding, ftz: bool) -> TestCase {
    sqrt::<false>(rnd, ftz)
}
pub(super) fn sqrt_approx(ftz: bool) -> TestCase {
    sqrt::<true>(Rounding::Default, ftz)
}

fn sqrt<const APPROX: bool>(rnd: Rounding, ftz: bool) -> TestCase {
    let test =
        Box::new(move |cuda: &Cuda| test::run_range::<Sqrt<APPROX>>(cuda, Sqrt { rnd, ftz }));
    let mode = if APPROX { "approx" } else { rnd.as_str() };
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("sqrt_{}{}", mode, ftz), test)
}

pub struct Sqrt<const APPROX: bool> {
    ftz: bool,
    rnd: Rounding,
}

const APPROX_TOLERANCE: f64 = 0.00000011920928955078125f64; // 2^-23

impl<const APPROX: bool> TestCommon for Sqrt<APPROX> {
    type Input = f32;

    type Output = f32;

    fn ptx(&self) -> String {
        let rnd = if APPROX { "approx" } else { self.rnd.as_str() };
        let mode = format!("{}{}", rnd, if self.ftz { ".ftz" } else { "" });
        let mut src = PTX.replace("<MODE>", &mode);
        src.push('\0');
        src
    }

    fn host_verify(
        &self,
        mut input: Self::Input,
        output: Self::Output,
    ) -> Result<(), Self::Output> {
        fn sqrt_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => f32::NAN,
                f if f.is_normal() && f.is_sign_negative() => f32::NAN,
                f if f.is_subnormal() && f.is_sign_negative() => -0.0,
                f if f.to_ne_bytes() == (-0.0f32).to_ne_bytes() => -0.0,
                0.0 => 0.0,
                f if f.is_subnormal() && f.is_sign_positive() => 0.0,
                f32::INFINITY => f32::INFINITY,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if APPROX {
            if let Some(mut expected) = sqrt_approx_special(input) {
                flush_to_zero_f32(&mut expected, self.ftz);
                if expected.to_ne_bytes() == output.to_ne_bytes() {
                    Ok(())
                } else {
                    Err(expected)
                }
            } else {
                let precise_result = sqrt_host(input);
                let mut result_f32 = precise_result as f32;
                flush_to_zero_f32(&mut result_f32, self.ftz);
                let precise_output = output as f64;
                let diff = (precise_output - result_f32 as f64).abs();
                if diff <= APPROX_TOLERANCE {
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

const RANGE_MIN: f32 = 1f32;
const RANGE_MAX: f32 = 4f32;

impl<const APPROX: bool> RangeTest for Sqrt<APPROX> {
    const MAX_VALUE: u32 = if APPROX {
        (unsafe { mem::transmute::<_, u32>(RANGE_MAX) - mem::transmute::<_, u32>(RANGE_MIN) }) + 127
    } else {
        u32::MAX
    };

    fn generate(&self, input: u32) -> Self::Input {
        if APPROX {
            let max_number = unsafe { mem::transmute::<_, u32>(RANGE_MAX) };
            if input > max_number {
                match input - max_number {
                    1 => f32::NEG_INFINITY,
                    2 => common::MAX_NEGATIVE_SUBNORMAL,
                    3 => -0.0,
                    4 => 0.0,
                    5 => common::MAX_POSITIVE_SUBNORMAL,
                    6 => f32::INFINITY,
                    7 => f32::NAN,
                    8 => -1.0,
                    _ => 0.0,
                }
            } else {
                unsafe { mem::transmute::<_, f32>(input + mem::transmute::<_, u32>(RANGE_MIN)) }
            }
        } else {
            unsafe { mem::transmute::<_, f32>(input) }
        }
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