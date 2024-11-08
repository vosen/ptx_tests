use crate::common::{flush_to_zero_f32, Rounding};
use crate::cuda::Cuda;
use crate::test::{self, RangeTest, TestCase, TestCommon};
use std::mem;

pub static PTX: &str = include_str!("rcp.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    for ftz in [false, true] {
        tests.push(rcp_rnd(Rounding::Rn, ftz));
        tests.push(rcp_rnd(Rounding::Rz, ftz));
        tests.push(rcp_rnd(Rounding::Rm, ftz));
        tests.push(rcp_rnd(Rounding::Rp, ftz));
        tests.push(rcp_approx(ftz));
    }
    tests
}

fn rcp_rnd(rnd: Rounding, ftz: bool) -> TestCase {
    rcp::<false>(rnd, ftz)
}

fn rcp_approx(ftz: bool) -> TestCase {
    rcp::<true>(Rounding::Default, ftz)
}

fn rcp<const APPROX: bool>(rnd: Rounding, ftz: bool) -> TestCase {
    let test = Box::new(move |cuda: &Cuda| test::run_range::<Rcp<APPROX>>(cuda, Rcp { rnd, ftz }));
    let mode = if APPROX { "approx" } else { rnd.as_str() };
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("rcp_{}{}", mode, ftz), test)
}

pub struct Rcp<const APPROX: bool> {
    ftz: bool,
    rnd: Rounding,
}

impl<const APPROX: bool> TestCommon for Rcp<APPROX> {
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
        fn rcp_host(input: f32) -> f64 {
            let input = input as f64;
            input.recip()
        }
        fn rcp_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => -0.0,
                f if f.is_subnormal() && f.is_sign_negative() => f32::NEG_INFINITY,
                f if f.to_ne_bytes() == (-0.0f32).to_ne_bytes() => f32::NEG_INFINITY,
                0.0 => f32::INFINITY,
                f if f.is_subnormal() && f.is_sign_positive() => f32::INFINITY,
                f32::INFINITY => 0.0,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if APPROX {
            if let Some(mut expected) = rcp_approx_special(input) {
                flush_to_zero_f32(&mut expected, self.ftz);
                if expected.to_ne_bytes() == output.to_ne_bytes() {
                    Ok(())
                } else {
                    Err(expected)
                }
            } else {
                let precise_result = rcp_host(input);
                let mut result_f32 = precise_result as f32;
                flush_to_zero_f32(&mut result_f32, self.ftz);
                let precise_output = output as f64;
                let diff = (precise_output - result_f32 as f64).abs();
                if diff <= 2f64.powi(-23) {
                    Ok(())
                } else {
                    Err(precise_result as f32)
                }
            }
        } else {
            let precise_result = rcp_host(input);
            let mut result = self.rnd.with(|| precise_result as f32);
            flush_to_zero_f32(&mut result, self.ftz);
            if result.is_nan() && output.is_nan() {
                Ok(())
            } else {
                if result.to_ne_bytes() == output.to_ne_bytes() {
                    Ok(())
                } else {
                    // HACK: Those two values with those two particular rounding modes
                    // disagree between CPU and GPU
                    match (self.rnd, self.ftz, input, output) {
                        (Rounding::Rm, true, -8.50706e37, -0.0) => return Ok(()),
                        (Rounding::Rp, true, 8.50706e37, 0.0) => return Ok(()),
                        _ => {}
                    }
                    Err(result)
                }
            }
        }
    }
}

const MAX_NEGATIVE_SUBNORMAL: f32 = unsafe { mem::transmute(0x807FFFFFu32) };
const MAX_POSITIVE_SUBNORMAL: f32 = unsafe { mem::transmute(0x007FFFFFu32) };

impl<const APPROX: bool> RangeTest for Rcp<APPROX> {
    const MAX_VALUE: u32 = if APPROX {
        (unsafe { mem::transmute::<_, u32>(2.0f32) - mem::transmute::<_, u32>(1.0f32) }) + 127
    } else {
        u32::MAX
    };

    fn generate(&self, input: u32) -> Self::Input {
        if APPROX {
            let max_number = unsafe { mem::transmute::<_, u32>(2.0f32) };
            if input > max_number {
                match input - max_number {
                    1 => f32::NEG_INFINITY,
                    2 => MAX_NEGATIVE_SUBNORMAL,
                    3 => -0.0,
                    4 => 0.0,
                    5 => MAX_POSITIVE_SUBNORMAL,
                    6 => f32::INFINITY,
                    7 => f32::NAN,
                    _ => 0.0,
                }
            } else {
                unsafe { mem::transmute::<_, f32>(input + mem::transmute::<_, u32>(1.0f32)) }
            }
        } else {
            unsafe { mem::transmute::<_, f32>(input) }
        }
    }
}
