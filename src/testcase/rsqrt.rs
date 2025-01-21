use crate::common::{self, flush_to_zero_f32};
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};
use std::mem;

pub static PTX: &str = include_str!("rsqrt.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    for ftz in [false, true] {
        tests.push(rsqrt_approx(ftz));
    }
    tests
}

fn rsqrt_approx(ftz: bool) -> TestCase {
    let test = make_range(SqrtApprox { ftz });
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("rsqrt_approx{}", ftz), test)
}

pub struct SqrtApprox {
    ftz: bool,
}

const APPROX_TOLERANCE: f64 = 0.00000018068749505405403165188548580484929545894665f64; // 2^-22.4

impl TestPtx for SqrtApprox {
    fn body(&self) -> String {
        let mode = format!("approx{}", if self.ftz { ".ftz" } else { "" });
        PTX.replace("<MODE>", &mode)
    }

    fn args(&self) -> &[&str] {
        &[
            "input",
            "output",
        ]
    }
}

impl TestCommon for SqrtApprox {
    type Input = f32;

    type Output = f32;

    fn host_verify(
        &self,
        mut input: Self::Input,
        output: Self::Output,
    ) -> Result<(), Self::Output> {
        fn rsqrt_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => f32::NAN,
                f if f.is_normal() && f.is_sign_negative() => f32::NAN,
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
        if let Some(mut expected) = rsqrt_approx_special(input) {
            flush_to_zero_f32(&mut expected, self.ftz);
            if expected.to_ne_bytes() == output.to_ne_bytes() {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let precise_result = rsqrt_host(input);
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
    }
}

const RANGE_MIN: f32 = 1f32;
const RANGE_MAX: f32 = 4f32;

impl RangeTest for SqrtApprox {
    const MAX_VALUE: u32 =
        (unsafe { mem::transmute::<_, u32>(RANGE_MAX) - mem::transmute::<_, u32>(RANGE_MIN) })
            + 127;

    fn generate(&self, input: u32) -> Self::Input {
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
    }
}

fn rsqrt_host(input: f32) -> f64 {
    let input = input as f64;
    input.sqrt().recip()
}
