use crate::common::{self, flush_to_zero_f32};
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};
use core::f32;
use rug::Float;

pub static PTX: &str = include_str!("lg2.ptx");

const PRECISION: u32 = 64;

pub fn all_tests() -> Vec<TestCase> {
    vec![lg2(false), lg2(true)]
}

fn lg2(ftz: bool) -> TestCase {
    let mut tolerance = Float::with_val(PRECISION, -22.6f64);
    tolerance.exp2_mut();
    let test = make_range(Lg2 { ftz, tolerance });
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("lg2_approx{}", ftz), test)
}

pub struct Lg2 {
    ftz: bool,
    tolerance: Float,
}

impl TestPtx for Lg2 {
    fn body(&self) -> String {
        let ftz = if self.ftz { ".ftz" } else { "" };
        PTX.replace("<FTZ>", &ftz)
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl TestCommon for Lg2 {
    type Input = f32;

    type Output = f32;

    fn host_verify(
        &self,
        mut input: Self::Input,
        output: Self::Output,
    ) -> Result<(), Self::Output> {
        fn lg2_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => f32::NAN,
                f if f.is_subnormal() && f.is_sign_negative() => f32::NEG_INFINITY,
                f if f.to_ne_bytes() == (-0.0f32).to_ne_bytes() => f32::NEG_INFINITY,
                0.0 => f32::NEG_INFINITY,
                f if f.is_subnormal() && f.is_sign_positive() => f32::NEG_INFINITY,
                f32::INFINITY => f32::INFINITY,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if let Some(mut expected) = lg2_approx_special(input) {
            flush_to_zero_f32(&mut expected, self.ftz);
            if (expected.is_nan() && output.is_nan())
                || (expected.to_ne_bytes() == output.to_ne_bytes())
            {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let precise_result = lg2_host(input);
            let actual_result = Float::with_val(PRECISION, output);
            let mut diff = precise_result.clone() - actual_result.clone();
            diff = diff.abs();
            if diff <= self.tolerance {
                Ok(())
            } else {
                Err(precise_result.to_f32())
            }
        }
    }
}

const RANGE_MIN: f32 = 1f32;
const RANGE_MAX: f32 = 4f32;

impl RangeTest for Lg2 {
    const MAX_VALUE: u32 = (f32::to_bits(RANGE_MAX) - f32::to_bits(RANGE_MIN)) + 127;

    fn generate(&self, input: u32) -> Self::Input {
        let max_number = f32::to_bits(RANGE_MAX);
        if input > max_number {
            match input - max_number {
                1 => f32::NEG_INFINITY,
                2 => common::MAX_NEGATIVE_SUBNORMAL,
                3 => -0.0,
                4 => 0.0,
                5 => common::MAX_POSITIVE_SUBNORMAL,
                6 => f32::INFINITY,
                7 => f32::NAN,
                _ => 0.0,
            }
        } else {
            f32::from_bits(input + f32::to_bits(RANGE_MIN))
        }
    }
}

fn lg2_host(input: f32) -> rug::Float {
    let input = rug::Float::with_val(PRECISION, input);
    input.log2()
}
