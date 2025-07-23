use crate::common::{self, flush_to_zero_f32};
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};
use core::f32;

static PTX: &str = include_str!("lg2.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![lg2(false), lg2(true)]
}

fn lg2(ftz: bool) -> TestCase {
    let test = make_range(Lg2 { ftz });
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("lg2_approx{}", ftz), test)
}

struct Lg2 {
    ftz: bool,
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
                f if f.to_bits() == (-0.0f32).to_bits() => f32::NEG_INFINITY,
                f if f.is_finite() && f.is_sign_negative() => f32::NAN,
                0.0 => f32::NEG_INFINITY,
                f32::INFINITY => f32::INFINITY,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if let Some(expected) = lg2_approx_special(input) {
            if (expected.is_nan() && output.is_nan())
                || (expected.to_ne_bytes() == output.to_ne_bytes())
            {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let mut precise_result = lg2_host(input);
            flush_to_zero_f32(&mut precise_result, self.ftz);
            if within_tolerance(input, precise_result, output as f64) {
                Ok(())
            } else {
                Err(precise_result as f32)
            }
        }
    }
}

const RANGE_MIN: f32 = 0.5f32;
const RANGE_MAX: f32 = 2f32;
const TOLERANCE: f64 = 0.0000002384185791015625; // 2^-22

fn within_tolerance(x: f32, precise_result: f64, output: f64) -> bool {
    if x > RANGE_MIN && x < RANGE_MAX {
        common::absolute_diff(precise_result, output, TOLERANCE)
    } else {
        common::relative_diff(precise_result, output, TOLERANCE)
    }
}

impl RangeTest for Lg2 {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}

fn lg2_host(input: f32) -> f64 {
    (input as f64).log2()
}
