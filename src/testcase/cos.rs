use crate::common::{self, flush_to_zero_f32};
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};
use core::f32;

pub static PTX: &str = include_str!("cos.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    for ftz in [false, true] {
        tests.push(cos(ftz));
    }
    tests
}

fn cos(ftz: bool) -> TestCase {
    let test = make_range(Cos { ftz });
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("cos_approx{}", ftz), test)
}

struct Cos {
    ftz: bool,
}

impl TestPtx for Cos {
    fn body(&self) -> String {
        let ftz = if self.ftz { ".ftz" } else { "" };
        PTX.replace("<FTZ>", &ftz)
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl TestCommon for Cos {
    type Input = f32;

    type Output = f32;

    fn host_verify(
        &self,
        mut input: Self::Input,
        output: Self::Output,
    ) -> Result<(), Self::Output> {
        fn cos_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => f32::NAN,
                f if f.to_bits() == (-0.0f32).to_bits() => 1.0,
                0.0 => 1.0,
                f32::INFINITY => f32::NAN,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if let Some(mut expected) = cos_approx_special(input) {
            flush_to_zero_f32(&mut expected, self.ftz);
            if (expected.is_nan() && output.is_nan())
                || (expected.to_ne_bytes() == output.to_ne_bytes())
            {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let mut precise_result = cos_host(input);
            flush_to_zero_f32(&mut precise_result, self.ftz);
            let diff = (precise_result - output as f64).abs();
            if common::is_within_sincos_bounds(input, diff) {
                Ok(())
            } else {
                Err(precise_result as f32)
            }
        }
    }
}

impl RangeTest for Cos {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}

fn cos_host(input: f32) -> f64 {
    let input = input as f64;
    input.cos()
}
