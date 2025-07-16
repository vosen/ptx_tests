use crate::common;
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};
use core::f32;

pub static PTX: &str = include_str!("ex2.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new("ex2_approx".to_string(), make_range(Ex2::new(false))),
        TestCase::new("ex2_approx_ftz".to_string(), make_range(Ex2::new(true))),
    ]
}

pub struct Ex2 {
    pub ftz: bool,
}

impl Ex2 {
    pub fn new(ftz: bool) -> Self {
        Self { ftz }
    }
}

impl TestPtx for Ex2 {
    fn body(&self) -> String {
        PTX.replace("<FLUSH>", if self.ftz { ".ftz" } else { "" })
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl TestCommon for Ex2 {
    type Input = f32;
    type Output = f32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        fn ex2_approx_special(input: f32) -> Option<f32> {
            //These special cases are defined in PTX documentation
            Some(match input {
                x if x.is_nan() => f32::NAN,
                x if x == f32::NEG_INFINITY => 0.0,
                x if x == f32::INFINITY => f32::INFINITY,
                x if x.to_bits() == (-0.0f32).to_bits() => 1.0,
                x if x.to_bits() == (0.0f32).to_bits() => 1.0,
                _ => return None,
            })
        }
        if let Some(expected) = ex2_approx_special(input) {
            if (expected.is_nan() && output.is_nan())
                || (expected.to_ne_bytes() == output.to_ne_bytes())
            {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let result_f32 = ex2_host(input, self.ftz);
            if output.to_bits().abs_diff(result_f32.to_bits()) <= 2 {
                Ok(())
            } else {
                Err(result_f32)
            }
        }
    }
}

// sweep all 32 bit values
impl RangeTest for Ex2 {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}

fn ex2_host(input: f32, ftz: bool) -> f32 {
    let mut input_mod = input;
    common::flush_to_zero_f32(&mut input_mod, ftz);
    let exact = (input_mod as f64).exp2();
    let mut result = exact as f32;
    common::flush_to_zero_f32(&mut result, ftz);
    result
}
