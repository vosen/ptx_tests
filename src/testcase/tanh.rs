use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};
use core::f32;

pub static PTX: &str = include_str!("tanh.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![tanh()]
}

fn tanh() -> TestCase {
    let test = make_range(Tanh { });
    TestCase::new("tanh_approx".to_string(), test)
}

pub struct Tanh {}

const APPROX_REL_TOLERANCE: f64 = 0.00048828125; //2^-11, from PTX documentation 

impl TestPtx for Tanh {
    fn body(&self) -> String {
        PTX.to_string()
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl TestCommon for Tanh {
    type Input = f32;
    type Output = f32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        fn tanh_approx_special(input: f32) -> Option<f32> { //These special cases are defined in PTX documentation
            Some(match input {
                x if x.is_nan() => f32::NAN,
                x if x == f32::NEG_INFINITY => -1.0,
                x if x == f32::INFINITY => 1.0,
                x if x.to_ne_bytes() == (-0.0f32).to_ne_bytes() => -0.0,
                x if x.to_ne_bytes() == (0.0f32).to_ne_bytes() => 0.0,
                _ => return None,
            })
        }
        if let Some(expected) = tanh_approx_special(input) {
            if (expected.is_nan() && output.is_nan())
                || (expected.to_ne_bytes() == output.to_ne_bytes())
            {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let precise_result = tanh_host(input);
            let result_f32 = precise_result as f32;
            if result_f32 == 0.0 {
                if output.to_bits() == 0 { // For zero expected, enforce output should be true 0
                    Ok(())
                } else {
                    Err(result_f32)
                }
            } else {
                let diff = (output as f64 - precise_result).abs();
                let relative_error = diff / (precise_result.abs());
                if relative_error <= APPROX_REL_TOLERANCE {
                    Ok(())
                } else {
                    Err(result_f32)
                }
            }
        }
    }
}

// sweep all 32 bit values
impl RangeTest for Tanh {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}

fn tanh_host(input: f32) -> f64 {
    (input as f64).tanh()
}