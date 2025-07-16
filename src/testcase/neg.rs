use crate::common::flush_to_zero_f32;
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};

pub static PTX: &str = include_str!("neg.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new("neg".to_string(), make_range(Neg { ftz: false })),
        TestCase::new("neg_ftz".to_string(), make_range(Neg { ftz: true })),
    ]
}

pub struct Neg {
    pub ftz: bool,
}

impl TestPtx for Neg {
    fn body(&self) -> String {
        let ftz = if self.ftz { ".ftz" } else { "" };
        PTX.replace("<FTZ>", ftz)
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl TestCommon for Neg {
    type Input = f32;
    type Output = f32;

    fn host_verify(&self, input: f32, output: f32) -> Result<(), f32> {
        let mut expected = -input;
        if self.ftz {
            flush_to_zero_f32(&mut expected, true);
        }
        if expected.is_nan() && output.is_nan() {
            Ok(())
        } else if expected.to_bits() == output.to_bits() {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl RangeTest for Neg {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}
