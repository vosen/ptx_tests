use crate::common::flush_to_zero_f32;
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};

pub static PTX: &str = include_str!("abs.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new("abs".to_string(), make_range(Abs { ftz: false })),
        TestCase::new("abs_ftz".to_string(), make_range(Abs { ftz: true })),
    ]
}

pub struct Abs {
    pub ftz: bool,
}

impl TestPtx for Abs {
    fn body(&self) -> String {
        let ftz = if self.ftz { ".ftz" } else { "" };
        PTX.replace("<FTZ>", ftz)
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl TestCommon for Abs {
    type Input = f32;
    type Output = f32;

    fn host_verify(&self, input: f32, output: f32) -> Result<(), f32> {
        let mut expected = input.abs();
        flush_to_zero_f32(&mut expected, self.ftz);
        if expected.is_nan() && output.is_nan() {
            Ok(())
        } else if expected.to_bits() == output.to_bits() {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl RangeTest for Abs {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}
