use float8::{F8E4M3, F8E5M2};

use crate::test::{make_range, Fp8, RangeTest, TestCase, TestCommon, TestPtx};

pub static PTX: &str = include_str!("cvt_rn_f16x2_f8x2type.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "cvt_rn_f16x2_e4m3".to_string(),
            make_range(Cvt::<F8E4M3>::new()),
        ),
        TestCase::new(
            "cvt_rn_f16x2_e5m2".to_string(),
            make_range(Cvt::<F8E5M2>::new()),
        ),
    ]
}

struct Cvt<FromElem: Fp8> {
    _phantom: std::marker::PhantomData<FromElem>,
}

impl<FromElem: Fp8> Cvt<FromElem> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<FromElem: Fp8> TestPtx for Cvt<FromElem> {
    fn body(&self) -> String {
        let t = format!("{}x2", FromElem::name());
        PTX.replace("<INPUT>", &t)
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

fn fp16_verify(expected: f16, output: f16) -> bool {
    if expected.is_nan() && output.is_nan() {
        true
    } else if expected.to_bits() == output.to_bits() {
        true
    } else {
        false
    }
}

impl<FromElem: Fp8> TestCommon for Cvt<FromElem> {
    type Input = u16;
    type Output = u32;

    fn host_verify(&self, input: u16, output: u32) -> Result<(), u32> {
        let output = (
            f16::from_bits(output as u16),
            f16::from_bits((output >> 16) as u16),
        );
        let expected: (f16, f16) = (
            FromElem::from_bits(input as u8).to_f16(),
            FromElem::from_bits((input >> 8) as u8).to_f16(),
        );
        if fp16_verify(expected.0, output.0) && fp16_verify(expected.1, output.1) {
            Ok(())
        } else {
            Err(((expected.0.to_bits() as u32) << 16) | expected.1.to_bits() as u32)
        }
    }
}

impl<FromElem: Fp8> RangeTest for Cvt<FromElem> {
    const MAX_VALUE: u32 = u8::MAX as u32;
    fn generate(&self, input: u32) -> Self::Input {
        input as u16 | ((input as u16) << 8)
    }
}
