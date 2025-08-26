use float8::{F8E4M3, F8E5M2};
use num::Float;

use crate::test::{make_range, PtxScalar, RangeTest, TestCase, TestCommon, TestPtx};

pub static PTX: &str = include_str!("cvt_rn_satfinite_f8x2_f32.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "cvt_rn_satfinite_e4m3x2_f32".to_string(),
            make_range(Cvt::<F8E4M3>::new()),
        ),
        TestCase::new(
            "cvt_rn_satfinite_e5m2x2_f32".to_string(),
            make_range(Cvt::<F8E5M2>::new()),
        ),
    ]
}

trait Fp8: PtxScalar + Float {
    fn from_f32(x: f32) -> Self;
    fn from_bits(bits: u8) -> Self;
    fn to_bits(&self) -> u8;
    // This is necessary because float8 is_nan hardcodes a few NaN values and returns "false" for
    // others.
    fn is_nan_correct(&self) -> bool;
}

impl Fp8 for F8E4M3 {
    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }
    fn from_bits(bits: u8) -> Self {
        Self::from_bits(bits)
    }
    fn to_bits(&self) -> u8 {
        self.to_bits()
    }
    fn is_nan_correct(&self) -> bool {
        return (self.to_bits() >> 3) & 0b1111 == 0b1111;
    }
}

impl Fp8 for F8E5M2 {
    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }
    fn from_bits(bits: u8) -> Self {
        Self::from_bits(bits)
    }
    fn to_bits(&self) -> u8 {
        self.to_bits()
    }
    fn is_nan_correct(&self) -> bool {
        return (self.to_bits() >> 2) & 0b11111 == 0b11111;
    }
}

struct Cvt<ToElem: Fp8> {
    _phantom: std::marker::PhantomData<ToElem>,
}

impl<ToElem: Fp8> Cvt<ToElem> {
    fn new() -> Self {
        assert!(ToElem::float());
        assert_eq!(ToElem::size_of(), 1);
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<ToElem: Fp8> TestPtx for Cvt<ToElem> {
    fn body(&self) -> String {
        let t = format!("{}x2", ToElem::name());
        PTX.replace("<OUTPUT>", &t)
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

fn fp8_verify<ToElem: Fp8>(expected: ToElem, output: ToElem) -> bool {
    if expected.is_nan_correct() && output.is_nan_correct() {
        true
    } else if expected.to_bits() == output.to_bits() {
        true
    } else {
        false
    }
}

impl<ToElem: Fp8> TestCommon for Cvt<ToElem> {
    type Input = (f32, f32);
    type Output = u16;

    fn host_verify(&self, input: (f32, f32), output: u16) -> Result<(), u16> {
        let output = (
            ToElem::from_bits((output >> 8) as u8),
            ToElem::from_bits(output as u8),
        );
        let expected: (ToElem, ToElem) = (Fp8::from_f32(input.0), Fp8::from_f32(input.1));
        if fp8_verify(expected.0, output.0) && fp8_verify(expected.1, output.1) {
            Ok(())
        } else {
            Err(((expected.0.to_bits() as u16) << 8) | expected.1.to_bits() as u16)
        }
    }
}

impl<ToElem: Fp8> RangeTest for Cvt<ToElem> {
    fn generate(&self, input: u32) -> Self::Input {
        (f32::from_bits(input), f32::from_bits(input))
    }
}
