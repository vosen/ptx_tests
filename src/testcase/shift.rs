use crate::test::{make_range, PtxScalar, RangeTest, TestCase, TestCommon};
use num::PrimInt;
use std::mem;

pub static PTX: &str = include_str!("shift.ptx");

pub(crate) fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "shl_b16".to_string(),
            make_range(Shl {}),
        ),
        TestCase::new(
            "shr_u16".to_string(),
            make_range::<Shr<u16>>(Shr { _phantom: std::marker::PhantomData }),
        ),
        TestCase::new(
            "shr_s16".to_string(),
            make_range::<Shr<i16>>(Shr { _phantom: std::marker::PhantomData }),
        ),
    ]
}

struct Shl {}

impl TestCommon for Shl {
    type Input = (u16, u16);
    type Output = u16;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (value, shift) = input;
        let expected = if shift >= 16 { 0 } else { value << shift };
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }

    fn ptx(&self) -> String {
        PTX.replace("<OP>", "shl.b16")
    }

    fn ptx_args(&self) -> &[&str] {
        &[
            "input_a",
            "input_b",
            "output",
        ]
    }
}

impl RangeTest for Shl {
    fn generate(&self, input: u32) -> Self::Input {
        unsafe { mem::transmute::<_, (u16, u16)>(input) }
    }
}

struct Shr<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar + PrimInt> TestCommon for Shr<T> {
    type Input = (T, u16);
    type Output = T;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (value, shift) = input;
        let expected = if shift >= 16 {
            if T::signed() {
                value.signed_shr(15)
            } else {
                <T as num::Zero>::zero()
            }
        } else if T::signed() {
            value.signed_shr(shift as u32)
        } else {
            value.unsigned_shr(shift as u32)
        };
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }

    fn ptx(&self) -> String {
        let op = if T::signed() { "shr.s16" } else { "shr.u16" };
        PTX.replace("<OP>", op)
    }

    fn ptx_args(&self) -> &[&str] {
        &[
            "input_a",
            "input_b",
            "output",
        ]
    }
}

impl RangeTest for Shr<u16> {
    fn generate(&self, input: u32) -> Self::Input {
        unsafe { mem::transmute::<_, (u16, u16)>(input) }
    }
}

impl RangeTest for Shr<i16> {
    fn generate(&self, input: u32) -> Self::Input {
        unsafe { mem::transmute::<_, (i16, u16)>(input) }
    }
}
