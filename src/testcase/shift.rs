use crate::test::{self, PtxScalar, RangeTest, TestCase, TestCommon};
use num::PrimInt;
use std::mem;

pub static PTX: &str = include_str!("shift.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "shl_b16".to_string(),
            Box::new(|cuda| test::run_range::<Shl>(cuda, Shl {})),
        ),
        TestCase::new(
            "shr_u16".to_string(),
            Box::new(|cuda| {
                test::run_range::<Shr<u16>>(
                    cuda,
                    Shr {
                        _phantom: std::marker::PhantomData,
                    },
                )
            }),
        ),
        TestCase::new(
            "shr_s16".to_string(),
            Box::new(|cuda| {
                test::run_range::<Shr<i16>>(
                    cuda,
                    Shr {
                        _phantom: std::marker::PhantomData,
                    },
                )
            }),
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
        let mut src = PTX.replace("<OP>", "shl.b16");
        src.push('\0');
        src
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
        let mut src = PTX.replace("<OP>", op);
        src.push('\0');
        src
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
