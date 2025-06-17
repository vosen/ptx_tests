use crate::test::{make_range, PtxScalar, RangeTest, TestCase, TestCommon, TestPtx};
use num::cast::AsPrimitive;
use num::{PrimInt, Saturating};
use std::mem;
use num::traits::WrappingSub;

pub static PTX: &str = include_str!("sub.ptx");

fn sub_with_saturation<T>(a: T, b: T, saturate: bool) -> T
where
    T: PtxScalar + PrimInt + Saturating + WrappingSub,
{
    if saturate {
        a.saturating_sub(b)
    } else {
        a.wrapping_sub(&b)
    }
}

fn verify_substraction<T>(input: (T, T), output: T, saturate: bool) -> Result<(), T>
where
    T: PtxScalar + PrimInt + Saturating + WrappingSub,
{
    let (a, b) = input;
    let expected = sub_with_saturation(a, b, saturate);
    if expected == output {
        Ok(())
    } else {
        Err(expected)
    }
}

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new("sub_u16".to_string(), make_range(SubTest::<u16>::default())),
        TestCase::new("sub_i16".to_string(), make_range(SubTest::<i16>::default())),
        TestCase::new("sub_sat_s32".to_string(), make_range(SubSatTest::default())),
    ]
}

#[derive(Default)]
pub struct SubTest<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar> TestPtx for SubTest<T> {
    fn body(&self) -> String {
        PTX.replace("<TYPE>", T::name())
           .replace("<SAT>", "") // no saturation modifier.
           .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string())
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

impl<T> TestCommon for SubTest<T>
where
    T: PtxScalar + PrimInt + Copy + AsPrimitive<usize> + Saturating + WrappingSub,
{
    type Input = (T, T);
    type Output = T;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        verify_substraction(input, output, false)
    }
}

macro_rules! impl_subtest_for {
    ($ty:ty) => {
        impl RangeTest for SubTest<$ty> {
            const MAX_VALUE: u32 = u32::MAX;
            fn generate(&self, input: u32) -> Self::Input {
                let op1 = (input >> 16) as $ty;
                let op2 = (input & 0xffff) as $ty;
                (op1, op2)
            }
        }
    };
}
impl_subtest_for!(u16);
impl_subtest_for!(i16);


#[derive(Default)]
pub struct SubSatTest;

impl TestPtx for SubSatTest {
    fn body(&self) -> String {
        PTX.replace("<TYPE>", "s32")
           .replace("<SAT>", ".sat")
           .replace("<TYPE_SIZE>", "4")
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

impl TestCommon for SubSatTest {
    type Input = (i32, i32);
    type Output = i32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        verify_substraction(input, output, true)
    }
}

impl RangeTest for SubSatTest {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        let [b1,b2,b3, b4] = input.to_ne_bytes();
        let op1 = i32::from_ne_bytes([b1, 0, 0, b2]);
        let op2 = i32::from_ne_bytes([b3, 0, 0, b4]);
        (op1, op2)
    }
}
