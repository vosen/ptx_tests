use crate::test::{
    make_range, PtxScalar, RangeTest, TestCase, TestCommon, TestPtx,
};
use num::cast::AsPrimitive;
use num::{PrimInt, Saturating};
use num::traits::WrappingAdd;
use std::mem;

pub static PTX: &str = include_str!("add.ptx");

fn add_with_saturation<T>(a: T, b: T, saturate: bool) -> T
where
    T: PtxScalar + PrimInt + Saturating + WrappingAdd,
{
    if saturate {
        a.saturating_add(b)
    } else {
        a.wrapping_add(&b)
    }
}

fn verify_addition<T>(input: (T, T), output: T, saturate: bool) -> Result<(), T>
where
    T: PtxScalar + PrimInt + Saturating + WrappingAdd, 
{
    let (a, b) = input;
    let expected = add_with_saturation(a, b, saturate);
    if expected == output {
        Ok(())
    } else {
        Err(expected)
    }
}


pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new("add_u16".to_string(), make_range(AddTest::<u16>::default())),
        TestCase::new("add_i16".to_string(), make_range(AddTest::<i16>::default())),
        TestCase::new("add_sat_s32".to_string(), make_range(AddSatTest::default())),
    ]
}

#[derive(Default)]
pub struct AddTest<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar> TestPtx for AddTest<T> {
    fn body(&self) -> String {
        PTX.replace("<TYPE>", T::name())
           .replace("<SAT>", "") // no saturation modifier
           .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string())
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

impl<T> TestCommon for AddTest<T>
where
    T: PtxScalar + PrimInt + Copy + AsPrimitive<usize> + Saturating + WrappingAdd,
{
    type Input = (T, T);
    type Output = T;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        verify_addition(input, output, false)
    }
}

macro_rules! impl_addtest_for {
    ($ty:ty) => {
        impl RangeTest for AddTest<$ty> {
            const MAX_VALUE: u32 = u32::MAX;
            fn generate(&self, input: u32) -> Self::Input {
                let op1 = (input >> 16) as $ty;
                let op2 = (input & 0xffff) as $ty;
                (op1, op2)
            }
        }
    };
}
impl_addtest_for!(u16);
impl_addtest_for!(i16);

#[derive(Default)]
pub struct AddSatTest;

impl TestPtx for AddSatTest {
    fn body(&self) -> String {
        PTX.replace("<TYPE>", "s32")
           .replace("<SAT>", ".sat")
           .replace("<TYPE_SIZE>", "4")
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

impl TestCommon for AddSatTest {
    type Input = (i32, i32);
    type Output = i32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        verify_addition(input, output, true)
    }
}

impl RangeTest for AddSatTest {
    const MAX_VALUE: u32 = u32::MAX; // 2^32 - 1 combinations
    fn generate(&self, input: u32) -> Self::Input {
        let [b1,b2,b3, b4] = input.to_ne_bytes();
        let op1 = i32::from_ne_bytes([b1, 0, 0, b2]);
        let op2 = i32::from_ne_bytes([b3, 0, 0, b4]);
        (op1, op2)
    }
}