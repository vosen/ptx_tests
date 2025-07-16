use crate::test::{make_range, PtxScalar, RangeTest, TestCase, TestCommon, TestPtx};
use num::cast::AsPrimitive;
use num::PrimInt;
use std::mem;

pub static PTX: &str = include_str!("mul.ptx");

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum Mode {
    Low,
    High,
}

impl Mode {
    fn to_ptx(&self) -> &'static str {
        match self {
            Mode::Low => "lo",
            Mode::High => "hi",
        }
    }
}

fn verify_multiplication<T, U>(input: (T, T), output: T, mode: Mode) -> Result<(), T>
where
    T: PrimInt + AsPrimitive<U>,
    U: PrimInt + 'static,
{
    let (a, b) = input;
    let a: U = a.as_();
    let b: U = b.as_();
    let result_wide = a * b;
    assert_eq!(mem::size_of::<U>(), mem::size_of::<[T; 2]>());
    let result_wide = unsafe { mem::transmute_copy::<_, [T; 2]>(&result_wide) };
    let result = match mode {
        Mode::Low => result_wide[0],
        Mode::High => result_wide[1],
    };
    if result == output {
        Ok(())
    } else {
        Err(result)
    }
}

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = Vec::new();
    for mode in [Mode::Low, Mode::High] {
        tests.push(mul::<u16, u32>(mode));
        tests.push(mul::<i16, i32>(mode));
    }
    tests.push(mul_wide::<u16, u32>());
    tests.push(mul_wide::<i16, i32>());
    tests
}

fn mul<T: PtxScalar, U: PtxScalar>(mode: Mode) -> TestCase
where
    T: PrimInt + AsPrimitive<U>,
    U: PrimInt + AsPrimitive<T>,
{
    let test = make_range(MulTest::<T, U>::new(mode));
    TestCase::new(format!("mul_{}_{}", mode.to_ptx(), T::name()), test)
}

pub struct MulTest<T: PtxScalar, U: PtxScalar> {
    _phantom: (std::marker::PhantomData<T>, std::marker::PhantomData<U>),
    mode: Mode,
}

impl<T: PtxScalar, U: PtxScalar> MulTest<T, U> {
    pub fn new(mode: Mode) -> Self {
        Self {
            _phantom: (std::marker::PhantomData, std::marker::PhantomData),
            mode,
        }
    }
}

impl<T: PtxScalar, U: PtxScalar> TestPtx for MulTest<T, U> {
    fn body(&self) -> String {
        PTX.replace("<TYPE_IN>", T::name())
            .replace("<TYPE_OUT>", T::name())
            .replace("<TYPE_IN_SIZE>", &mem::size_of::<T>().to_string())
            .replace("<TYPE_OUT_SIZE>", &mem::size_of::<T>().to_string())
            .replace("<MODE>", self.mode.to_ptx())
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

impl<T, U> TestCommon for MulTest<T, U>
where
    T: PtxScalar + PrimInt + AsPrimitive<U>,
    U: PtxScalar + PrimInt + 'static,
{
    type Input = (T, T);
    type Output = T;
    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        verify_multiplication::<T, U>(input, output, self.mode)
    }
}

impl<T, U> RangeTest for MulTest<T, U>
where
    T: PtxScalar + PrimInt + AsPrimitive<U>,
    U: PtxScalar + PrimInt + 'static,
{
    const MAX_VALUE: u32 = u32::MAX;
    fn generate(&self, input: u32) -> Self::Input {
        assert_eq!(mem::size_of::<u32>(), mem::size_of::<Self::Input>());
        unsafe { mem::transmute_copy(&input) }
    }
}

fn mul_wide<T: PtxScalar, U: PtxScalar>() -> TestCase
where
    T: PrimInt + AsPrimitive<U>,
    U: PrimInt + AsPrimitive<T>,
{
    let test = make_range(MulWideTest::<T, U>::new());
    TestCase::new(format!("mul_wide_{}", T::name()), test)
}

pub struct MulWideTest<T: PtxScalar, U: PtxScalar> {
    _phantom: (std::marker::PhantomData<T>, std::marker::PhantomData<U>),
}

impl<T: PtxScalar, U: PtxScalar> MulWideTest<T, U> {
    fn new() -> Self {
        Self {
            _phantom: (std::marker::PhantomData, std::marker::PhantomData),
        }
    }
}

impl<T: PtxScalar, U: PtxScalar> TestPtx for MulWideTest<T, U> {
    fn body(&self) -> String {
        PTX.replace("<TYPE_IN>", T::name())
            .replace("<TYPE_OUT>", U::name())
            .replace("<TYPE_IN_SIZE>", &mem::size_of::<T>().to_string())
            .replace("<TYPE_OUT_SIZE>", &mem::size_of::<U>().to_string())
            .replace("<MODE>", "wide")
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

impl<T, U> TestCommon for MulWideTest<T, U>
where
    T: PtxScalar + PrimInt + AsPrimitive<U>,
    U: PtxScalar + PrimInt + 'static,
{
    type Input = (T, T);
    type Output = U;
    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        verify_mul_wide(input, output)
    }
}

fn verify_mul_wide<T, U>(input: (T, T), output: U) -> Result<(), U>
where
    T: PrimInt + AsPrimitive<U>,
    U: PrimInt + 'static,
{
    let (a, b) = input;
    let a: U = a.as_();
    let b: U = b.as_();
    let result_wide = a * b;
    if result_wide == output {
        Ok(())
    } else {
        Err(result_wide)
    }
}

impl<T, U> RangeTest for MulWideTest<T, U>
where
    T: PtxScalar + PrimInt + AsPrimitive<U>,
    U: PtxScalar + PrimInt + 'static,
{
    const MAX_VALUE: u32 = u32::MAX;
    fn generate(&self, input: u32) -> (T, T) {
        assert_eq!(mem::size_of::<u32>(), mem::size_of::<(T, T)>());
        unsafe { mem::transmute_copy(&input) }
    }
}
