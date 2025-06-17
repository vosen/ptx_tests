use std::mem;
use std::marker::PhantomData;
use crate::test::{make_range, PtxScalar, RangeTest, TestCase, TestCommon, TestPtx};

pub static PTX: &str = include_str!("testp.ptx");

#[derive(Clone, Copy)]
pub enum TestpMode {
    Finite,
    Infinite,
    Number,
    NotANumber,
    Normal,
    Subnormal,
}

impl TestpMode {
    fn as_str(&self) -> &'static str {
        match self {
            TestpMode::Finite     => "finite",
            TestpMode::Infinite   => "infinite",
            TestpMode::Number     => "number",
            TestpMode::NotANumber => "notanumber",
            TestpMode::Normal     => "normal",
            TestpMode::Subnormal  => "subnormal",
        }
    }
}

pub fn all_tests() -> Vec<TestCase> {
    let modes = [
        TestpMode::Finite,
        TestpMode::Infinite,
        TestpMode::Number,
        TestpMode::NotANumber,
        TestpMode::Normal,
        TestpMode::Subnormal,
    ];

    modes.iter().map(|mode| {
        TestCase::new(
            format!("testp_{}_f32", mode.as_str()),
            make_range(Testp::<f32>::new(*mode))
        )
    }).collect::<Vec<_>>()

}

pub struct Testp<T: PtxScalar> {
    mode: TestpMode,
    _phantom: PhantomData<T>,
}

impl<T: PtxScalar> Testp<T> {
    pub fn new(mode: TestpMode) -> Self {
        Self {
            mode,
            _phantom: PhantomData,
        }
    }
}

impl<T: PtxScalar> TestPtx for Testp<T> {
    fn body(&self) -> String {
        PTX.replace("<MODE>", self.mode.as_str())
           .replace("<TYPE>", T::name())
           .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string())
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl<T> TestCommon for Testp<T>
where
    T: PtxScalar + num::Float + num::Zero,
{
    type Input = T;
    type Output = u32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let expected = match self.mode {
            TestpMode::Finite     => input.is_finite() as u32,
            TestpMode::Infinite   => input.is_infinite() as u32,
            TestpMode::Number     => (!input.is_nan()) as u32,
            TestpMode::NotANumber => input.is_nan() as u32,
            TestpMode::Normal     => (input.is_normal() || input == <T as num::Zero>::zero()) as u32, //This is required because PTX returns true for zero, but Rust does not.
            TestpMode::Subnormal  => input.is_subnormal() as u32,
        };
        if output == expected {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl RangeTest for Testp<f32> {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}
