use crate::test::{self, make_range, PtxScalar, TestCase, TestCommon, TestPtx};
use num::PrimInt;
use rand::{distributions::Standard, prelude::Distribution};
use std::mem;

pub static PTX: &str = include_str!("brev.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        brev(),
    ]
}

fn brev() -> TestCase
where
    Standard: Distribution<u32>,
{
    let bits = mem::size_of::<u32>() * 8;
    let test = make_range(Brev::<u32>::new());
    TestCase::new(format!("brev_b{}", bits), test)
}

pub struct Brev<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar> Brev<T> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: PtxScalar> TestPtx for Brev<T> {
    fn body(&self) -> String {
        let bits = mem::size_of::<T>() * 8;
        PTX
            .replace("<TYPE>", format!("b{}", bits).as_str())
            .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string())
    }

    fn args(&self) -> &[&str] {
        &[
            "input",
            "output",
        ]
    }
}

impl<T: PtxScalar + PrimInt> TestCommon for Brev<T> {
    type Input = T;

    type Output = T;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let expected = input.reverse_bits();
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl test::RangeTest for Brev<u32> {
    fn generate(&self, input: u32) -> Self::Input {
        input
    }
}
