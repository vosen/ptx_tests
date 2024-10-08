use crate::test::{self, PtxScalar, TestCase, TestCommon};
use num::PrimInt;
use rand::{distributions::Standard, prelude::Distribution};
use std::mem;

pub static PTX: &str = include_str!("brev.ptx");

pub(super) fn b32() -> TestCase {
    brev()
}

fn brev() -> TestCase
where
    Standard: Distribution<u32>,
{
    let bits = mem::size_of::<u32>() * 8;
    TestCase {
        test: test::run_range::<Brev<u32>>,
        name: format!("brev_b{}", bits),
    }
}

pub struct Brev<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar + PrimInt> TestCommon for Brev<T> {
    type Input = T;

    type Output = T;

    fn ptx() -> String {
        let bits = mem::size_of::<T>() * 8;
        let mut src: String = PTX
            .replace("<TYPE>", format!("b{}", bits).as_str())
            .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string());
        src.push('\0');
        src
    }

    fn host_verify(input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let expected = input.reverse_bits();
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl test::RangeTest for Brev<u32> {
    fn generate(input: u32) -> Self::Input {
        input
    }
}
