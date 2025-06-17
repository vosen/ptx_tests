use crate::test::{make_random, PtxScalar, RandomTest, RangeTest, TestCase, TestCommon, TestPtx};
use num::cast::AsPrimitive;
use num::PrimInt;
use num::{traits::FromBytes, Zero};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::fmt::Debug;
use std::mem;

pub static PTX: &str = include_str!("bfe.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        bfe_rng::<u32>(),
        bfe_rng::<i32>(),
        bfe_rng::<u64>(),
        bfe_rng::<i64>(),
    ]
}

fn bfe_rng<T: PtxScalar + AsPrimitive<usize> + PrimInt + Default>() -> TestCase
where
    Standard: Distribution<T>,
{
    let test = make_random(Bfe::<T>::default());
    TestCase::new(format!("bfe_rng_{}", T::name()), test)
}

#[derive(Default)]
pub struct Bfe<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar> TestPtx for Bfe<T> {
    fn body(&self) -> String {
        PTX
            .replace("<TYPE>", T::name())
            .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string())
    }

    fn args(&self) -> &[&str] {
        &[
            "input",
            "positions",
            "lengths",
            "output",
        ]
    }
}

impl<T: PtxScalar + AsPrimitive<usize> + PrimInt> TestCommon for Bfe<T> {
    type Input = (T, u32, u32);

    type Output = T;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        fn bfe_host<T: PtxScalar + AsPrimitive<usize> + PrimInt>(
            value: T,
            pos: u32,
            len: u32,
        ) -> T {
            // ERRATA: len and pos parameters in 64 bit variant use whole 32 bits, not just bottom 8
            let pos = if mem::size_of::<T>() == 4 {
                pos.to_le_bytes()[0] as usize
            } else {
                pos as usize
            };
            let len = if mem::size_of::<T>() == 4 {
                len.to_le_bytes()[0] as usize
            } else {
                len as usize
            };
            let msb = mem::size_of::<T>() * 8 - 1;
            let sbit = if T::unsigned() || len == 0 {
                false
            } else {
                get_bit(value, Ord::min(pos + len - 1, msb))
            };
            let mut d = <T as Zero>::zero();
            for i in 0..=msb {
                let bit = if i < len && pos + i <= msb {
                    get_bit(value, pos + i)
                } else {
                    sbit
                };
                set_bit(&mut d, i, bit)
            }
            d
        }
        let (value, len, pos) = input;
        let expected = bfe_host(value, len, pos);
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl<T: PtxScalar + AsPrimitive<usize> + PrimInt + FromBytes> RangeTest for Bfe<T>
where
    for<'a> T::Bytes: TryFrom<&'a [u8]>,
    for<'a> <<T as FromBytes>::Bytes as TryFrom<&'a [u8]>>::Error: Debug,
{
    fn generate(&self, input: u32) -> Self::Input {
        let len = input.to_le_bytes()[0] as u32;
        let pos = input.to_le_bytes()[1] as u32;
        let value = [
            input.to_le_bytes()[3],
            input.to_le_bytes()[2],
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let value = T::from_be_bytes(&T::Bytes::try_from(&value).unwrap());
        (value, pos, len)
    }
}

impl<T: PtxScalar + AsPrimitive<usize> + PrimInt + Default> RandomTest for Bfe<T>
where
    Standard: Distribution<T>,
{
    fn generate<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Input {
        let value = rng.gen();
        let len = (rng.gen::<u16>() & 0x1ff) as u32;
        let pos = (rng.gen::<u16>() & 0x1ff) as u32;
        (value, len, pos)
    }
}

fn get_bit<T: PtxScalar + AsPrimitive<usize>>(value: T, n: usize) -> bool {
    assert!(n < mem::size_of::<T>() * 8);
    let value: usize = value.as_();
    value & (1 << n) != 0
}

fn set_bit<T: PtxScalar + PrimInt>(value: &mut T, n: usize, bit: bool) {
    assert!(n < mem::size_of::<T>() * 8);
    let mask = T::one().unsigned_shl(n as u32);
    if bit {
        *value = value.bitor(mask);
    } else {
        *value = value.bitand(mask.not());
    }
}
