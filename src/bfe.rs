use crate::cuda::Cuda;
use crate::test::{self, PtxScalar, RandomTest, RangeTest, TestCase, TestCommon};
use num::cast::AsPrimitive;
use num::PrimInt;
use num::{traits::FromBytes, Zero};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::fmt::Debug;
use std::mem;

pub static PTX: &str = include_str!("bfe.ptx");

pub(super) fn rng_u32() -> TestCase {
    bfe_rng::<u32>()
}
pub(super) fn rng_s32() -> TestCase {
    bfe_rng::<i32>()
}
pub(super) fn rng_u64() -> TestCase {
    bfe_rng::<u64>()
}
pub(super) fn rng_s64() -> TestCase {
    bfe_rng::<i64>()
}

fn bfe_rng<T: PtxScalar + AsPrimitive<usize> + PrimInt>() -> TestCase
where
    Standard: Distribution<T>,
{
    let test = Box::new(move |cuda: &Cuda| test::run_random::<Bfe<T>>(cuda));
    TestCase::new(format!("bfe_rng_{}", T::name()), test)
}

pub struct Bfe<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar + AsPrimitive<usize> + PrimInt> TestCommon for Bfe<T> {
    type Input = (T, u32, u32);

    type Output = T;

    fn ptx(&self) -> String {
        let mut src = PTX
            .replace("<TYPE>", T::name())
            .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string());
        src.push('\0');
        src
    }

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

impl<T: PtxScalar + AsPrimitive<usize> + PrimInt> RandomTest for Bfe<T>
where
    Standard: Distribution<T>,
{
    fn generate<R: Rng + ?Sized>(rng: &mut R) -> Self::Input {
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
