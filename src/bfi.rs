use crate::test::{self, PtxScalar, RandomTest, TestCase, TestCommon};
use num::{cast::AsPrimitive, PrimInt};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::mem;

pub static PTX: &str = include_str!("bfi.ptx");

pub(super) fn rng_b32() -> TestCase {
    bfi_rng::<u32>()
}
pub(super) fn rng_b64() -> TestCase {
    bfi_rng::<u64>()
}

fn bfi_rng<T: PtxScalar + PrimInt + AsPrimitive<usize>>() -> TestCase
where
    Standard: Distribution<T>,
{
    let bits = mem::size_of::<T>() * 8;
    TestCase {
        test: test::run_random::<Bfi<T>>,
        name: format!("bfi_rng_b{}", bits),
    }
}

pub struct Bfi<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar + PrimInt + AsPrimitive<usize>> TestCommon for Bfi<T> {
    type Input = (T, T, u32, u32);

    type Output = T;

    fn ptx() -> String {
        let bits = mem::size_of::<T>() * 8;
        let mut src = PTX
            .replace("<TYPE>", format!("b{}", bits).as_str())
            .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string());
        src.push('\0');
        src
    }

    fn host_verify(input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        fn bfi_host<T: PtxScalar + PrimInt + AsPrimitive<usize>>(
            a: T,
            b: T,
            pos: u32,
            len: u32,
        ) -> T {
            let msb = mem::size_of::<T>() * 8 - 1;
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
            let mut f = b;
            for i in 0..len {
                if pos + i > msb {
                    break;
                }
                let bit = get_bit(a, i);
                set_bit(&mut f, pos + i, bit);
            }
            f
        }
        let (a, b, len, pos) = input;
        let expected = bfi_host(a, b, len, pos);
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl<T: PtxScalar + PrimInt + AsPrimitive<usize>> RandomTest for Bfi<T>
where
    Standard: Distribution<T>,
{
    fn generate<R: Rng + ?Sized>(rng: &mut R) -> Self::Input {
        let a = rng.gen();
        let b = rng.gen();
        let len = (rng.gen::<u16>() & 0x1ff) as u32;
        let pos = (rng.gen::<u16>() & 0x1ff) as u32;
        (a, b, len, pos)
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
