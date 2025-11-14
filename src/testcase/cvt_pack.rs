use crate::test::{make_random, RandomTest, TestPtx};
use crate::test::{PtxScalar, TestCase, TestCommon};
use std::marker::PhantomData;

pub static PTX: &str = include_str!("cvt_pack.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "cvt_pack_sat_u8_s32_b32".to_string(),
            make_random(CvtPack::<u8>::default()),
        ),
        TestCase::new(
            "cvt_pack_sat_s8_s32_b32".to_string(),
            make_random(CvtPack::<i8>::default()),
        ),
    ]
}

#[derive(Default)]
struct CvtPack<T> {
    _phantom: PhantomData<T>,
}

impl<T: PtxScalar> TestPtx for CvtPack<T> {
    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "input_c", "output"]
    }

    fn body(&self) -> String {
        PTX.replace("<TYPE>", T::name())
    }
}

impl<T: PtxScalar + Into<i32> + TryFrom<i32>> TestCommon for CvtPack<T> {
    type Input = (i32, i32, u32);
    type Output = u32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b, c) = input;
        let ta = a.max(T::min_value().into()).min(T::max_value().into());
        let tb = b.max(T::min_value().into()).min(T::max_value().into());
        let expected = (tb as u8 as u32) | ((ta as u8 as u32) << 8) | (c << 16);
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl<T: PtxScalar + Into<i32> + TryFrom<i32>> RandomTest for CvtPack<T> {
    fn generate<R: rand::Rng>(&self, rng: &mut R) -> Self::Input {
        (rng.gen(), rng.gen(), rng.gen())
    }
}
