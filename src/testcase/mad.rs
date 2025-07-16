use crate::common::WideningMul;
use crate::test::{make_random, PtxScalar, RandomTest, TestCase, TestCommon, TestPtx};
use num::cast::AsPrimitive;
use num::PrimInt;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use std::mem;

static PTX: &str = include_str!("mad.ptx");

#[derive(PartialEq, Eq, Copy, Clone)]
enum Mode {
    Low,
    High,
    Wide,
}

impl Mode {
    fn ptx_modifier(&self) -> &'static str {
        match self {
            Mode::Low => "lo",
            Mode::High => "hi",
            Mode::Wide => "wide",
        }
    }
}

fn mad_low_high<T, U>(a: T, b: T, c: U, mode: Mode, sat: bool) -> U
where
    T: PtxScalar + PrimInt + AsPrimitive<usize> + AsPrimitive<U> + WideningMul,
    U: PtxScalar + PrimInt + AsPrimitive<usize> + AsPrimitive<T>,
    usize: AsPrimitive<U>,
{
    let (lo, hi) = a.widening_mul(b);
    let wide_a: U = a.as_();
    let wide_b: U = b.as_();
    let wide = wide_a * wide_b;
    let mul_result = match mode {
        Mode::Low => lo.as_(),
        Mode::High => hi.as_(),
        Mode::Wide => wide,
    };
    if sat {
        mul_result.saturating_add(c)
    } else {
        mul_result + c
    }
}

fn verify_mad<T, U>(input: (T, T, U), output: U, mode: Mode, sat: bool) -> Result<(), U>
where
    T: PtxScalar + PrimInt + AsPrimitive<usize> + AsPrimitive<U> + WideningMul,
    U: PtxScalar + PrimInt + AsPrimitive<usize> + AsPrimitive<T>,
    usize: AsPrimitive<U>,
{
    let (a, b, c) = input;
    let expected = mad_low_high::<T, U>(a, b, c, mode, sat);
    if expected == output {
        Ok(())
    } else {
        Err(expected)
    }
}

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = Vec::new();
    for mode in [Mode::Low, Mode::High] {
        tests.push(TestCase::new(
            format!("mad_{}_u16", mode.ptx_modifier()),
            make_random(MadTest::<u16, u16>::new(mode, false)),
        ));
        tests.push(TestCase::new(
            format!("mad_{}_s16", mode.ptx_modifier()),
            make_random(MadTest::<i16, i16>::new(mode, false)),
        ));
    }
    tests.push(TestCase::new(
        "mad_wide_u16".to_string(),
        make_random(MadTest::<u16, u32>::new(Mode::Wide, false)),
    ));
    tests.push(TestCase::new(
        "mad_wide_s16".to_string(),
        make_random(MadTest::<i16, i32>::new(Mode::Wide, false)),
    ));
    tests.push(TestCase::new(
        "mad_hi_sat_s32".to_string(),
        make_random(MadTest::<i32, i32>::new(Mode::High, true)),
    ));
    tests
}

struct MadTest<T: PtxScalar, U: PtxScalar> {
    mode: Mode,
    saturate: bool,
    _phantom: (std::marker::PhantomData<T>, std::marker::PhantomData<U>),
}

impl<T: PtxScalar, U: PtxScalar> MadTest<T, U> {
    fn new(mode: Mode, saturate: bool) -> Self {
        Self {
            _phantom: (std::marker::PhantomData, std::marker::PhantomData),
            mode,
            saturate,
        }
    }
}

impl<T: PtxScalar, U: PtxScalar> TestPtx for MadTest<T, U> {
    fn body(&self) -> String {
        let sat = if self.saturate { ".sat" } else { "" };
        PTX.replace("<STYPE>", T::name())
            .replace("<STYPE_SIZE>", &mem::size_of::<T>().to_string())
            .replace("<DTYPE>", U::name())
            .replace("<DTYPE_SIZE>", &mem::size_of::<U>().to_string())
            .replace("<MODE>", self.mode.ptx_modifier())
            .replace("<SAT>", sat)
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "input_c", "output"]
    }
}

impl<T, U> TestCommon for MadTest<T, U>
where
    T: PtxScalar + PrimInt + AsPrimitive<usize> + AsPrimitive<U> + WideningMul,
    U: PtxScalar + PrimInt + AsPrimitive<usize> + AsPrimitive<T>,
    usize: AsPrimitive<U>,
{
    type Input = (T, T, U);
    type Output = U;
    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        verify_mad::<T, U>(input, output, self.mode, self.saturate)
    }
}

impl<T, U> RandomTest for MadTest<T, U>
where
    T: PtxScalar + PrimInt + AsPrimitive<usize> + AsPrimitive<U> + WideningMul,
    U: PtxScalar + PrimInt + AsPrimitive<usize> + AsPrimitive<T>,
    usize: AsPrimitive<U>,
    Standard: Distribution<T> + Distribution<U>,
{
    fn generate<R: rand::Rng>(&self, rng: &mut R) -> Self::Input {
        let a = rng.gen::<T>();
        let b = rng.gen::<T>();
        let c = rng.gen::<U>();
        (a, b, c)
    }
}
