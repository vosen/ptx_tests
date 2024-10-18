use crate::{
    common,
    test::{self, PtxScalar, RangeTest, TestCase, TestCommon},
};
use num::PrimInt;
use std::mem;

pub static PTX: &str = include_str!("minmax.ptx");

pub(crate) fn all_tests() -> Vec<TestCase> {
    let mut tests = Vec::new();
    for ftz in [false, true] {
        for nan in [false, true] {
            tests.push(min(ftz, nan));
        }
    }
    tests
}

fn min(ftz: bool, nan: bool) -> TestCase {
    let name = format!(
        "min{}{}",
        if ftz { "_ftz" } else { "" },
        if nan { "_nan" } else { "" }
    );
    TestCase::new(
        name.to_string(),
        Box::new(move |cuda| test::run_range::<Min>(cuda, Min { ftz, nan })),
    )
}

struct Min {
    ftz: bool,
    nan: bool,
}

impl TestCommon for Min {
    type Input = (half::f16, half::f16);
    type Output = half::f16;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        fn min_host(mut a: half::f16, mut b: half::f16, nan: bool, ftz: bool) -> half::f16 {
            common::flush_to_zero_f16(&mut a, ftz);
            common::flush_to_zero_f16(&mut b, ftz);
            if a.is_nan() && b.is_nan() {
                half::f16::NAN
            } else if nan && (a.is_nan() || b.is_nan()) {
                half::f16::NAN
            } else if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else {
                a.min(b)
            }
        }
        let (a, b) = input;
        let expected = min_host(a, b, self.nan, self.ftz);
        if expected.to_ne_bytes() == output.to_ne_bytes() {
            Ok(())
        } else {
            Err(expected)
        }
    }

    fn ptx(&self) -> String {
        let name = format!(
            "min{}{}.f16",
            if self.ftz { ".ftz" } else { "" },
            if self.nan { ".nan" } else { "" }
        );
        let mut src = PTX
            .replace("<TYPE_SIZE>", "2")
            .replace("<TYPE>", "f16")
            .replace("<BTYPE>", "b16")
            .replace("<OP>", &name);
        src.push('\0');
        src
    }
}

impl RangeTest for Min {
    fn generate(&self, input: u32) -> Self::Input {
        unsafe { mem::transmute::<_, (half::f16, half::f16)>(input) }
    }
}
