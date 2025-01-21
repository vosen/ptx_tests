use crate::{common, test::{make_range, RangeTest, TestCase, TestCommon, TestPtx}};
use std::mem;

pub static PTX: &str = include_str!("minmax.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = Vec::new();
    for ftz in [false, true] {
        for nan in [false, true] {
            tests.push(min(ftz, nan));
            tests.push(max(ftz, nan));
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
        make_range(Min { ftz, nan }),
    )
}

fn max(ftz: bool, nan: bool) -> TestCase {
    let name = format!(
        "max{}{}",
        if ftz { "_ftz" } else { "" },
        if nan { "_nan" } else { "" }
    );
    TestCase::new(
        name.to_string(),
        make_range(Max { ftz, nan }),
    )
}

struct Min {
    ftz: bool,
    nan: bool,
}

impl TestPtx for Min {
    fn body(&self) -> String {
        let name = format!(
            "min{}{}.f16",
            if self.ftz { ".ftz" } else { "" },
            if self.nan { ".NaN" } else { "" }
        );
        PTX
            .replace("<TYPE_SIZE>", "2")
            .replace("<TYPE>", "f16")
            .replace("<BTYPE>", "b16")
            .replace("<OP>", &name)
    }

    fn args(&self) -> &[&str] {
        &[
            "input_a",
            "input_b",
            "output",
        ]
    }
}

impl TestCommon for Min {
    type Input = (half::f16, half::f16);
    type Output = half::f16;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b) = input;
        let expected = minmax_host(a, b, self.nan, self.ftz, f16::min);
        if (expected.is_nan() && output.is_nan())
            || (expected.to_ne_bytes() == output.to_ne_bytes())
        {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl RangeTest for Min {
    fn generate(&self, input: u32) -> Self::Input {
        unsafe { mem::transmute::<_, (half::f16, half::f16)>(input) }
    }
}

struct Max {
    ftz: bool,
    nan: bool,
}

impl TestPtx for Max {
    fn body(&self) -> String {
        let name = format!(
            "max{}{}.f16",
            if self.ftz { ".ftz" } else { "" },
            if self.nan { ".NaN" } else { "" }
        );
        PTX
            .replace("<TYPE_SIZE>", "2")
            .replace("<TYPE>", "f16")
            .replace("<BTYPE>", "b16")
            .replace("<OP>", &name)
    }

    fn args(&self) -> &[&str] {
        &[
            "input_a",
            "input_b",
            "output",
        ]
    }
}

impl TestCommon for Max {
    type Input = (half::f16, half::f16);
    type Output = half::f16;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b) = input;
        let expected = minmax_host(a, b, self.nan, self.ftz, f16::max);
        if (expected.is_nan() && output.is_nan())
            || (expected.to_ne_bytes() == output.to_ne_bytes())
        {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl RangeTest for Max {
    fn generate(&self, input: u32) -> Self::Input {
        unsafe { mem::transmute::<_, (half::f16, half::f16)>(input) }
    }
}

fn minmax_host(
    mut a: half::f16,
    mut b: half::f16,
    nan: bool,
    ftz: bool,
    fn_: fn(f16, f16) -> f16,
) -> half::f16 {
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
        unsafe {
            mem::transmute(fn_(
                mem::transmute::<_, f16>(a),
                mem::transmute::<_, f16>(b),
            ))
        }
    }
}
