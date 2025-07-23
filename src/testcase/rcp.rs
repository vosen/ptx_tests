use crate::common::{self, flush_to_zero_f32, Rounding};
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};

pub static PTX: &str = include_str!("rcp.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    for ftz in [false, true] {
        tests.push(rcp_rnd(Rounding::Rn, ftz));
        tests.push(rcp_rnd(Rounding::Rz, ftz));
        tests.push(rcp_rnd(Rounding::Rm, ftz));
        tests.push(rcp_rnd(Rounding::Rp, ftz));
        tests.push(rcp_approx(ftz));
    }
    tests
}

fn rcp_rnd(rnd: Rounding, ftz: bool) -> TestCase {
    rcp::<false>(rnd, ftz)
}

fn rcp_approx(ftz: bool) -> TestCase {
    rcp::<true>(Rounding::Default, ftz)
}

fn rcp<const APPROX: bool>(rnd: Rounding, ftz: bool) -> TestCase {
    let test = make_range::<Rcp<APPROX>>(Rcp { rnd, ftz });
    let mode = if APPROX { "approx" } else { rnd.as_str() };
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("rcp_{}{}", mode, ftz), test)
}

pub struct Rcp<const APPROX: bool> {
    ftz: bool,
    rnd: Rounding,
}

impl<const APPROX: bool> TestPtx for Rcp<APPROX> {
    fn body(&self) -> String {
        let rnd = if APPROX { "approx" } else { self.rnd.as_str() };
        let mode = format!("{}{}", rnd, if self.ftz { ".ftz" } else { "" });
        PTX.replace("<MODE>", &mode)
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl<const APPROX: bool> TestCommon for Rcp<APPROX> {
    type Input = f32;

    type Output = f32;

    fn host_verify(
        &self,
        mut input: Self::Input,
        output: Self::Output,
    ) -> Result<(), Self::Output> {
        fn rcp_host(input: f32) -> f64 {
            let input = input as f64;
            input.recip()
        }
        fn rcp_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => -0.0,
                f if f.to_bits() == (-0.0f32).to_bits() => f32::NEG_INFINITY,
                0.0 => f32::INFINITY,
                f32::INFINITY => 0.0,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if APPROX {
            if let Some(mut expected) = rcp_approx_special(input) {
                flush_to_zero_f32(&mut expected, self.ftz);
                if expected.to_bits() == output.to_bits() || expected.is_nan() && output.is_nan() {
                    Ok(())
                } else {
                    Err(expected)
                }
            } else {
                let precise_result = rcp_host(input);
                let mut precise_result_f32 = precise_result as f32;
                flush_to_zero_f32(&mut precise_result_f32, self.ftz);
                common::is_float_equal(precise_result_f32, output, 1)
            }
        } else {
            let mut precise_result = rcp_host(input);
            flush_to_zero_f32(&mut precise_result, self.ftz);
            let result = self.rnd.with_f32(|| precise_result as f32);
            if result.is_nan() && output.is_nan() || result.to_bits() == output.to_bits() {
                Ok(())
            } else {
                Err(result)
            }
        }
    }
}

impl<const APPROX: bool> RangeTest for Rcp<APPROX> {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}
