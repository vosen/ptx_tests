use crate::common::{self, flush_to_zero_f32};
use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};

static PTX: &str = include_str!("rsqrt.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    for ftz in [false, true] {
        tests.push(rsqrt_approx(ftz));
    }
    tests
}

fn rsqrt_approx(ftz: bool) -> TestCase {
    let test = make_range(RsqrtApprox { ftz });
    let ftz = if ftz { "_ftz" } else { "" };
    TestCase::new(format!("rsqrt_approx{}", ftz), test)
}

struct RsqrtApprox {
    ftz: bool,
}

// "The maximum relative error for rsqrt.f32 over the entire positive finite floating-point range is 2-22.9."
const APPROX_TOLERANCE: f64 = 1.2776535302833237221396044791023040090875442087382755082640417434504573808163034373723357248111817477540886837817901624231062114398687081918865940855621736724999638430445291615667628609824212040748813550777310962679825276633957875861658012096775892733271105877905219477797519782545596659799487701878423017956258395925986530565459699106333603994393783465705938473841832970991142331162001327764331637031886166711551941504203926569957270914995323038918130406238546875538077885589871273927095747970568232228733689613831029602179447092784745613357178831749672637491728947253198E-7; // 2^-22.9

impl TestPtx for RsqrtApprox {
    fn body(&self) -> String {
        let mode = format!("approx{}", if self.ftz { ".ftz" } else { "" });
        PTX.replace("<MODE>", &mode)
    }

    fn args(&self) -> &[&str] {
        &["input", "output"]
    }
}

impl TestCommon for RsqrtApprox {
    type Input = f32;

    type Output = f32;

    fn host_verify(
        &self,
        mut input: Self::Input,
        output: Self::Output,
    ) -> Result<(), Self::Output> {
        fn rsqrt_approx_special(input: f32) -> Option<f32> {
            Some(match input {
                f32::NEG_INFINITY => f32::NAN,
                f if f.to_bits() == (-0.0f32).to_bits() => f32::NEG_INFINITY,
                f if f.is_finite() && f.is_sign_negative() => f32::NAN,
                0.0 => f32::INFINITY,
                f32::INFINITY => 0.0,
                f if f.is_nan() => f32::NAN,
                _ => return None,
            })
        }
        flush_to_zero_f32(&mut input, self.ftz);
        if let Some(mut expected) = rsqrt_approx_special(input) {
            flush_to_zero_f32(&mut expected, self.ftz);
            if expected.to_bits() == output.to_bits() || expected.is_nan() && output.is_nan() {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let precise_result = rsqrt_host(input);
            if common::relative_diff(precise_result, output as f64, APPROX_TOLERANCE) {
                Ok(())
            } else {
                Err(precise_result as f32)
            }
        }
    }
}

impl RangeTest for RsqrtApprox {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, input: u32) -> Self::Input {
        f32::from_bits(input)
    }
}

fn rsqrt_host(input: f32) -> f64 {
    let input = input as f64;
    input.sqrt().recip()
}
