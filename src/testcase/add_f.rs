use crate::common;
use crate::test::{make_random, RandomTest, TestCase, TestCommon, TestPtx};
use rand::Rng;
use std::marker::PhantomData;

pub static PTX: &str = include_str!("add_f.ptx");

#[derive(Clone)]
pub struct AddF32 {
    pub rnd: common::Rounding,
    pub ftz: bool,
    pub sat: bool,
    _phantom: PhantomData<f32>,
}

impl AddF32 {
    pub fn new(rnd: common::Rounding, ftz: bool, sat: bool) -> Self {
        Self {
            rnd,
            ftz,
            sat,
            _phantom: PhantomData,
        }
    }
}

impl TestPtx for AddF32 {
    fn body(&self) -> String {
        PTX.replace("<RND>", self.rnd.as_ptx())
            .replace("<FLUSH>", if self.ftz { ".ftz" } else { "" })
            .replace("<SAT>", if self.sat { ".sat" } else { "" })
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

impl TestCommon for AddF32 {
    type Input = (f32, f32);
    type Output = f32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let expected = add_host_f32(input.0, input.1, self.rnd, self.ftz, self.sat);
        if expected.to_bits() == output.to_bits() {
            Ok(())
        } else if expected.is_nan() && output.is_nan() {
            Ok(())
        } else if expected == 0.0 && output == 0.0 {
            // Allow +0.0 and -0.0 to be treated as equivalent.
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl RandomTest for AddF32 {
    fn generate<R: Rng + Sized>(&self, rng: &mut R) -> Self::Input {
        let special_cases: [(f32, f32); 23] = [
            (0.0, f32::INFINITY),
            (f32::INFINITY, 0.0),
            (f32::INFINITY, -0.0),
            (f32::NEG_INFINITY, 0.0),
            (f32::NEG_INFINITY, -0.0),
            (f32::NAN, 0.0),
            (f32::NAN, -0.0),
            (0.0, f32::NAN),
            (f32::NAN, 1.0),
            (
                crate::common::MAX_NEGATIVE_SUBNORMAL,
                crate::common::MAX_POSITIVE_SUBNORMAL,
            ),
            (
                crate::common::MAX_NEGATIVE_SUBNORMAL,
                f32::from_bits(rng.gen()),
            ),
            (
                crate::common::MAX_POSITIVE_SUBNORMAL,
                f32::from_bits(rng.gen()),
            ),
            (
                f32::from_bits(rng.gen()),
                crate::common::MAX_POSITIVE_SUBNORMAL,
            ),
            (
                f32::from_bits(rng.gen()),
                crate::common::MAX_NEGATIVE_SUBNORMAL,
            ),
            (crate::common::MAX_POSITIVE_SUBNORMAL, 1.0),
            (crate::common::MAX_POSITIVE_SUBNORMAL, 10.0),
            (
                crate::common::MIN_NEGATIVE_SUBNORMAL,
                crate::common::MIN_POSITIVE_SUBNORMAL,
            ),
            (
                crate::common::MIN_NEGATIVE_SUBNORMAL,
                f32::from_bits(rng.gen()),
            ),
            (
                crate::common::MIN_POSITIVE_SUBNORMAL,
                f32::from_bits(rng.gen()),
            ),
            (
                f32::from_bits(rng.gen()),
                crate::common::MIN_POSITIVE_SUBNORMAL,
            ),
            (
                f32::from_bits(rng.gen()),
                crate::common::MIN_NEGATIVE_SUBNORMAL,
            ),
            (crate::common::MIN_POSITIVE_SUBNORMAL, 1.0),
            (crate::common::MIN_POSITIVE_SUBNORMAL, 10.0),
        ];
        if rng.gen_bool(0.01) {
            // With 1% probability, choose one of the special cases.
            special_cases[rng.gen_range(0..special_cases.len())]
        } else {
            (f32::from_bits(rng.gen()), f32::from_bits(rng.gen()))
        }
    }
}

fn add_host_f32(a: f32, b: f32, rnd: common::Rounding, ftz: bool, sat: bool) -> f32 {
    let mut a_mod = a;
    let mut b_mod = b;
    common::flush_to_zero_f32(&mut a_mod, ftz);
    common::flush_to_zero_f32(&mut b_mod, ftz);
    let exact = (a_mod as f64) + (b_mod as f64);
    let mut result = rnd.with_f32(|| exact as f32);
    common::flush_to_zero_f32(&mut result, ftz);
    if sat {
        if result.is_nan() {
            result = 0.0;
        } else if result < 0.0 {
            result = 0.0;
        } else if result > 1.0 {
            result = 1.0;
        }
    }
    result
}

pub fn all_tests() -> Vec<TestCase> {
    use common::Rounding;
    let mut tests = Vec::new();

    let roundings: [Rounding; 4] = [Rounding::Rn, Rounding::Rz, Rounding::Rm, Rounding::Rp];
    let ftz_options = [false, true];
    let sat_options = [false, true];

    for &rounding in &roundings {
        for &ftz in &ftz_options {
            for &sat in &sat_options {
                let name = format!(
                    "add_{}{}{}_f32",
                    rounding.as_str(),
                    if ftz { "_ftz" } else { "" },
                    if sat { "_sat" } else { "" }
                );
                tests.push(TestCase::new(
                    name,
                    make_random(AddF32::new(rounding, ftz, sat)),
                ));
            }
        }
    }
    tests
}
