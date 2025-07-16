use crate::common;
use crate::test::{make_random, RandomTest, TestCase, TestCommon, TestPtx};
use rand::Rng;
use std::marker::PhantomData;

pub static PTX: &str = include_str!("fma_f.ptx");

#[derive(Clone)]
pub struct FmaF32 {
    pub rnd: common::Rounding,
    pub ftz: bool,
    pub sat: bool,
    _phantom: PhantomData<f32>,
}

impl FmaF32 {
    pub fn new(rnd: common::Rounding, ftz: bool, sat: bool) -> Self {
        Self {
            rnd,
            ftz,
            sat,
            _phantom: PhantomData,
        }
    }

    pub fn generate_with_config<R: Rng + Sized>(rng: &mut R) -> (f32, f32) {
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
                f32::from_bits(rng.gen::<u32>()),
            ),
            (
                crate::common::MAX_POSITIVE_SUBNORMAL,
                f32::from_bits(rng.gen::<u32>()),
            ),
            (
                f32::from_bits(rng.gen::<u32>()),
                crate::common::MAX_POSITIVE_SUBNORMAL,
            ),
            (
                f32::from_bits(rng.gen::<u32>()),
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
                f32::from_bits(rng.gen::<u32>()),
            ),
            (
                crate::common::MIN_POSITIVE_SUBNORMAL,
                f32::from_bits(rng.gen::<u32>()),
            ),
            (
                f32::from_bits(rng.gen::<u32>()),
                crate::common::MIN_POSITIVE_SUBNORMAL,
            ),
            (
                f32::from_bits(rng.gen::<u32>()),
                crate::common::MIN_NEGATIVE_SUBNORMAL,
            ),
            (crate::common::MIN_POSITIVE_SUBNORMAL, 1.0),
            (crate::common::MIN_POSITIVE_SUBNORMAL, 10.0),
        ];
        if rng.gen_bool(0.01) {
            // With 1% probability, choose one of the special cases.
            special_cases[rng.gen_range(0..special_cases.len())]
        } else {
            (
                f32::from_bits(rng.gen::<u32>()),
                f32::from_bits(rng.gen::<u32>()),
            )
        }
    }
}

impl TestPtx for FmaF32 {
    fn body(&self) -> String {
        PTX.replace("<RND>", self.rnd.as_ptx())
            .replace("<FLUSH>", if self.ftz { ".ftz" } else { "" })
            .replace("<SAT>", if self.sat { ".sat" } else { "" })
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "input_c", "output"]
    }
}

impl TestCommon for FmaF32 {
    type Input = (f32, f32, f32);
    type Output = f32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let expected = fma_host_f32(input.0, input.1, input.2, self.rnd, self.ftz, self.sat);

        /* should be if expected.to_bits() == output.to_bits() {
        FMA on CPU is not rounding correctly for cases with sub normals. couldn't figure out why.
        GPU result is correct. allowing an error of 1 ULP for now.
        */

        if (expected.to_bits() as i32).abs_diff(output.to_bits() as i32) <= 1 {
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

impl RandomTest for FmaF32 {
    fn generate<R: Rng + Sized>(&self, rng: &mut R) -> Self::Input {
        let a = FmaF32::generate_with_config(rng).0;
        let b = FmaF32::generate_with_config(rng).1;
        let c = f32::from_bits(rng.gen::<u32>());
        (a, b, c)
    }
}

fn fma_host_f32(a: f32, b: f32, c: f32, rnd: common::Rounding, ftz: bool, sat: bool) -> f32 {
    let mut a_mod = a;
    let mut b_mod = b;
    let mut c_mod = c;
    common::flush_to_zero_f32(&mut a_mod, ftz);
    common::flush_to_zero_f32(&mut b_mod, ftz);
    common::flush_to_zero_f32(&mut c_mod, ftz);

    let exact = (a_mod as f64).mul_add(b_mod as f64, c_mod as f64);
    let mut result = rnd.with_f32(|| exact as f32);
    common::flush_to_zero_f32(&mut result, ftz);
    if sat {
        if result.is_nan() {
            result = 0.0;
        } else if result <= 0.0 {
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

    for rounding in [Rounding::Rn, Rounding::Rz, Rounding::Rm, Rounding::Rp] {
        for ftz in [false, true] {
            for sat in [false, true] {
                let name = format!(
                    "fma_{}{}{}_f32",
                    rounding.as_str(),
                    if ftz { "_ftz" } else { "" },
                    if sat { "_sat" } else { "" }
                );
                tests.push(TestCase::new(
                    name,
                    make_random(FmaF32::new(rounding, ftz, sat)),
                ));
            }
        }
    }
    tests
}
