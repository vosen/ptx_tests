use crate::{
    common,
    test::{make_random, RandomTest, TestCase, TestCommon, TestPtx},
};
use rand::Rng;
use std::marker::PhantomData;

static PTX: &str = include_str!("div.ptx");

#[derive(Clone, Copy)]
enum DivVariant {
    Approx,
    Full,
    Rnd(common::Rounding),
}

impl DivVariant {
    fn rounding(self) -> Option<common::Rounding> {
        match self {
            DivVariant::Approx => None,
            DivVariant::Full => None,
            DivVariant::Rnd(rnd) => Some(rnd),
        }
    }
}

#[derive(Clone)]
struct DivF32 {
    variant: DivVariant,
    ftz: bool,
    _phantom: PhantomData<f32>,
}

impl DivF32 {
    fn new(variant: DivVariant, ftz: bool) -> Self {
        Self {
            variant,
            ftz,
            _phantom: PhantomData,
        }
    }
}

impl TestPtx for DivF32 {
    fn body(&self) -> String {
        let variant_str = match self.variant {
            DivVariant::Approx => ".approx".to_string(),
            DivVariant::Full => ".full".to_string(),
            DivVariant::Rnd(ref rnd) => rnd.as_ptx().to_string(),
        };
        PTX.replace("<VARIANT>", &variant_str)
            .replace("<FLUSH>", if self.ftz { ".ftz" } else { "" })
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output"]
    }
}

impl TestCommon for DivF32 {
    type Input = (f32, f32);
    type Output = f32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (mut a, mut b) = input;
        common::flush_to_zero_f32(&mut a, self.ftz);
        common::flush_to_zero_f32(&mut b, self.ftz);
        let mut exact_f64 = (a as f64) / (b as f64);
        common::flush_to_zero_f32(&mut exact_f64, self.ftz);
        let exact_f32 = match self.variant.rounding() {
            Some(rnd) => rnd.with_f32(|| {
                let result = exact_f64 as f32;
                result
            }),
            None => exact_f64 as f32,
        };
        match self.variant {
            DivVariant::Full => is_float_equal(exact_f32, output, 2),
            DivVariant::Rnd(_) => is_float_equal(exact_f32, output, 0),
            DivVariant::Approx => is_approx_equal(a, b, exact_f32, output),
        }
    }
}

fn is_approx_equal(a: f32, b: f32, exact_f32: f32, gpu_output: f32) -> Result<(), f32> {
    if exact_f32.is_nan() && gpu_output.is_nan() {
        return Ok(());
    }
    if exact_f32 == f32::INFINITY && gpu_output == f32::INFINITY {
        return Ok(());
    }
    if exact_f32 == f32::NEG_INFINITY && gpu_output == f32::NEG_INFINITY {
        return Ok(());
    }
    let lower_bound = 2.0_f64.powi(-126);
    let upper_bound = 2.0_f64.powi(126);
    let b_abs = b.abs() as f64;
    if b_abs < lower_bound {
        // PTX docs don't explicitly state what is the expected precision here,
        // so we just accept anything
        Ok(())
    } else if b_abs <= upper_bound {
        is_float_equal(exact_f32, gpu_output, 2)
    } else {
        if a.is_infinite() {
            if gpu_output.is_nan() {
                Ok(())
            } else {
                Err(f32::NAN)
            }
        } else {
            if gpu_output == 0.0 {
                Ok(())
            } else {
                Err(0.0)
            }
        }
    }
}

fn is_float_equal(exact_f32: f32, output: f32, expected_ulp: u32) -> Result<(), f32> {
    if exact_f32.is_nan() && output.is_nan() {
        return Ok(());
    }
    let exact_bits = exact_f32.to_bits();
    let ulp = exact_bits.abs_diff(output.to_bits());
    if ulp <= expected_ulp {
        Ok(())
    } else {
        Err(exact_f32)
    }
}

impl RandomTest for DivF32 {
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
            special_cases[rng.gen_range(0..special_cases.len())]
        } else {
            (
                f32::from_bits(rng.gen::<u32>()),
                f32::from_bits(rng.gen::<u32>()),
            )
        }
    }
}

pub fn all_tests() -> Vec<TestCase> {
    use common::Rounding;
    let mut tests = Vec::new();
    for ftz in [false, true] {
        for variant in [
            DivVariant::Approx,
            DivVariant::Full,
            DivVariant::Rnd(Rounding::Rn),
            DivVariant::Rnd(Rounding::Rz),
            DivVariant::Rnd(Rounding::Rm),
            DivVariant::Rnd(Rounding::Rp),
        ] {
            let variant_name = match variant {
                DivVariant::Approx => "approx".to_string(),
                DivVariant::Full => "full".to_string(),
                DivVariant::Rnd(ref r) => format!("{}", r.as_str()),
            };
            let name = format!("div_{}{}_f32", variant_name, if ftz { "_ftz" } else { "" });
            tests.push(TestCase::new(name, make_random(DivF32::new(variant, ftz))));
        }
    }
    tests
}
