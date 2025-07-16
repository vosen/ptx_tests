use num::cast::AsPrimitive;
use num::traits::FromPrimitive;

use crate::common::Comparison;
use crate::test::{
    make_random, make_range, PtxScalar, RandomTest, RangeTest, TestCase, TestCommon, TestPtx,
};
use std::marker::PhantomData;
use std::mem;

static SET_BOOL: &'static str = include_str!("set_bool.ptx");
static SET: &'static str = include_str!("set.ptx");

#[derive(Clone, Copy)]
pub enum BoolOp {
    And,
    Or,
    Xor,
}

impl BoolOp {
    pub fn as_str(self) -> &'static str {
        match self {
            BoolOp::And => "and",
            BoolOp::Or => "or",
            BoolOp::Xor => "xor",
        }
    }
}

pub struct SetIntTest<T, U> {
    cmp_op: Comparison,
    bool_input: Option<(BoolOp, bool)>,
    _marker: PhantomData<(T, U)>,
}

impl<T, U> SetIntTest<T, U> {
    pub fn new(cmp_op: Comparison, optional_input: Option<(BoolOp, bool)>) -> Self {
        SetIntTest {
            cmp_op,
            bool_input: optional_input,
            _marker: PhantomData,
        }
    }
}

impl<T, U> TestPtx for SetIntTest<T, U>
where
    T: crate::test::PtxScalar,
    U: crate::test::PtxScalar,
{
    fn body(&self) -> String {
        format_set::<T, U>(self.cmp_op, self.bool_input)
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "output_d"]
    }
}

fn format_set<T: PtxScalar, U: PtxScalar>(
    cmp_op: Comparison,
    bool_input: Option<(BoolOp, bool)>,
) -> String {
    let (bool_op, op_suffix) = match bool_input {
        Some((bool_op, bool_value)) => (
            format!(".{}", bool_op.as_str()),
            format!(", {}", bool_value as u8),
        ),
        None => ("".to_string(), "".to_string()),
    };
    SET.replace("<STYPE>", T::name())
        .replace("<DTYPE>", U::name())
        .replace("<STYPE_SIZE>", &mem::size_of::<T>().to_string())
        .replace("<DTYPE_SIZE>", &mem::size_of::<U>().to_string())
        .replace("<CMP>", cmp_op.as_str())
        .replace("<BOOL_OP>", &bool_op)
        .replace("<OP_SUFFIX>", &op_suffix)
}

fn format_set_bool<T: PtxScalar>(cmp_op: Comparison, bool_op: BoolOp, ftz: bool) -> String {
    let ftz = if ftz { ".ftz" } else { "" };
    SET_BOOL
        .replace("<TYPE>", T::name())
        .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string())
        .replace("<CMP>", cmp_op.as_str())
        .replace("<BOOL_OP>", bool_op.as_str())
        .replace("<FTZ>", ftz)
}

impl<T, U> TestCommon for SetIntTest<T, U>
where
    T: crate::test::PtxScalar + PartialEq + PartialOrd + Copy + AsPrimitive<f32>,
    U: crate::test::PtxScalar + PartialEq + FromPrimitive + Copy,
{
    type Input = (T, T);
    type Output = U;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        verify(self.cmp_op, false, input, self.bool_input, output)
    }
}

fn verify<T, U>(
    cmp_op: Comparison,
    ftz: bool,
    input: (T, T),
    optional_input: Option<(BoolOp, bool)>,
    output: U,
) -> Result<(), U>
where
    T: crate::test::PtxScalar + PartialEq + PartialOrd + Copy + AsPrimitive<f32>,
    U: crate::test::PtxScalar + PartialEq + FromPrimitive + Copy,
{
    let (a, b) = input;

    let cmp_result = if T::float() {
        cmp_op.cmp_float(a, b, ftz)
    } else {
        cmp_op.cmp_int(a, b)
    };

    let combined = match optional_input {
        Some((bool_op, predicate)) => match bool_op {
            BoolOp::And => cmp_result && predicate,
            BoolOp::Or => cmp_result || predicate,
            BoolOp::Xor => cmp_result != predicate,
        },
        None => cmp_result,
    };

    let expected = if U::float() {
        if combined {
            FromPrimitive::from_u32(1).unwrap()
        } else {
            FromPrimitive::from_u32(0).unwrap()
        }
    } else {
        if combined {
            FromPrimitive::from_u32(0xffffffff).unwrap()
        } else {
            FromPrimitive::from_u32(0).unwrap()
        }
    };

    if output == expected {
        Ok(())
    } else {
        Err(expected)
    }
}

// We do range test for integer, sweep all possible 16 bit combinations
impl<T, U> RangeTest for SetIntTest<T, U>
where
    T: crate::test::PtxScalar + FromU16 + PartialOrd + AsPrimitive<f32> + Copy,
    U: crate::test::PtxScalar + PartialEq + FromPrimitive + Copy,
{
    const MAX_VALUE: u32 = u32::MAX;
    fn generate(&self, input: u32) -> Self::Input {
        let a_bits = (input >> 16) as u16;
        let b_bits = (input & 0xffff) as u16;
        let a_val = T::from_u16(a_bits);
        let b_val = T::from_u16(b_bits);
        (a_val, b_val)
    }
}

struct SetFloatTest {
    cmp_op: Comparison,
    bool_op: BoolOp,
    ftz: bool,
}

impl TestPtx for SetFloatTest {
    fn body(&self) -> String {
        format_set_bool::<f32>(self.cmp_op, self.bool_op, self.ftz)
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "input_c", "output_d"]
    }
}

impl TestCommon for SetFloatTest {
    type Input = (f32, f32, bool);
    type Output = f32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b, predicate) = input;
        verify(
            self.cmp_op,
            self.ftz,
            (a, b),
            Some((self.bool_op, predicate)),
            output,
        )
    }
}

impl RandomTest for SetFloatTest {
    fn generate<R: rand::Rng>(&self, rng: &mut R) -> Self::Input {
        let special_cases = [
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
            (crate::common::MAX_POSITIVE_SUBNORMAL, 2.0),
        ];
        if rng.gen_bool(0.01) {
            let (x, y) = special_cases[rng.gen_range(0..special_cases.len())];
            (x, y, rng.gen())
        } else {
            (
                f32::from_bits(rng.gen()),
                f32::from_bits(rng.gen()),
                rng.gen(),
            )
        }
    }
}

pub trait FromU16 {
    fn from_u16(n: u16) -> Self;
}
impl FromU16 for u16 {
    fn from_u16(n: u16) -> Self {
        n
    }
}
impl FromU16 for i16 {
    fn from_u16(n: u16) -> Self {
        n as i16
    }
}

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = Vec::new();

    for cmp in Comparison::iter_int() {
        tests.push(TestCase::new(
            format!("set_{}_u32_u16", cmp.as_str()),
            make_range(SetIntTest::<u16, u32>::new(cmp, None)),
        ));
        if cmp.signed() {
            tests.push(TestCase::new(
                format!("set_{}_u32_s16", cmp.as_str()),
                make_range(SetIntTest::<i16, u32>::new(cmp, None)),
            ));
        }
        for bool_op in [BoolOp::And, BoolOp::Or, BoolOp::Xor] {
            for integer_predicate in [true, false] {
                tests.push(TestCase::new(
                    format!(
                        "set_{}_{}_u32_u16_{}",
                        cmp.as_str(),
                        bool_op.as_str(),
                        integer_predicate.to_string()
                    ),
                    make_range(SetIntTest::<u16, u32>::new(
                        cmp,
                        Some((bool_op, integer_predicate)),
                    )),
                ));
                if !cmp.signed() {
                    continue;
                }
                tests.push(TestCase::new(
                    format!(
                        "set_{}_{}_u32_s16_{}",
                        cmp.as_str(),
                        bool_op.as_str(),
                        integer_predicate.to_string()
                    ),
                    make_range(SetIntTest::<i16, u32>::new(
                        cmp,
                        Some((bool_op, integer_predicate)),
                    )),
                ));
            }
        }
    }

    for cmp_op in Comparison::iter_float() {
        for ftz in [true, false] {
            let ftz_text = if ftz { "_ftz" } else { "" };
            for bool_op in [BoolOp::And, BoolOp::Or, BoolOp::Xor] {
                tests.push(TestCase::new(
                    format!(
                        "set_{}_{}{}_f32_f32",
                        cmp_op.as_str(),
                        bool_op.as_str(),
                        ftz_text
                    ),
                    make_random(SetFloatTest {
                        cmp_op,
                        bool_op,
                        ftz,
                    }),
                ));
            }
        }
    }

    tests
}
