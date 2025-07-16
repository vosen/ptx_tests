use crate::test::{make_random, PtxScalar, RandomTest, TestCase, TestCommon, TestPtx};
use num::{cast::AsPrimitive, traits::WrappingAdd};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::{marker::PhantomData, ops::Shr};

static PTX: &str = include_str!("vshr.ptx");

#[derive(Clone, Copy)]
enum VshrMode {
    Clamp,
    Wrap,
}

#[derive(Clone, Copy)]
enum SecondaryOp {
    Add,
    //Min,
    //Max,
}

struct Vshr<D: PtxScalar, A: PtxScalar> {
    mode: VshrMode,
    sat: bool,
    op2: Option<SecondaryOp>,
    _phantom: PhantomData<(D, A)>,
}

impl<
        D: PtxScalar + num::traits::AsPrimitive<i64> + WrappingAdd,
        A: PtxScalar + num::traits::AsPrimitive<i64> + Shr<u32, Output = A>,
    > TestPtx for Vshr<D, A>
{
    fn body(&self) -> String {
        let dtype = D::name();
        let atype = A::name();
        let sat = if self.sat { ".sat" } else { "" };
        let mode_str = match self.mode {
            VshrMode::Clamp => ".clamp",
            VshrMode::Wrap => ".wrap",
        };
        let (op2_str, op2_args) = match self.op2 {
            None => ("", ""),
            Some(SecondaryOp::Add) => (".add", ", c"),
            //Some(SecondaryOp::Min) => (".min", ", c"),
            //Some(SecondaryOp::Max) => (".max", ", c"),
        };

        // Updated PTX no longer uses .asel/.bsel or merge destination specifiers.
        PTX.replace("<DTYPE>", dtype)
            .replace("<ATYPE>", atype)
            .replace("<SAT>", sat)
            .replace("<MODE>", mode_str)
            .replace("<OP2>", op2_str)
            .replace("<OP2_ARGS>", op2_args)
    }

    fn args(&self) -> &[&str] {
        &["input0", "input1", "input2", "output"]
    }
}

impl<
        D: PtxScalar + num::traits::AsPrimitive<i64> + WrappingAdd + PartialOrd,
        A: PtxScalar + num::traits::AsPrimitive<i64> + Shr<u32, Output = A> + WrappingAdd + PartialOrd,
    > TestCommon for Vshr<D, A>
where
    i64: num::traits::AsPrimitive<A>,
    i64: num::traits::AsPrimitive<D>,
{
    type Input = (A, u32, D);
    type Output = D;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b, c_raw) = input;
        let expected = vshr_host::<D, A>(a, b, c_raw, self);
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl<
        D: PtxScalar + num::traits::AsPrimitive<i64> + WrappingAdd + PartialOrd,
        A: PtxScalar + num::traits::AsPrimitive<i64> + Shr<u32, Output = A> + WrappingAdd + PartialOrd,
    > RandomTest for Vshr<D, A>
where
    Standard: Distribution<A> + Distribution<D>,
    i64: num::traits::AsPrimitive<A>,
    i64: num::traits::AsPrimitive<D>,
{
    fn generate<R: Rng>(&self, rng: &mut R) -> Self::Input {
        (rng.gen::<A>(), rng.next_u32(), rng.gen::<D>())
    }
}

fn vshr_host<
    D: PtxScalar + num::traits::AsPrimitive<i64> + WrappingAdd + PartialOrd,
    A: PtxScalar + num::traits::AsPrimitive<i64> + Shr<u32, Output = A> + WrappingAdd + PartialOrd,
>(
    a: A,
    b: u32,
    c: D,
    config: &Vshr<D, A>,
) -> D
where
    i64: num::traits::AsPrimitive<A>,
    i64: num::traits::AsPrimitive<D>,
{
    let shifted = shift_right(a, b, config.mode);
    let saturated: D = if config.sat {
        if shifted.as_() < D::min_value().as_() {
            D::min_value().as_()
        } else if shifted.as_() > D::max_value().as_() {
            D::max_value().as_()
        } else {
            shifted.as_()
        }
    } else {
        shifted.as_()
    }
    .as_();
    let post_secondary_op = match config.op2 {
        Some(SecondaryOp::Add) => saturated.wrapping_add(&c),
        //Some(SecondaryOp::Min) => {
        //    if saturated < c {
        //        saturated
        //    } else {
        //        c
        //    }
        //}
        //Some(SecondaryOp::Max) => {
        //    if saturated > c {
        //        saturated
        //    } else {
        //        c
        //    }
        //}
        None => saturated,
    };
    post_secondary_op
}

fn shift_right<
    A: PtxScalar + num::traits::AsPrimitive<i64> + Shr<u32, Output = A> + WrappingAdd + PartialOrd,
>(
    a: A,
    b: u32,
    mode: VshrMode,
) -> A {
    let tb = match mode {
        VshrMode::Clamp => {
            if b >= 32 {
                if A::signed() {
                    if a < <A as num::Zero>::zero() {
                        return <A as num::Zero>::zero() - A::one();
                    } else {
                        return <A as num::Zero>::zero();
                    }
                } else {
                    return <A as num::Zero>::zero();
                }
            } else {
                b
            }
        }
        VshrMode::Wrap => b & 0x1f,
    };
    let shifted = a >> tb;
    shifted
}

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    for &mode in &[VshrMode::Clamp, VshrMode::Wrap] {
        for op2 in [
            None,
            Some(SecondaryOp::Add),
            // We are not running min/max tests because their results defy
            // common sense.
            // It looks like starting sm_70 there is no hardware vshr instruction
            // and instead ptx expands to a sequence of instructions. I suspect
            // that sequence is wrong, but I don't have the time to prove it.
            // Some(SecondaryOp::Min),
            // Some(SecondaryOp::Max),
        ] {
            for &sat in &[false, true] {
                tests.push(vshr_test::<u32, i32>(mode, op2, sat));
                tests.push(vshr_test::<u32, u32>(mode, op2, sat));
                tests.push(vshr_test::<i32, i32>(mode, op2, sat));
                tests.push(vshr_test::<i32, u32>(mode, op2, sat));
            }
        }
    }

    tests
}

fn vshr_test<
    D: PtxScalar + num::traits::AsPrimitive<i64> + WrappingAdd + PartialOrd,
    A: PtxScalar + num::traits::AsPrimitive<i64> + Shr<u32, Output = A> + WrappingAdd + PartialOrd,
>(
    mode: VshrMode,
    op2: Option<SecondaryOp>,
    sat: bool,
) -> TestCase
where
    Standard: Distribution<A> + Distribution<D>,
    i64: num::traits::AsPrimitive<A>,
    i64: num::traits::AsPrimitive<D>,
{
    let config = Vshr::<D, A> {
        mode,
        sat,
        op2,
        _phantom: PhantomData,
    };
    let test_name = format!(
        "vshr_{}_{}_u32{}_{}{}",
        D::name(),
        A::name(),
        if sat { "_sat" } else { "" },
        match mode {
            VshrMode::Clamp => "clamp",
            VshrMode::Wrap => "wrap",
        },
        match op2 {
            None => "",
            Some(SecondaryOp::Add) => "_add",
            //Some(SecondaryOp::Min) => "_min",
            //Some(SecondaryOp::Max) => "_max",
        },
    );
    TestCase::new(test_name, make_random(config))
}
