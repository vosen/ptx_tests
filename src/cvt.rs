use crate::{
    cuda::{CUresult, Cuda},
    test::{self, PtxScalar, TestCase, TestCommon},
};
use num::traits::AsPrimitive;
use num::{traits::ToBytes, Float, Integer};
use std::{
    any::{self, Any, TypeId},
    cmp::Ordering,
    mem, ptr,
};

fn is_invalid_cvt<Output: PtxScalar, Input: PtxScalar>(
    rounding: &str,
    ftz: bool,
    sat: bool,
) -> bool {
    if sat {
        if Input::signed()
            && Output::signed()
            && mem::size_of::<Output>() >= mem::size_of::<Input>()
        {
            return true;
        }
        if Input::unsigned()
            && Output::unsigned()
            && mem::size_of::<Output>() >= mem::size_of::<Input>()
        {
            return true;
        }
        if Input::unsigned()
            && Output::signed()
            && mem::size_of::<Output>() > mem::size_of::<Input>()
        {
            return true;
        }
    }
    if ftz {
        if !Output::is_f32() && !Input::is_f32() {
            return true;
        }
    }
    if rounding.is_empty()
        && Output::float()
        && Input::float()
        && mem::size_of::<Output>() < mem::size_of::<Input>()
    {
        return true;
    }
    if rounding.is_empty() && Output::float() && !Input::float() {
        return true;
    }
    if rounding.is_empty() && !Output::float() && Input::float() {
        return true;
    }
    if rounding.ends_with('i') {
        if !(Input::float() && !Output::float()
            || Input::float()
                && Output::float()
                && mem::size_of::<Output>() == mem::size_of::<Input>())
        {
            return true;
        }
    } else if rounding.starts_with('.') {
        if !(!Input::float() && Output::float()
            || Input::float()
                && Output::float()
                && mem::size_of::<Output>() < mem::size_of::<Input>())
        {
            return true;
        }
    }
    false
}

macro_rules! gen_test {
    ($vec:expr) => {
        gen_test!($vec, [0,1,2,3,4,5,6,7,8]);
    };
    ($vec:expr, [$($rnd:expr),*]) => {
        $(
            gen_test!($vec, $rnd, [false, true]);
        )*
    };
    ($vec:expr, $rnd:expr, [$($ftz:expr),*]) => {
        $(
            gen_test!($vec, $rnd, $ftz, [false, true]);
        )*
    };
    ($vec:expr, $rnd:expr, $ftz:expr, [$($sat:expr),*]) => {
        $(
            gen_test!($vec, $rnd, $ftz, $sat, [i16, u16, i32, u32, half::f16, f32]);
        )*
    };
    ($vec:expr, $rnd:expr, $ftz:expr, $sat:expr, [$($input:ty),*]) => {
        $(
            gen_test!($vec, $rnd, $ftz, $sat, $input, [i16, u16, i32, u32, i64, u64, half::f16, f32, f64]);
        )*
    };
    ($vec:expr, $rnd:expr, $ftz:expr, $sat:expr, $input:ty, [$($output:ty),*]) => {
        $(
            $vec.push(test_case::<$output, $input>($rnd, $ftz, $sat));
        )*
    };
}

struct Cvt<To: PtxScalar, From: PtxScalar> {
    rnd: u8,
    ftz: bool,
    sat: bool,
    _phantom: std::marker::PhantomData<(To, From)>,
}

impl<To: PtxScalar, From: PtxScalar> Cvt<To, From> {
    fn new(rnd: u8, ftz: bool, sat: bool) -> Self {
        Self {
            rnd,
            ftz,
            sat,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<To: PtxScalar, From: PtxScalar + HostConvert<To>> TestCommon for Cvt<To, From> {
    type Input = From;

    type Output = To;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let rnd = unsafe { mem::transmute::<_, Rounding>(self.rnd) };
        <Self::Input as HostConvert<Self::Output>>::convert(input, rnd, self.ftz, self.sat, output)
    }

    fn ptx(&self) -> String {
        let src = include_str!("cvt.ptx");
        let ftz = if self.ftz { ".ftz" } else { "" };
        let sat = if self.sat { ".sat" } else { "" };
        let rnd = unsafe { mem::transmute::<_, Rounding>(self.rnd) };
        let rnd = rnd.as_ptx();
        let modifiers = format!("{}{}{}", rnd, ftz, sat);
        let input_bits = mem::size_of::<Self::Input>() * 8;
        let output_bits = mem::size_of::<Self::Output>() * 8;
        let mut src = src
            .replace("<INPUT>", Self::Input::name())
            // PTX disallows ld.half::f16, but allows ld.b16 and implictly converts to half::f16
            .replace("<INPUT_LD>", &format!("b{input_bits}"))
            .replace("<INPUT_SIZE>", &mem::size_of::<Self::Input>().to_string())
            .replace("<OUTPUT>", Self::Output::name())
            .replace("<OUTPUT_ST>", &format!("b{output_bits}"))
            .replace("<OUTPUT_SIZE>", &mem::size_of::<Self::Output>().to_string())
            .replace("<MODIFIERS>", &modifiers);
        src.push('\0');
        src
    }
}

impl<To: PtxScalar, From: PtxScalar + HostConvert<To>> test::RangeTest for Cvt<To, From> {
    const MAX_VALUE: u32 = (2usize.pow((mem::size_of::<From>() * 8) as u32) - 1) as u32;

    fn generate(&self, input: u32) -> Self::Input {
        unsafe {
            match mem::size_of::<From>() {
                2 => mem::transmute_copy(&(input as u16)),
                4 => mem::transmute_copy(&input),
                _ => unreachable!(),
            }
        }
    }
    fn is_valid(&self) -> bool {
        let rnd = unsafe { mem::transmute::<_, Rounding>(self.rnd) };
        !is_invalid_cvt::<To, From>(rnd.as_ptx(), self.ftz, self.sat)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rounding {
    Default = 0,
    Rni,
    Rzi,
    Rmi,
    Rpi,
    Rn,
    Rz,
    Rm,
    Rp,
}

impl Rounding {
    fn as_llvm(self) -> u32 {
        match self {
            Rounding::Rzi | Rounding::Rz => 0,
            Rounding::Default | Rounding::Rni | Rounding::Rn => 1,
            Rounding::Rpi | Rounding::Rp => 2,
            Rounding::Rmi | Rounding::Rm => 3,
        }
    }

    fn as_ptx(self) -> &'static str {
        match self {
            Rounding::Default => "",
            Rounding::Rni => ".rni",
            Rounding::Rzi => ".rzi",
            Rounding::Rmi => ".rmi",
            Rounding::Rpi => ".rpi",
            Rounding::Rn => ".rn",
            Rounding::Rz => ".rz",
            Rounding::Rm => ".rm",
            Rounding::Rp => ".rp",
        }
    }

    fn is_integer(self) -> bool {
        match self {
            Rounding::Rzi | Rounding::Rni | Rounding::Rmi | Rounding::Rpi => true,
            Rounding::Default | Rounding::Rz | Rounding::Rn | Rounding::Rm | Rounding::Rp => false,
        }
    }
}

fn test_case<To: PtxScalar, From: PtxScalar + HostConvert<To>>(
    rnd: u8,
    ftz: bool,
    sat: bool,
) -> TestCase {
    let rnd_typed = unsafe { mem::transmute::<_, Rounding>(rnd) };
    let rnd_txt = match rnd_typed {
        Rounding::Default => "",
        Rounding::Rni => "_rni",
        Rounding::Rzi => "_rzi",
        Rounding::Rmi => "_rmi",
        Rounding::Rpi => "_rpi",
        Rounding::Rn => "_rn",
        Rounding::Rz => "_rz",
        Rounding::Rm => "_rm",
        Rounding::Rp => "_rp",
    };
    let ftz_txt = if ftz { "_ftz" } else { "" };
    let sat_txt = if sat { "_sat" } else { "" };
    let name = format!(
        "cvt{rnd_txt}{ftz_txt}{sat_txt}_{}_{}",
        To::name(),
        From::name()
    );
    let test = Box::new(move |cuda: &Cuda| {
        test::run_range::<Cvt<To, From>>(cuda, Cvt::<To, From>::new(rnd, ftz, sat))
    });
    TestCase { test, name }
}

pub(super) fn all_tests() -> Vec<TestCase> {
    let mut result = Vec::new();
    gen_test!(result);
    result
}

trait HostConvert<To: PtxScalar>: Copy {
    fn convert(self, rnd: Rounding, ftz: bool, sat: bool, expected: To) -> Result<(), To>;
}

macro_rules! unimplemented_convert {
    () => {
        unimplemented_convert!{ [i16, u16, i32, u32/*, half::f16,  f32 */] }
    };
    ([$($input:ty),*]) => {
        $(
            unimplemented_convert! {$input,  [i16, u16, i32, u32, i64, u64, half::f16, f32, f64] }
        )*
    };
    ($input:ty, [$($output:ty),*]) => {
        $(
            impl HostConvert<$output> for $input {
                fn convert(self, _: Rounding, _: bool, _: bool, expected: $output) -> Result<(), $output> {
                    unimplemented!()
                }
            }
        )*
    };
}

unimplemented_convert!();

// That's the easiest and most consistent way to set rounding mode in Rust, sorry
extern "C" {
    #[link_name = "llvm.get.rounding"]
    fn llvm_get_rounding() -> u32;
    #[link_name = "llvm.set.rounding"]
    fn llvm_set_rounding(r: u32);
}

trait FloatToFloat<T> {
    fn float_to_float(self) -> T;
}

impl FloatToFloat<half::f16> for f32 {
    fn float_to_float(self) -> half::f16 {
        // IMPORTANT: This is a hack!
        // We use unstable f16 type to make sure that rustc emits fptrunc
        // If we use half::f16, the library will emit its own x86 inline assembly,
        // which will ignore rounding mode set in LLVM
        unsafe { mem::transmute(self as f16) }
    }
}

impl FloatToFloat<f32> for f32 {
    fn float_to_float(self) -> Self {
        self
    }
}

impl FloatToFloat<f64> for f32 {
    fn float_to_float(self) -> f64 {
        self as _
    }
}

impl FloatToFloat<half::f16> for half::f16 {
    fn float_to_float(self) -> Self {
        self
    }
}

impl FloatToFloat<f32> for half::f16 {
    fn float_to_float(self) -> f32 {
        self.as_()
    }
}

impl FloatToFloat<f64> for half::f16 {
    fn float_to_float(self) -> f64 {
        self.as_()
    }
}

macro_rules! float_to_float {
    ([$($input:ty),*]) => {
        $(
            float_to_float! { $input, [half::f16, f32, f64] }
        )*
    };
    ($input:ty, [$($output:ty),*]) => {
        $(
            impl HostConvert<$output> for $input {
                fn convert(mut self, rnd: Rounding, ftz: bool, sat: bool, expected: $output) -> Result<(), $output> {
                    use num::traits::AsPrimitive;
                    use num::traits::ConstZero;
                    use num::traits::ConstOne;
                    let mut host_result = if rnd.is_integer() && mem::size_of::<$input>() == mem::size_of::<$output>() {
                        FloatAsInteger::round(self, rnd).as_()
                    } else {
                        // ERRATA: .ftz with explicit rounding mode does _not_ flush input zeros
                        if ftz && (mem::size_of::<$input>() == 4 && rnd == Rounding::Default) {
                            if self.is_subnormal() {
                                if self < <$input>::ZERO {
                                    self = <$input>::neg_zero();
                                } else if self > <$input>::ZERO {
                                    self = <$input>::ZERO;
                                }
                            }
                        }
                        let env_rnd = unsafe { llvm_get_rounding() };
                        unsafe {llvm_set_rounding(rnd.as_llvm()) };
                        let mut host_result: $output = self.float_to_float();
                        unsafe {llvm_set_rounding(env_rnd) };
                        if ftz && (mem::size_of::<$output>() == 4) {
                            if host_result.is_subnormal() {
                                if host_result < <$output>::ZERO {
                                    host_result = <$output>::neg_zero();
                                } else if host_result > <$output>::ZERO {
                                    host_result = <$output>::ZERO;
                                }
                            }
                        }
                        host_result
                    };
                    if sat {
                        if host_result.is_nan() {
                            host_result = <$output>::ZERO
                        } else if host_result <= <$output>::ZERO {
                            host_result = <$output>::ZERO;
                        } else if host_result > <$output>::ONE {
                            host_result = <$output>::ONE;
                        }
                    }
                    if host_result.is_nan() && expected.is_nan() {
                        return Ok(());
                    }
                    if host_result.to_ne_bytes() != expected.to_ne_bytes() {
                        Err(host_result)
                    } else {
                        Ok(())
                    }
                }
            }
        )*
    }
}

float_to_float!([half::f16, f32]);

macro_rules! float_to_int {
    ([$($input:ty),*]) => {
        $(
            float_to_int! { $input, [i16, u16, i32, u32, i64, u64] }
        )*
    };
    ($input:ty, [$($output:ty),*]) => {
        $(
            impl HostConvert<$output> for $input {
                fn convert(mut self, rnd: Rounding, ftz: bool, sat: bool, expected: $output) -> Result<(), $output> {
                    use num::traits::AsPrimitive;
                    use num::traits::ConstZero;
                    if ftz {
                        if self.is_subnormal() {
                            self = <$input>::ZERO;
                        }
                    }
                    let host_result: $output = if self.is_nan() {
                        // We don't check NaN conversion, even on NV GPUs it is
                        // not consistent
                        return Ok(())
                    }  else {
                        FloatAsInteger::round(self, rnd).as_()
                    };
                    //let rounded = FloatAsInteger::round(self, rnd);
                    //let host_result: $output = unsafe { mem::transmute::<_, <$input as Underlying>::Target>(rounded) as $output };
                    if host_result.to_ne_bytes() != expected.to_ne_bytes() {
                        Err(host_result)
                    } else {
                        Ok(())
                    }
                }
            }
        )*
    };
}

float_to_int!([half::f16, f32]);

trait FloatAsInteger {
    fn round(self, mode: Rounding) -> Self;
}

impl FloatAsInteger for f32 {
    fn round(self, mode: Rounding) -> Self {
        let rnd_fn = match mode {
            Rounding::Default | Rounding::Rni => f32::round_ties_even,
            Rounding::Rzi => f32::trunc,
            Rounding::Rmi => f32::floor,
            Rounding::Rpi => f32::ceil,
            Rounding::Rn => f32::round_ties_even,
            Rounding::Rz => f32::trunc,
            Rounding::Rm => f32::floor,
            Rounding::Rp => f32::ceil,
        };
        rnd_fn(self)
    }
}

impl FloatAsInteger for half::f16 {
    fn round(self, mode: Rounding) -> Self {
        let this = unsafe { mem::transmute::<_, f16>(self) };
        let rnd_fn = match mode {
            Rounding::Default | Rounding::Rni => f16::round_ties_even,
            Rounding::Rzi => f16::trunc,
            Rounding::Rmi => f16::floor,
            Rounding::Rpi => f16::ceil,
            Rounding::Rn => f16::round_ties_even,
            Rounding::Rz => f16::trunc,
            Rounding::Rm => f16::floor,
            Rounding::Rp => f16::ceil,
        };
        unsafe { mem::transmute::<_, half::f16>(rnd_fn(this)) }
    }
}
