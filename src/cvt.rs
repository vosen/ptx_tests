use crate::{
    cuda::Cuda,
    test::{self, PtxScalar, ResultMismatch, TestCase, TestCommon},
};
use num::traits::AsPrimitive;
use num::traits::ConstOne;
use num::traits::ConstZero;
use num::Float;
use std::mem;

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
    ($vec:expr, $invalid:expr) => {
        gen_test!(
            $vec,
            $invalid,
            [
                Rounding::Default,
                Rounding::Rni,
                Rounding::Rzi,
                Rounding::Rmi,
                Rounding::Rpi,
                Rounding::Rn,
                Rounding::Rz,
                Rounding::Rm,
                Rounding::Rp
            ]
        );
    };
    ($vec:expr, $invalid:expr, [$($rnd:expr),*]) => {
        $(
            gen_test!($vec, $invalid, $rnd, [false, true]);
        )*
    };
    ($vec:expr, $invalid:expr, $rnd:expr, [$($ftz:expr),*]) => {
        $(
            gen_test!($vec, $invalid, $rnd, $ftz, [false, true]);
        )*
    };
    ($vec:expr, $invalid:expr, $rnd:expr, $ftz:expr, [$($sat:expr),*]) => {
        $(
            gen_test!($vec, $invalid, $rnd, $ftz, $sat, [i16, u16, i32, u32, half::f16, f32]);
        )*
    };
    ($vec:expr, $invalid:expr, $rnd:expr, $ftz:expr, $sat:expr, [$($input:ty),*]) => {
        $(
            gen_test!($vec, $invalid, $rnd, $ftz, $sat, $input, [i16, u16, i32, u32, i64, u64, half::f16, f32, f64]);
        )*
    };
    ($vec:expr, $invalid:expr, $rnd:expr, $ftz:expr, $sat:expr, $input:ty, [$($output:ty),*]) => {
        $(
            {
                let (name, test) = test_case::<$output, $input>($rnd, $ftz, $sat);
                if is_invalid_cvt::<$output, $input>($rnd.as_ptx(), $ftz, $sat) {
                    $invalid.push((name, test));
                } else {
                    $vec.push(test::TestCase::new(name, test));
                }
            }
        )*
    };
}

struct Cvt<To: PtxScalar, From: PtxScalar> {
    rnd: Rounding,
    ftz: bool,
    sat: bool,
    _phantom: std::marker::PhantomData<(To, From)>,
}

impl<To: PtxScalar, From: PtxScalar> Cvt<To, From> {
    fn new(rnd: Rounding, ftz: bool, sat: bool) -> Self {
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
        <Self::Input as HostConvert<Self::Output>>::convert(input, self.rnd, self.ftz, self.sat, output)
    }

    fn ptx(&self) -> String {
        let src = include_str!("cvt.ptx");
        let ftz = if self.ftz { ".ftz" } else { "" };
        let sat = if self.sat { ".sat" } else { "" };
        let rnd = self.rnd.as_ptx();
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
        !is_invalid_cvt::<To, From>(self.rnd.as_ptx(), self.ftz, self.sat)
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
    rnd: Rounding,
    ftz: bool,
    sat: bool,
) -> (
    String,
    Box<dyn FnOnce(&Cuda) -> Result<bool, ResultMismatch>>,
) {
    let rnd_txt = match rnd {
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
    (name, test)
}

pub(super) fn all_tests() -> Vec<TestCase> {
    let mut result = Vec::new();
    let mut invalid_tests = Vec::new();
    gen_test!(result, invalid_tests);
    result.push(TestCase::join_invalid_tests(
        "cvt_invalid".to_string(),
        invalid_tests,
    ));
    result
}

trait HostConvert<To: PtxScalar>: Copy {
    fn convert(self, rnd: Rounding, ftz: bool, sat: bool, expected: To) -> Result<(), To>;
}

// That's the easiest and most consistent way to set rounding mode in Rust, sorry
extern "C" {
    #[link_name = "llvm.get.rounding"]
    fn llvm_get_rounding() -> u32;
    #[link_name = "llvm.set.rounding"]
    fn llvm_set_rounding(r: u32);
}

// IMPORTANT: This is a hack!
// We use this trait purely to transmute half::f16 into f16 type to make sure that rustc emits
// llvm assemblt. If we use half::f16, the library will emit its own x86 inline assembly,
// which will ignore rounding mode set in LLVM
// Using f16 directly is an even bigger problem because num-traits does not support it.
trait ConvertAs<T> {
    fn as_hack(self) -> T;
}

macro_rules! convert_as {
    () => {
        convert_as! { [i16, u16, i32, u32, f32] }
    };
    ([$($input:ty),*]) => {
        $(
            convert_as! { $input, [i16, u16, i32, u32, i64, u64, f32, f64] }
        )*
    };
    ($input:ty, [$($output:ty),*]) => {
        $(
            impl ConvertAs<$output> for $input {
                fn as_hack(self) -> $output {
                    self.as_()
                }
            }
        )*

        impl ConvertAs<half::f16> for $input {
            fn as_hack(self) -> half::f16 {
                unsafe { mem::transmute(self as f16) }
            }
        }
    }
}

convert_as!();

macro_rules! convert_as_from_f16 {
    () => {
        impl ConvertAs<half::f16> for half::f16 {
            fn as_hack(self) -> half::f16 {
                self
            }
        }

        convert_as_from_f16! { [i16, u16, i32, u32, i64, u64, f32, f64]}
    };
    ([$($output:ty),*]) => {
        $(
            impl ConvertAs<$output> for half::f16 {
                fn as_hack(self) -> $output {
                    (unsafe { mem::transmute::<_, f16>(self) }) as $output
                }
            }
        )*
    };
}

convert_as_from_f16!();

macro_rules! as_hack {
    ([$($input:ty),*]) => {
        $(
            as_hack! { $input, [half::f16, f32, f64] }
        )*
    };
    ($input:ty, [$($output:ty),*]) => {
        $(
            impl HostConvert<$output> for $input {
                fn convert(mut self, rnd: Rounding, ftz: bool, sat: bool, expected: $output) -> Result<(), $output> {
                    use num::traits::AsPrimitive;
                    if float_to_float_input_ftz::<$output, $input>(ftz, rnd) && self.is_subnormal() {
                        if self < <$input>::ZERO {
                            self = <$input>::neg_zero();
                        } else if self > <$input>::ZERO {
                            self = <$input>::ZERO;
                        }
                    }
                    let mut host_result = if rnd.is_integer() && mem::size_of::<$input>() == mem::size_of::<$output>() {
                        FloatAsInteger::round(self, rnd).as_()
                    } else {
                        let env_rnd = unsafe { llvm_get_rounding() };
                        unsafe { llvm_set_rounding(rnd.as_llvm()) };
                        let host_result: $output = self.as_hack();
                        unsafe { llvm_set_rounding(env_rnd) };
                        host_result
                    };
                    if ftz && mem::size_of::<$output>() == 4 {
                        if host_result.is_subnormal() {
                            if host_result < <$output>::ZERO {
                                host_result = <$output>::neg_zero();
                            } else if host_result > <$output>::ZERO {
                                host_result = <$output>::ZERO;
                            }
                        }
                    }
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

fn float_to_float_input_ftz<To, From>(ftz: bool, rnd: Rounding) -> bool {
    if ftz && mem::size_of::<From>() == 4 {
        if mem::size_of::<To>() == 2 {
            // ERRATA: <rnd>.ftz{.sat}.f16.f32 flushes subnormals only if <rnd> is explicit integer rounding
            rnd.is_integer()
        } else {
            true
        }
    } else {
        false
    }
}

as_hack!([half::f16, f32]);

macro_rules! float_to_int {
    ([$($input:ty),*]) => {
        $(
            float_to_int! { $input, [i16, u16, i32, u32, i64, u64] }
        )*
    };
    ($input:ty, [$($output:ty),*]) => {
        $(
            impl HostConvert<$output> for $input {
                fn convert(mut self, rnd: Rounding, ftz: bool, _sat: bool, expected: $output) -> Result<(), $output> {
                    use num::traits::AsPrimitive;
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

macro_rules! int_to_int {
    ([$($input:ty),*]) => {
        $(
            int_to_int! { $input, [i16, u16, i32, u32, i64, u64] }
        )*
    };
    ($input:ty, [$($output:ty),*]) => {
        $(
            impl HostConvert<$output> for $input {
                fn convert(self, _rnd: Rounding, ftz: bool, sat: bool, expected: $output) -> Result<(), $output> {
                    assert!(!ftz);
                    let result: $output = if sat {
                        if (self as i128) <= (<$output>::MIN as i128) {
                            <$output>::MIN
                        } else if (self as i128) >= (<$output>::MAX as i128)  {
                            <$output>::MAX
                        } else {
                            self.as_hack()
                        }
                    } else {
                        self.as_hack()
                    };
                    if result.to_ne_bytes() != expected.to_ne_bytes() {
                        Err(result)
                    } else {
                        Ok(())
                    }
                }
            }
        )*
    };
}

int_to_int!([i16, u16, i32, u32]);

macro_rules! int_to_float {
    ([$($input:ty),*]) => {
        $(
            int_to_float! { $input, [half::f16, f32, f64] }
        )*
    };
    ($input:ty, [$($output:ty),*]) => {
        $(
            impl HostConvert<$output> for $input {
                fn convert(self, rnd: Rounding, _ftz: bool, sat: bool, expected: $output) -> Result<(), $output> {
                    let env_rnd = unsafe { llvm_get_rounding() };
                    unsafe { llvm_set_rounding(rnd.as_llvm()) };
                    let mut host_result: $output = self.as_hack();
                    unsafe { llvm_set_rounding(env_rnd) };
                    if sat {
                         if host_result <= <$output>::ZERO {
                            host_result = <$output>::ZERO;
                        } else if host_result > <$output>::ONE {
                            host_result = <$output>::ONE;
                        }
                    }
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

int_to_float!([i16, u16, i32, u32]);
