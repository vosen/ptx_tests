use num::{cast::AsPrimitive, Float};
use std::f32;
use std::mem;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rounding {
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
    pub fn as_llvm(self) -> u32 {
        match self {
            Rounding::Rzi | Rounding::Rz => 0,
            Rounding::Default | Rounding::Rni | Rounding::Rn => 1,
            Rounding::Rpi | Rounding::Rp => 2,
            Rounding::Rmi | Rounding::Rm => 3,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Rounding::Default => "",
            Rounding::Rni => "rni",
            Rounding::Rzi => "rzi",
            Rounding::Rmi => "rmi",
            Rounding::Rpi => "rpi",
            Rounding::Rn => "rn",
            Rounding::Rz => "rz",
            Rounding::Rm => "rm",
            Rounding::Rp => "rp",
        }
    }

    pub fn as_ptx(self) -> &'static str {
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

    pub fn is_integer(self) -> bool {
        match self {
            Rounding::Rzi | Rounding::Rni | Rounding::Rmi | Rounding::Rpi => true,
            Rounding::Default | Rounding::Rz | Rounding::Rn | Rounding::Rm | Rounding::Rp => false,
        }
    }

    pub fn with_f32(&self, f: impl FnOnce() -> f32) -> f32 {
        let old = unsafe { llvm_get_rounding() };
        unsafe { llvm_set_rounding(self.as_llvm()) };
        // Without black_box, the compiler _sometimes_, but not always,
        // moves the function outside of the rounding mode change
        let result = std::hint::black_box(f());
        unsafe { llvm_set_rounding(old) };
        result
    }
}

// That's the easiest and most consistent way to set rounding mode in Rust, sorry
extern "C" {
    #[link_name = "llvm.get.rounding"]
    pub fn llvm_get_rounding() -> u32;
    #[link_name = "llvm.set.rounding"]
    pub fn llvm_set_rounding(r: u32);
}

pub const MAX_NEGATIVE_NORMAL: f32 = f32::from_bits(0x80800000u32);
pub const MIN_POSITIVE_NORMAL: f32 = f32::from_bits(0x00800000u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Lo,
    Ls,
    Hi,
    Hs,
    Equ,
    Neu,
    Ltu,
    Leu,
    Gtu,
    Geu,
    Num,
    Nan,
}

impl Comparison {
    pub fn as_str(self) -> &'static str {
        match self {
            Comparison::Eq => "eq",
            Comparison::Ne => "ne",
            Comparison::Lt => "lt",
            Comparison::Le => "le",
            Comparison::Gt => "gt",
            Comparison::Ge => "ge",
            Comparison::Lo => "lo",
            Comparison::Ls => "ls",
            Comparison::Hi => "hi",
            Comparison::Hs => "hs",
            Comparison::Equ => "equ",
            Comparison::Neu => "neu",
            Comparison::Ltu => "ltu",
            Comparison::Leu => "leu",
            Comparison::Gtu => "gtu",
            Comparison::Geu => "geu",
            Comparison::Num => "num",
            Comparison::Nan => "nan",
        }
    }

    pub fn iter_int() -> impl Iterator<Item = Comparison> {
        [
            Comparison::Eq,
            Comparison::Ne,
            Comparison::Lt,
            Comparison::Le,
            Comparison::Gt,
            Comparison::Ge,
            Comparison::Lo,
            Comparison::Ls,
            Comparison::Hi,
            Comparison::Hs,
        ]
        .iter()
        .copied()
    }

    pub fn signed(self) -> bool {
        match self {
            Comparison::Eq
            | Comparison::Ne
            | Comparison::Lt
            | Comparison::Le
            | Comparison::Gt
            | Comparison::Ge => true,
            _ => false,
        }
    }

    pub fn iter_float() -> impl Iterator<Item = Comparison> {
        [
            Comparison::Eq,
            Comparison::Ne,
            Comparison::Lt,
            Comparison::Le,
            Comparison::Gt,
            Comparison::Ge,
            Comparison::Equ,
            Comparison::Neu,
            Comparison::Ltu,
            Comparison::Leu,
            Comparison::Gtu,
            Comparison::Geu,
            Comparison::Num,
            Comparison::Nan,
        ]
        .iter()
        .copied()
    }

    pub fn cmp_int<T>(self, a: T, b: T) -> bool
    where
        T: PartialOrd + PartialEq,
    {
        match self {
            Comparison::Eq => a == b,
            Comparison::Ne => a != b,
            Comparison::Lt => a < b,
            Comparison::Le => a <= b,
            Comparison::Gt => a > b,
            Comparison::Ge => a >= b,
            Comparison::Lo => a < b,
            Comparison::Ls => a <= b,
            Comparison::Hi => a > b,
            Comparison::Hs => a >= b,
            Comparison::Equ => a == b,
            Comparison::Neu => a != b,
            Comparison::Ltu => a < b,
            Comparison::Leu => a <= b,
            Comparison::Gtu => a > b,
            Comparison::Geu => a >= b,
            Comparison::Num => true,
            Comparison::Nan => false,
        }
    }

    pub fn cmp_float<T>(self, a: T, b: T, ftz: bool) -> bool
    where
        T: PartialOrd + PartialEq + AsPrimitive<f32>,
    {
        let mut a_val: f32 = a.as_();
        let mut b_val: f32 = b.as_();
        flush_to_zero_f32(&mut a_val, ftz);
        flush_to_zero_f32(&mut b_val, ftz);
        match self {
            Comparison::Eq => a_val == b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Ne => a_val != b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Lt => a_val < b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Le => a_val <= b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Gt => a_val > b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Ge => a_val >= b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Lo => a_val < b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Ls => a_val <= b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Hi => a_val > b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Hs => a_val >= b_val && !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Equ => {
                if a_val.is_nan() || b_val.is_nan() {
                    true
                } else {
                    a_val == b_val
                }
            }
            Comparison::Neu => {
                if a_val.is_nan() || b_val.is_nan() {
                    true
                } else {
                    a_val != b_val
                }
            }
            Comparison::Ltu => {
                if a_val.is_nan() || b_val.is_nan() {
                    true
                } else {
                    a_val < b_val
                }
            }
            Comparison::Leu => {
                if a_val.is_nan() || b_val.is_nan() {
                    true
                } else {
                    a_val <= b_val
                }
            }
            Comparison::Gtu => {
                if a_val.is_nan() || b_val.is_nan() {
                    true
                } else {
                    a_val > b_val
                }
            }
            Comparison::Geu => {
                if a_val.is_nan() || b_val.is_nan() {
                    true
                } else {
                    a_val >= b_val
                }
            }
            Comparison::Num => !a_val.is_nan() && !b_val.is_nan(),
            Comparison::Nan => a_val.is_nan() || b_val.is_nan(),
        }
    }
}

pub const MAX_NEGATIVE_SUBNORMAL: f32 = f32::from_bits(0x807FFFFFu32);
pub const MAX_POSITIVE_SUBNORMAL: f32 = f32::from_bits(0x007FFFFFu32);
pub const MIN_POSITIVE_SUBNORMAL: f32 = f32::from_bits(0x00000001u32);
pub const MIN_NEGATIVE_SUBNORMAL: f32 = f32::from_bits(0x80000001u32);
//pub const SIGNALING_NAN: f32 = unsafe { mem::transmute(0x7F800001u32) };

pub const MAX_NEGATIVE_SUBNORMAL_F16: half::f16 = unsafe { mem::transmute(0x83FFu16) };
pub const MAX_POSITIVE_SUBNORMAL_F16: half::f16 = unsafe { mem::transmute(0x03FFu16) };

//pub const MAX_NEGATIVE_SUBNORMAL_F64: f64 = unsafe { mem::transmute(0x800FFFFFFFFFFFFFu64) };
//pub const MAX_POSITIVE_SUBNORMAL_F64: f64 = unsafe { mem::transmute(0x000FFFFFFFFFFFFFu64) };
//pub const SIGNALING_NAN_F64: f64 = unsafe { mem::transmute(0x7FF0000000000001u64) };

pub fn flush_to_zero_f32<T: Float + Copy + 'static>(x: &mut T, ftz: bool)
where
    f32: AsPrimitive<T>,
{
    if !ftz {
        return;
    }
    if *x < T::neg_zero() && *x > MAX_NEGATIVE_NORMAL.as_() {
        *x = T::neg_zero()
    } else if *x > T::zero() && *x < MIN_POSITIVE_NORMAL.as_() {
        *x = T::zero()
    }
}

pub fn flush_to_zero_f16<T: Float + Copy + 'static>(x: &mut T, ftz: bool)
where
    half::f16: AsPrimitive<T>,
{
    if !ftz {
        return;
    }
    if *x < T::neg_zero() && *x >= MAX_NEGATIVE_SUBNORMAL_F16.as_() {
        *x = T::neg_zero()
    } else if *x > T::zero() && *x <= MAX_POSITIVE_SUBNORMAL_F16.as_() {
        *x = T::zero()
    }
}

pub trait WideningMul: Sized {
    fn widening_mul(self, b: Self) -> (Self, Self);
}

macro_rules! widening_mul_impl {
    ($($t:ty => $u:ty),+) => {
        $(
            impl WideningMul for $t {
                fn widening_mul(self, b: $t) -> ($t, $t) {
                    assert_eq!(std::mem::size_of::<$u>(), std::mem::size_of::<$t>() * 2);
                    let res = (self as $u) * (b as $u);
                    (res as $t, (res >> std::mem::size_of::<$t>() * 8) as $t)
                }
            }
        )+
    };
}

widening_mul_impl! {
    i8 => i16,
    i16 => i32,
    i32 => i64,
    u8 => u16,
    u16 => u32,
    u32 => u64
}

pub fn relative_diff(precise_result: f64, gpu_result: f64, epsilon: f64) -> bool {
    if precise_result.to_bits() == gpu_result.to_bits()
        || precise_result.is_nan() && gpu_result.is_nan()
    {
        return true;
    }
    let diff = (precise_result - gpu_result).abs();
    diff <= epsilon * precise_result
}

pub fn is_within_sincos_bounds(input: f32, diff: f64) -> bool {
    const SMALL_TOLERANCE: f64 = 6.74349576174304508591503488640641249928317964256738697613086575503698577147534865784829871333905016295916483989196918976271074321675652128526745556402175822931872685397866054895305609942032233267102942518978105357155930774019652373661287984199947928719307102790109781605873799724465658E-7f64; // 2^-20.5
    const LARGE_TOLERANCE: f64 = 0.00003757154581741077528379495450341012908557549319765679304007158014667055419065424982922432707674847844356659399733246764458976456844014372423408856194127577921238887982199164153968150905768262057536124294850876976021494026469646993054912722508303245485051436772572465845031419928132242244f64; // 2^-14.7

    if input >= f32::consts::PI * -2.0 && input <= f32::consts::PI * 2.0 {
        diff <= SMALL_TOLERANCE
    } else if input >= f32::consts::PI * -100.0 && input <= f32::consts::PI * 100.0 {
        diff <= LARGE_TOLERANCE
    } else {
        true
    }
}

pub fn is_float_equal(exact_f32: f32, output: f32, expected_ulp: u32) -> Result<(), f32> {
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
