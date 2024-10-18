use std::mem;

use num::{cast::AsPrimitive, Float};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
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

    pub fn with<T>(&self, f: impl FnOnce() -> T) -> T {
        let old = unsafe { llvm_get_rounding() };
        unsafe { llvm_set_rounding(self.as_llvm()) };
        let result = f();
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

const MAX_NEGATIVE_SUBNORMAL: f32 = unsafe { mem::transmute(0x807FFFFFu32) };
const MAX_POSITIVE_SUBNORMAL: f32 = unsafe { mem::transmute(0x007FFFFFu32) };

const MAX_NEGATIVE_SUBNORMAL_F16: half::f16 = unsafe { mem::transmute(0x83FFu16) };
const MAX_POSITIVE_SUBNORMAL_F16: half::f16 = unsafe { mem::transmute(0x03FFu16) };

pub fn flush_to_zero_f32<T: Float + Copy + 'static>(x: &mut T, ftz: bool)
where
    f32: AsPrimitive<T>,
{
    if !ftz {
        return;
    }
    if *x < T::neg_zero() && *x >= MAX_NEGATIVE_SUBNORMAL.as_() {
        *x = T::neg_zero()
    } else if *x > T::zero() && *x <= MAX_POSITIVE_SUBNORMAL.as_() {
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
