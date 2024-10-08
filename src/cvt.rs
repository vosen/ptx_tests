use crate::{
    cuda::{CUresult, Cuda},
    test::PtxScalar,
};
use std::{mem, ptr};

pub fn run_many(
    cuda: &Cuda,
    modifiers: &[&str],
    input: &str,
    input_size: &str,
    output: &str,
    output_size: &str,
) -> CUresult {
    let src = include_str!("cvt.ptx");
    let modifiers = modifiers.join("");
    let input_bits = input_size.parse::<u32>().unwrap() * 8;
    let output_bits = output_size.parse::<u32>().unwrap() * 8;
    let mut src = src
        .replace("<INPUT>", input)
        // PTX disallows ld.f16, but allows ld.b16 and implictly converts to f16
        .replace("<INPUT_LD>", &format!("b{input_bits}"))
        .replace("<INPUT_SIZE>", input_size)
        .replace("<OUTPUT>", output)
        .replace("<OUTPUT_ST>", &format!("b{output_bits}"))
        .replace("<OUTPUT_SIZE>", output_size)
        .replace("<MODIFIERS>", &modifiers);
    src.push('\0');
    let mut module = ptr::null_mut();
    let result = unsafe { cuda.cuModuleLoadData(&mut module, src.as_ptr() as _) };
    if result.is_ok() {
        unsafe { cuda.cuModuleUnload(module) }.unwrap();
    }
    result
}

fn run_one<Output: PtxScalar, Input: PtxScalar>(cuda: &Cuda, rounding: &str, ftz: bool, sat: bool) {
    let valid = !is_invalid_cvt::<Output, Input>(rounding, ftz, sat);
    let src = include_str!("cvt.ptx");
    let ftz = if ftz { ".ftz" } else { "" };
    let sat = if sat { ".sat" } else { "" };
    let modifiers = format!("{}{}{}", rounding, ftz, sat);
    let input_bits = mem::size_of::<Input>() * 8;
    let output_bits = mem::size_of::<Output>() * 8;
    let mut src = src
        .replace("<INPUT>", Input::name())
        // PTX disallows ld.f16, but allows ld.b16 and implictly converts to f16
        .replace("<INPUT_LD>", &format!("b{input_bits}"))
        .replace("<INPUT_SIZE>", &mem::size_of::<Input>().to_string())
        .replace("<OUTPUT>", Output::name())
        .replace("<OUTPUT_ST>", &format!("b{output_bits}"))
        .replace("<OUTPUT_SIZE>", &mem::size_of::<Output>().to_string())
        .replace("<MODIFIERS>", &modifiers);
    src.push('\0');
    let mut module = ptr::null_mut();
    let result = unsafe { cuda.cuModuleLoadData(&mut module, src.as_ptr() as _) };
    if result.is_ok() {
        unsafe { cuda.cuModuleUnload(module) }.unwrap();
    }
    let result = match result {
        Ok(_) => 0,
        Err(x) => x.get(),
    };
    if valid && result != 0 || !valid && result == 0 {
        eprintln!(
            "{:?}, {valid}, {result}",
            [&modifiers, Output::name(), Input::name()]
        );
    }
}

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
    ($cuda:expr) => {
        gen_test!($cuda, ["", ".rni", ".rzi", ".rmi", ".rpi", ".rn", ".rz", ".rm", ".rp"]);
    };
    ($cuda:expr, [$($rnd:expr),*]) => {
        $(
            gen_test!($cuda, $rnd, [false, true]);
        )*
    };
    ($cuda:expr, $rnd:expr, [$($ftz:expr),*]) => {
        $(
            gen_test!($cuda, $rnd, $ftz, [false, true]);
        )*
    };
    ($cuda:expr, $rnd:expr, $ftz:expr, [$($sat:expr),*]) => {
        $(
            gen_test!($cuda, $rnd, $ftz, $sat, [i16, u16, i32, u32, half::f16, f32]);
        )*
    };
    ($cuda:expr, $rnd:expr, $ftz:expr, $sat:expr, [$($input:ty),*]) => {
        $(
            gen_test!($cuda, $rnd, $ftz, $sat, $input, [i16, u16, i32, u32, i64, u64, half::f16, f32, f64]);
        )*
    };
    ($cuda:expr, $rnd:expr, $ftz:expr, $sat:expr, $input:ty, [$($output:ty),*]) => {
        $(
            run_one::<$output, $input>($cuda, $rnd, $ftz, $sat);
        )*
    };
}

pub fn run(cuda: &Cuda) {
    gen_test!(cuda);
}
