use num::{cast::AsPrimitive, Bounded, Num, PrimInt};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use crate::cuda::*;
use std::{mem, ptr};

pub static PTX: &str = include_str!("bfe2.ptx");

pub(super) fn u32() -> bool {
    test::<u32>()
}
pub(super) fn s32() -> bool {
    test::<i32>()
}
pub(super) fn u64() -> bool {
    test::<u64>()
}
pub(super) fn s64() -> bool {
    test::<i64>()
}

const RANDOM_ELEMENTS_COUNT: usize = 2usize.pow(14);
const THREADS: usize = 2usize.pow(8) * 2usize.pow(8);
const SEED: u64 = 0x761194f3027874ef;

fn test<T: crate::test::PtxScalar>() -> bool
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    let mut src = PTX
        .replace("<TYPE>", T::name())
        .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string());
    src.push('\0');
    let cuda = Cuda::new();
    unsafe { cuda.cuInit(0) }.unwrap();
    let mut ctx = ptr::null_mut();
    unsafe { cuda.cuCtxCreate_v2(&mut ctx, 0, 0) }.unwrap();
    let mut module = ptr::null_mut();
    unsafe { cuda.cuModuleLoadData(&mut module, src.as_ptr() as _) }.unwrap();
    let mut inputs = Vec::with_capacity(3 + RANDOM_ELEMENTS_COUNT);
    inputs.push(T::zero());
    inputs.push(T::min_value());
    inputs.push(T::max_value());
    let mut rng = XorShiftRng::seed_from_u64(SEED);
    for _ in 0..RANDOM_ELEMENTS_COUNT {
        inputs.push(rng.gen());
    }
    let mut dev_input = unsafe { mem::zeroed() };
    unsafe { cuda.cuMemAlloc_v2(&mut dev_input, inputs.len() * mem::size_of::<T>()) }.unwrap();
    unsafe {
        cuda.cuMemcpyHtoD_v2(
            dev_input,
            inputs.as_ptr() as _,
            inputs.len() * mem::size_of::<T>(),
        )
    }
    .unwrap();
    let mut dev_output = unsafe { mem::zeroed() };
    unsafe {
        cuda.cuMemAlloc_v2(
            &mut dev_output,
            THREADS * inputs.len() * mem::size_of::<T>(),
        )
    }
    .unwrap();
    let mut kernel = ptr::null_mut();
    unsafe { cuda.cuModuleGetFunction(&mut kernel, module, c"bfe".as_ptr()) }.unwrap();
    let count_elements = inputs.len() as u64;
    let mut args = [&dev_input, &count_elements, &dev_output];
    unsafe {
        cuda.cuLaunchKernel(
            kernel,
            (THREADS / 128) as u32,
            1,
            1,
            128,
            1,
            1,
            0,
            0 as _,
            args.as_mut_ptr() as _,
            ptr::null_mut(),
        )
    }
    .unwrap();
    let mut result = vec![T::zero(); THREADS * inputs.len()];
    unsafe {
        cuda.cuMemcpyDtoH_v2(
            result.as_mut_ptr() as _,
            dev_output,
            result.len() * mem::size_of::<T>(),
        )
    }
    .unwrap();
    unsafe { cuda.cuStreamSynchronize(0 as _) }.unwrap();
    unsafe { cuda.cuMemFree_v2(dev_input) }.unwrap();
    unsafe { cuda.cuMemFree_v2(dev_output) }.unwrap();
    unsafe { cuda.cuModuleUnload(module) }.unwrap();
    for global_id in 0..THREADS {
        for (i, value) in inputs.iter().copied().enumerate() {
            todo!()
        }
    }
    true
}
