use float8::{F8E4M3, F8E5M2};
use num::{Bounded, Num, PrimInt, Zero};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::{any::Any, fmt::Debug, mem, ptr, u32};

use crate::{
    cuda::{CUmodule, Cuda},
    TestContext,
};

struct CudaModule<'a> {
    cuda: &'a Cuda,
    value: CUmodule,
}

impl<'a> Drop for CudaModule<'a> {
    fn drop(&mut self) {
        unsafe { self.cuda.cuModuleUnload(self.value) }.unwrap();
    }
}

struct DevicePtr<'a> {
    cuda: &'a Cuda,
    value: u64,
}

impl<'a> Drop for DevicePtr<'a> {
    fn drop(&mut self) {
        unsafe { self.cuda.cuMemFree_v2(self.value) }.unwrap();
    }
}

pub trait TestPtx {
    fn args(&self) -> &[&str];
    fn body(&self) -> String;
}

pub trait TestCommon: TestPtx {
    type Input: OnDevice + DebugRich;
    type Output: OnDevice + DebugRich;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output>;
}

pub trait RangeTest: TestCommon {
    const MAX_VALUE: u32 = u32::MAX;
    fn generate(&self, input: u32) -> Self::Input;
}

pub trait RandomTest: TestCommon {
    fn generate<R: Rng>(&self, rng: &mut R) -> Self::Input;
}

pub trait OnDevice: Copy + Debug {
    const COMPONENTS: usize;
    fn write(self, buffers: &mut [Vec<u8>]);
    fn read(buffers: &[Vec<u8>], index: usize) -> Self;
    fn size_of() -> usize {
        mem::size_of::<Self>()
    }
    fn zero() -> Self {
        unsafe { mem::zeroed::<Self>() }
    }
}

impl OnDevice for bool {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        <u8 as OnDevice>::write(self as u8, buffers)
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        <u8 as OnDevice>::read(buffers, index) != 0
    }
}

impl OnDevice for u8 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}

impl OnDevice for i8 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}

impl OnDevice for u16 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}

impl OnDevice for i16 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}

impl OnDevice for u32 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for i32 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for u64 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for i64 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for F8E4M3 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].push(self.to_bits());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for F8E5M2 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].push(self.to_bits());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for half::f16 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for f32 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for f64 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl<X: OnDevice, Y: OnDevice> OnDevice for (X, Y) {
    const COMPONENTS: usize = 2;

    fn write(self, buffers: &mut [Vec<u8>]) {
        self.0.write(&mut buffers[0..]);
        self.1.write(&mut buffers[1..]);
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        (X::read(&buffers[0..], index), Y::read(&buffers[1..], index))
    }

    fn size_of() -> usize {
        X::size_of() + Y::size_of()
    }

    fn zero() -> Self {
        (X::zero(), Y::zero())
    }
}

impl<X: OnDevice, Y: OnDevice, Z: OnDevice> OnDevice for (X, Y, Z) {
    const COMPONENTS: usize = 3;

    fn write(self, buffers: &mut [Vec<u8>]) {
        self.0.write(&mut buffers[0..]);
        self.1.write(&mut buffers[1..]);
        self.2.write(&mut buffers[2..]);
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        (
            X::read(&buffers[0..], index),
            Y::read(&buffers[1..], index),
            Z::read(&buffers[2..], index),
        )
    }

    fn size_of() -> usize {
        X::size_of() + Y::size_of() + Z::size_of()
    }

    fn zero() -> Self {
        (X::zero(), Y::zero(), Z::zero())
    }
}

impl<X: OnDevice, Y: OnDevice, Z: OnDevice, W: OnDevice> OnDevice for (X, Y, Z, W) {
    const COMPONENTS: usize = 4;

    fn write(self, buffers: &mut [Vec<u8>]) {
        self.0.write(&mut buffers[0..]);
        self.1.write(&mut buffers[1..]);
        self.2.write(&mut buffers[2..]);
        self.3.write(&mut buffers[3..]);
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        (
            X::read(&buffers[0..], index),
            Y::read(&buffers[1..], index),
            Z::read(&buffers[2..], index),
            W::read(&buffers[3..], index),
        )
    }

    fn size_of() -> usize {
        X::size_of() + Y::size_of() + Z::size_of() + W::size_of()
    }

    fn zero() -> Self {
        (X::zero(), Y::zero(), Z::zero(), W::zero())
    }
}

pub trait DebugRich {
    fn debug_rich(&self) -> String;
}

macro_rules! impl_debug_rich {
    ($type:ident) => {
        impl DebugRich for $type {
            fn debug_rich(&self) -> String {
                format!("{self:#066b} {self:#X} {self}")
            }
        }
    };
}

impl DebugRich for bool {
    fn debug_rich(&self) -> String {
        self.to_string()
    }
}

impl_debug_rich!(u8);
impl_debug_rich!(i8);
impl_debug_rich!(u16);
impl_debug_rich!(i16);
impl_debug_rich!(u32);
impl_debug_rich!(i32);
impl_debug_rich!(u64);
impl_debug_rich!(i64);

impl DebugRich for F8E4M3 {
    fn debug_rich(&self) -> String {
        format!("{:#066b} {self:#X} {self:.24}", self.to_bits())
    }
}

impl DebugRich for F8E5M2 {
    fn debug_rich(&self) -> String {
        format!("{:#066b} {self:#X} {self:.24}", self.to_bits())
    }
}

impl DebugRich for f16 {
    fn debug_rich(&self) -> String {
        format!("{0:#066b} {0:#X} {self:.24}", self.to_bits())
    }
}

impl DebugRich for half::f16 {
    fn debug_rich(&self) -> String {
        format!("{self:#066b} {self:#X} {self:.24}")
    }
}

impl DebugRich for f32 {
    fn debug_rich(&self) -> String {
        let bits = self.to_bits();
        format!("{bits:#066b} {bits:#X} {self:.24}")
    }
}

impl DebugRich for f64 {
    fn debug_rich(&self) -> String {
        let bits = self.to_bits();
        format!("{bits:#066b} {bits:#X} {self:.24}")
    }
}

impl<T: DebugRich> DebugRich for (T,) {
    fn debug_rich(&self) -> String {
        self.0.debug_rich()
    }
}

impl<T1, T2> DebugRich for (T1, T2)
where
    T1: DebugRich,
    T2: DebugRich,
{
    fn debug_rich(&self) -> String {
        format!("(\n{},\n{},\n)", self.0.debug_rich(), self.1.debug_rich(),)
    }
}

impl<T1, T2, T3> DebugRich for (T1, T2, T3)
where
    T1: DebugRich,
    T2: DebugRich,
    T3: DebugRich,
{
    fn debug_rich(&self) -> String {
        format!(
            "(\n{},\n{},\n{},\n)",
            self.0.debug_rich(),
            self.1.debug_rich(),
            self.2.debug_rich(),
        )
    }
}

impl<T1, T2, T3, T4> DebugRich for (T1, T2, T3, T4)
where
    T1: DebugRich,
    T2: DebugRich,
    T3: DebugRich,
    T4: DebugRich,
{
    fn debug_rich(&self) -> String {
        format!(
            "(\n{},\n{},\n{},\n{},\n)",
            self.0.debug_rich(),
            self.1.debug_rich(),
            self.2.debug_rich(),
            self.3.debug_rich(),
        )
    }
}

pub trait PtxScalar: Copy + Num + Bounded + Debug + DebugRich + OnDevice + Any {
    fn name() -> &'static str;

    fn unsigned() -> bool {
        Self::min_value() == <Self as Zero>::zero()
    }
    fn float() -> bool {
        false
    }
    fn signed() -> bool {
        !Self::float() && !Self::unsigned()
    }
    fn is_f32() -> bool {
        Self::float() && Self::size_of() == 4
    }
}

impl PtxScalar for u8 {
    fn name() -> &'static str {
        "u8"
    }
}

impl PtxScalar for i8 {
    fn name() -> &'static str {
        "s8"
    }
}

impl PtxScalar for u16 {
    fn name() -> &'static str {
        "u16"
    }
}

impl PtxScalar for i16 {
    fn name() -> &'static str {
        "s16"
    }
}

impl PtxScalar for u32 {
    fn name() -> &'static str {
        "u32"
    }
}

impl PtxScalar for i32 {
    fn name() -> &'static str {
        "s32"
    }
}

impl PtxScalar for u64 {
    fn name() -> &'static str {
        "u64"
    }
}

impl PtxScalar for i64 {
    fn name() -> &'static str {
        "s64"
    }
}

impl PtxScalar for F8E4M3 {
    fn name() -> &'static str {
        "e4m3"
    }
    fn float() -> bool {
        true
    }
}

impl PtxScalar for F8E5M2 {
    fn name() -> &'static str {
        "e5m2"
    }
    fn float() -> bool {
        true
    }
}

impl PtxScalar for half::f16 {
    fn name() -> &'static str {
        "f16"
    }
    fn float() -> bool {
        true
    }
}

impl PtxScalar for f32 {
    fn name() -> &'static str {
        "f32"
    }
    fn float() -> bool {
        true
    }
}

impl PtxScalar for f64 {
    fn name() -> &'static str {
        "f64"
    }
    fn float() -> bool {
        true
    }
}

pub trait Fp8: PtxScalar + num::Float {
    fn from_f32(x: f32) -> Self;
    fn from_bits(bits: u8) -> Self;
    fn to_bits(&self) -> u8;
    fn to_f16(&self) -> f16;
    // This is necessary because float8 is_nan hardcodes a few NaN values and returns "false" for
    // others.
    fn is_nan_correct(&self) -> bool;
}

impl Fp8 for F8E4M3 {
    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }
    fn from_bits(bits: u8) -> Self {
        Self::from_bits(bits)
    }
    fn to_bits(&self) -> u8 {
        self.to_bits()
    }
    fn to_f16(&self) -> f16 {
        self.to_f64() as f16
    }
    fn is_nan_correct(&self) -> bool {
        self.is_nan()
    }
}

impl Fp8 for F8E5M2 {
    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }
    fn from_bits(bits: u8) -> Self {
        Self::from_bits(bits)
    }
    fn to_bits(&self) -> u8 {
        self.to_bits()
    }
    fn to_f16(&self) -> f16 {
        self.to_f64() as f16
    }
    fn is_nan_correct(&self) -> bool {
        matches!(
            self.to_bits() & 0b01111111,
            0b01111101 | 0b01111110 | 0b01111111
        )
    }
}

const SEED: u64 = 0x761194f3027874ef;
const GROUP_SIZE: usize = 128;
// Totally unscientific number that works on my machine
const SAFE_MEMORY_LIMIT: usize = 1 << 29;

fn load_module<'a>(ctx: &'a dyn TestContext, t: &dyn TestPtx) -> Result<CudaModule<'a>, TestError> {
    let cuda = ctx.cuda();

    match ctx.prepare_test_source(t) {
        Ok(src) => {
            let mut module = ptr::null_mut();
            let load_result = unsafe { cuda.cuModuleLoadData(&mut module, src.as_ptr() as _) };

            match load_result {
                Ok(()) => Ok(CudaModule {
                    cuda,
                    value: module,
                }),
                Err(code) => {
                    return Err(TestError::CompilationFail {
                        message: format!("CUDA Error {code}"),
                    })
                }
            }
        }
        Err(message) => return Err(TestError::CompilationFail { message }),
    }
}

pub fn run_random<Test: RandomTest>(
    ctx: &dyn TestContext,
    t: Test,
    fail_fast: bool,
) -> Result<(), TestError> {
    let cuda = ctx.cuda();

    let module = load_module(ctx, &t)?;
    let mut kernel = ptr::null_mut();
    unsafe { cuda.cuModuleGetFunction(&mut kernel, module.value, c"run".as_ptr()) }
        .map_err(|_| TestError::MissingRunFunction)?;

    let mut rng = XorShiftRng::seed_from_u64(SEED);
    let mut free_memory = 0;
    let mut total_memory = 0;
    unsafe { cuda.cuMemGetInfo_v2(&mut free_memory, &mut total_memory) }.unwrap();
    let max_memory = (total_memory / 2).min(SAFE_MEMORY_LIMIT);
    let total_elements = 2.pow(32);
    assert!(total_elements % GROUP_SIZE == 0);
    let element_size = Test::Input::size_of() + Test::Output::size_of();
    let required_memory = total_elements * element_size;
    let iterations = (required_memory / max_memory).max(1);
    let memory_batch_size: usize =
        next_multiple_of(required_memory / iterations, GROUP_SIZE * element_size);
    let mut inputs = vec![Vec::new(); Test::Input::COMPONENTS];
    let mut outputs = vec![Test::Output::zero(); memory_batch_size / element_size];

    let mut first_error = None;
    let mut total_cases = 0;
    let mut passed_cases = 0;

    for iteration in 0..iterations {
        assert_eq!(Test::Output::COMPONENTS, 1);
        let memory_batch_size = if iteration == iterations - 1 {
            required_memory - (memory_batch_size * (iterations - 1))
        } else {
            memory_batch_size
        };
        let element_batch_size = memory_batch_size / element_size;
        for vec in inputs.iter_mut() {
            vec.clear();
        }
        for _ in 0..element_batch_size {
            t.generate(&mut rng).write(&mut inputs);
        }
        outputs.resize(element_batch_size, Test::Output::zero());
        let dev_inputs: Vec<_> = inputs
            .iter()
            .map(|vec| {
                let devptr = cuda_malloc(cuda, vec.len());
                unsafe {
                    cuda.cuMemcpyHtoD_v2(devptr.value, vec.as_ptr().cast_mut().cast(), vec.len())
                }
                .unwrap();
                devptr
            })
            .collect();
        let dev_output = cuda_malloc(cuda, element_batch_size * Test::Output::size_of());
        let mut args = dev_inputs
            .iter()
            .map(|dev_ptr| &dev_ptr.value as *const u64)
            .collect::<Vec<_>>();
        args.push(&dev_output.value);
        unsafe {
            cuda.cuLaunchKernel(
                kernel,
                (element_batch_size / GROUP_SIZE) as u32,
                1,
                1,
                GROUP_SIZE as u32,
                1,
                1,
                0,
                0 as _,
                args.as_mut_ptr() as _,
                ptr::null_mut(),
            )
        }
        .unwrap();
        unsafe { cuda.cuStreamSynchronize(0 as _) }.unwrap();
        unsafe {
            cuda.cuMemcpyDtoH_v2(
                outputs.as_mut_ptr() as _,
                dev_output.value,
                outputs.len() * Test::Output::size_of(),
            )
        }
        .unwrap();
        for (i, output) in outputs.iter().copied().enumerate() {
            let input = Test::Input::read(&inputs, i);
            total_cases += 1;
            if let Err(expected) = t.host_verify(input, output) {
                first_error.get_or_insert((input, output, expected));
                if fail_fast {
                    break;
                }
            } else {
                passed_cases += 1;
            }
        }
    }

    if let Some((input, output, expected)) = first_error {
        Err(TestError::ResultMismatch {
            input: input.debug_rich(),
            output: output.debug_rich(),
            expected: expected.debug_rich(),
            total_cases,
            passed_cases,
        })
    } else {
        Ok(())
    }
}

fn cuda_malloc<'a>(cuda: &'a Cuda, size: usize) -> DevicePtr<'a> {
    let mut value = 0;
    unsafe { cuda.cuMemAlloc_v2(&mut value, size) }.unwrap();
    DevicePtr { cuda, value }
}

fn next_multiple_of(value: usize, multiple: usize) -> usize {
    ((value + multiple - 1) / multiple) * multiple
}

pub fn run_range<Test: RangeTest>(
    ctx: &dyn TestContext,
    t: Test,
    fail_fast: bool,
) -> Result<(), TestError> {
    let cuda = ctx.cuda();

    let module = load_module(ctx, &t)?;
    let mut kernel = ptr::null_mut();
    unsafe { cuda.cuModuleGetFunction(&mut kernel, module.value, c"run".as_ptr()) }
        .map_err(|_| TestError::MissingRunFunction)?;

    let mut free_memory = 0;
    let mut total_memory = 0;
    unsafe { cuda.cuMemGetInfo_v2(&mut free_memory, &mut total_memory) }.unwrap();
    let max_memory = (total_memory / 2).min(SAFE_MEMORY_LIMIT);
    let total_elements = Test::MAX_VALUE as usize + 1;
    if total_elements % GROUP_SIZE != 0 {
        panic!(
            "Total element count {} must be divisble by {}",
            total_elements, GROUP_SIZE
        );
    }
    let element_size = Test::Input::size_of() + Test::Output::size_of();
    let required_memory = total_elements * element_size;
    let iterations = (required_memory / max_memory).max(1);
    let memory_batch_size: usize =
        next_multiple_of(required_memory / iterations, GROUP_SIZE * element_size);
    let mut inputs = vec![Vec::new(); Test::Input::COMPONENTS];
    let mut outputs = vec![Test::Output::zero(); memory_batch_size / element_size];

    let mut first_error = None;
    let mut total_cases = 0;
    let mut passed_cases = 0;

    for iteration in 0..iterations {
        assert_eq!(Test::Output::COMPONENTS, 1);
        let elment_start = iteration * memory_batch_size / element_size;
        let memory_batch_size = if iteration == iterations - 1 {
            required_memory - (memory_batch_size * (iterations - 1))
        } else {
            memory_batch_size
        };
        let element_batch_size = memory_batch_size / element_size;
        for vec in inputs.iter_mut() {
            vec.clear();
        }
        for i in 0..element_batch_size {
            let input = t.generate((elment_start + i) as u32);
            input.write(&mut inputs);
        }
        outputs.resize(element_batch_size, Test::Output::zero());
        let dev_inputs: Vec<_> = inputs
            .iter()
            .map(|vec| {
                let devptr = cuda_malloc(cuda, vec.len());
                unsafe {
                    cuda.cuMemcpyHtoD_v2(devptr.value, vec.as_ptr().cast_mut().cast(), vec.len())
                }
                .unwrap();
                devptr
            })
            .collect();
        let dev_output = cuda_malloc(cuda, element_batch_size * Test::Output::size_of());
        let mut args = dev_inputs
            .iter()
            .map(|ptr| &ptr.value as *const u64)
            .collect::<Vec<_>>();
        args.push(&dev_output.value);
        unsafe {
            cuda.cuLaunchKernel(
                kernel,
                (element_batch_size / GROUP_SIZE) as u32,
                1,
                1,
                GROUP_SIZE as u32,
                1,
                1,
                0,
                0 as _,
                args.as_mut_ptr() as _,
                ptr::null_mut(),
            )
        }
        .unwrap();
        unsafe { cuda.cuStreamSynchronize(0 as _) }.unwrap();
        unsafe {
            cuda.cuMemcpyDtoH_v2(
                outputs.as_mut_ptr() as _,
                dev_output.value,
                outputs.len() * Test::Output::size_of(),
            )
        }
        .unwrap();
        for (i, output) in outputs.iter().copied().enumerate() {
            let input = Test::Input::read(&inputs, i);
            total_cases += 1;
            if let Err(expected) = t.host_verify(input, output) {
                first_error.get_or_insert((input, output, expected));
                if fail_fast {
                    break;
                }
            } else {
                passed_cases += 1;
            }
        }
    }

    if let Some((input, output, expected)) = first_error {
        Err(TestError::ResultMismatch {
            input: input.debug_rich(),
            output: output.debug_rich(),
            expected: expected.debug_rich(),
            total_cases,
            passed_cases,
        })
    } else {
        Ok(())
    }
}

pub type TestFunction = Box<dyn FnOnce(&dyn TestContext, bool) -> Result<(), TestError>>;

pub fn make_random<T: RandomTest + 'static>(t: T) -> TestFunction {
    return Box::new(move |ctx, fail_fast| run_random::<T>(ctx, t, fail_fast));
}

pub fn make_range<T: RangeTest + 'static>(t: T) -> TestFunction {
    return Box::new(move |ctx, fail_fast| run_range::<T>(ctx, t, fail_fast));
}

pub struct TestCase {
    pub test: TestFunction,
    pub name: String,
}

impl TestCase {
    pub fn new(name: String, test: TestFunction) -> Self {
        TestCase { test, name }
    }

    pub fn join_invalid_tests(name: String, tests: Vec<(String, TestFunction)>) -> Self {
        use TestError::*;

        let test = Box::new(move |ctx: &dyn TestContext, fail_fast: bool| {
            for (name, test) in tests {
                match test(ctx, fail_fast) {
                    Err(CompilationFail { .. } | MissingRunFunction) => {}
                    Ok(()) | Err(ResultMismatch { .. }) => return Err(CompilationSuccess { name }),
                    Err(CompilationSuccess { .. }) => {
                        unreachable!("tests may not report CompilationSuccess")
                    }
                }
            }
            Ok(())
        });
        TestCase { test, name }
    }
}

/// Errors that a test can produce.
pub enum TestError {
    /// Used when compilation fails, e.g. during CUDA module loading or NVRTC launch
    CompilationFail { message: String },
    /// Used when tests that should have failed compilation, succeed unexpectedly
    CompilationSuccess { name: String },
    /// Used when the test compiled successfully, but found mismatching values
    ResultMismatch {
        input: String,
        output: String,
        expected: String,
        total_cases: usize,
        passed_cases: usize,
    },
    /// Used when `cuModuleGetFunction` fails
    MissingRunFunction,
}
