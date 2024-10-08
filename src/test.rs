use crate::cuda::Cuda;
use half::f16;
use num::{cast::AsPrimitive, Bounded, Num, PrimInt, Zero};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::{fmt::Debug, mem, ptr, u32};

pub trait TestCommon {
    type Input: OnDevice;
    type Output: OnDevice;
    fn host_verify(input: Self::Input, output: Self::Output) -> Result<(), Self::Output>;
    fn ptx() -> String;
}

pub trait RangeTest: TestCommon {
    const MAX_VALUE: u32 = u32::MAX;
    fn generate(input: u32) -> Self::Input;
}

pub trait RandomTest: TestCommon {
    fn generate<R: Rng>(rng: &mut R) -> Self::Input;
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
impl OnDevice for f16 {
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

pub trait PtxScalar: Copy + Num + Bounded + Debug + OnDevice {
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

impl PtxScalar for f16 {
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

const SEED: u64 = 0x761194f3027874ef;
const GROUP_SIZE: usize = 128;
// Totally unscientific number that works on my machine
const SAFE_MEMORY_LIMIT: usize = 1 << 29;

pub fn run_random<T: RandomTest>(cuda: &Cuda) -> Result<(), TestError> {
    let src = T::ptx();
    let mut module = ptr::null_mut();
    unsafe { cuda.cuModuleLoadData(&mut module, src.as_ptr() as _) }.unwrap();
    let mut kernel = ptr::null_mut();
    unsafe { cuda.cuModuleGetFunction(&mut kernel, module, c"run".as_ptr()) }.unwrap();
    let mut rng = XorShiftRng::seed_from_u64(SEED);
    let mut free_memory = 0;
    let mut total_memory = 0;
    unsafe { cuda.cuMemGetInfo_v2(&mut free_memory, &mut total_memory) }.unwrap();
    let max_memory = (total_memory / 2).min(SAFE_MEMORY_LIMIT);
    let total_elements = 2.pow(32);
    assert!(total_elements % GROUP_SIZE == 0);
    let element_size = T::Input::size_of() + T::Output::size_of();
    let required_memory = total_elements * element_size;
    let iterations = (required_memory / max_memory).max(1);
    let memory_batch_size: usize =
        next_multiple_of(required_memory / iterations, GROUP_SIZE * element_size);
    let mut inputs = vec![Vec::new(); T::Input::COMPONENTS];
    let mut result = vec![T::Output::zero(); memory_batch_size / element_size];
    for iteration in 0..iterations {
        assert_eq!(T::Output::COMPONENTS, 1);
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
            T::generate(&mut rng).write(&mut inputs);
        }
        result.resize(element_batch_size, T::Output::zero());
        let dev_inputs: Vec<u64> = inputs
            .iter()
            .map(|vec| {
                let mut devptr = 0;
                unsafe { cuda.cuMemAlloc_v2(&mut devptr, vec.len()) }.unwrap();
                unsafe { cuda.cuMemcpyHtoD_v2(devptr, vec.as_ptr().cast_mut().cast(), vec.len()) }
                    .unwrap();
                devptr
            })
            .collect();
        let mut dev_output = 0;
        unsafe { cuda.cuMemAlloc_v2(&mut dev_output, element_batch_size * T::Output::size_of()) }
            .unwrap();
        let mut args = dev_inputs
            .iter()
            .map(|ptr| ptr as *const u64)
            .collect::<Vec<_>>();
        args.push(&dev_output);
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
                result.as_mut_ptr() as _,
                dev_output,
                result.len() * T::Output::size_of(),
            )
        }
        .unwrap();
        for (i, result) in result.iter().copied().enumerate() {
            let value = T::Input::read(&inputs, i);
            let result = result;
            if let Err(expected) = T::host_verify(value, result) {
                return Err(TestError {
                    input: format!("{:?}", value),
                    output: format!("{:?}", result),
                    expected: format!("{:?}", expected),
                });
            }
        }
        for devptr in dev_inputs {
            unsafe { cuda.cuMemFree_v2(devptr) }.unwrap();
        }
        unsafe { cuda.cuMemFree_v2(dev_output) }.unwrap();
    }
    unsafe { cuda.cuModuleUnload(module) }.unwrap();
    Ok(())
}

fn next_multiple_of(value: usize, multiple: usize) -> usize {
    ((value + multiple - 1) / multiple) * multiple
}

pub fn run_range<T: RangeTest>(cuda: &Cuda) -> Result<(), TestError> {
    let src = T::ptx();
    let mut module = ptr::null_mut();
    unsafe { cuda.cuModuleLoadData(&mut module, src.as_ptr() as _) }.unwrap();
    let mut kernel = ptr::null_mut();
    unsafe { cuda.cuModuleGetFunction(&mut kernel, module, c"run".as_ptr()) }.unwrap();
    let mut free_memory = 0;
    let mut total_memory = 0;
    unsafe { cuda.cuMemGetInfo_v2(&mut free_memory, &mut total_memory) }.unwrap();
    let max_memory = (total_memory / 2).min(SAFE_MEMORY_LIMIT);
    let total_elements = T::MAX_VALUE as usize + 1;
    assert!(total_elements % GROUP_SIZE == 0);
    let element_size = T::Input::size_of() + T::Output::size_of();
    let required_memory = total_elements * element_size;
    let iterations = (required_memory / max_memory).max(1);
    let memory_batch_size: usize =
        next_multiple_of(required_memory / iterations, GROUP_SIZE * element_size);
    let mut inputs = vec![Vec::new(); T::Input::COMPONENTS];
    let mut result = vec![T::Output::zero(); memory_batch_size / element_size];
    for iteration in 0..iterations {
        assert_eq!(T::Output::COMPONENTS, 1);
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
            let input = T::generate((elment_start + i) as u32);
            input.write(&mut inputs);
        }
        result.resize(element_batch_size, T::Output::zero());
        let dev_inputs: Vec<u64> = inputs
            .iter()
            .map(|vec| {
                let mut devptr = 0;
                unsafe { cuda.cuMemAlloc_v2(&mut devptr, vec.len()) }.unwrap();
                unsafe { cuda.cuMemcpyHtoD_v2(devptr, vec.as_ptr().cast_mut().cast(), vec.len()) }
                    .unwrap();
                devptr
            })
            .collect();
        let mut dev_output = 0;
        unsafe { cuda.cuMemAlloc_v2(&mut dev_output, element_batch_size * T::Output::size_of()) }
            .unwrap();
        let mut args = dev_inputs
            .iter()
            .map(|ptr| ptr as *const u64)
            .collect::<Vec<_>>();
        args.push(&dev_output);
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
                result.as_mut_ptr() as _,
                dev_output,
                result.len() * T::Output::size_of(),
            )
        }
        .unwrap();
        for (i, result) in result.iter().copied().enumerate() {
            let value = T::Input::read(&inputs, i);
            let result = result;
            if let Err(expected) = T::host_verify(value, result) {
                return Err(TestError {
                    input: format!("{:?}", value),
                    output: format!("{:?}", result),
                    expected: format!("{:?}", expected),
                });
            }
        }
        for devptr in dev_inputs {
            unsafe { cuda.cuMemFree_v2(devptr) }.unwrap();
        }
        unsafe { cuda.cuMemFree_v2(dev_output) }.unwrap();
    }
    unsafe { cuda.cuModuleUnload(module) }.unwrap();
    Ok(())
}

pub struct TestCase {
    pub test: fn(cuda: &Cuda) -> Result<(), TestError>,
    pub name: String,
}

pub struct TestError {
    pub input: String,
    pub output: String,
    pub expected: String,
}
