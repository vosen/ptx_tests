use crate::test::{RandomTest, TestCase, TestCommon, TestPtx};

pub static PTX: &str = include_str!("mul24.ptx");

#[derive(Default, Clone, Copy)]
pub struct Mul24 {
    pub signed: bool,
    pub hi: bool,
}

impl TestPtx for Mul24 {
    fn body(&self) -> String {
        let typ = if self.signed { "s32" } else { "u32" };
        let mode = if self.hi { "hi" } else { "lo" };
        PTX.replace("<MODE>", mode).replace("<TYPE>", typ)
    }

    fn args(&self) -> &[&str] {
        &["input0", "input1", "output"]
    }
}

impl TestCommon for Mul24 {
    type Input = (u32, u32);
    type Output = u32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b) = input;
        if !self.signed {
            let a24 = a & 0x00FF_FFFF;
            let b24 = b & 0x00FF_FFFF;
            let product = (a24 as u64) * (b24 as u64);
            let expected = if self.hi {
                (product >> 16) as u32
            } else {
                product as u32
            };
            if expected == output {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let a24 = (((a << 8) as i32) >> 8) as i64;
            let b24 = (((b << 8) as i32) >> 8) as i64;
            let product = a24 * b24;
            let expected = if self.hi {
                (product >> 16) as u32
            } else {
                product as u32
            };
            if expected == output {
                Ok(())
            } else {
                Err(expected)
            }
        }
    }
}

impl RandomTest for Mul24 {
    fn generate<R: rand::Rng>(&self, rng: &mut R) -> Self::Input {
        let a = rng.gen::<u32>();
        let b = rng.gen::<u32>();
        (a, b)
    }
}

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];

    for &signed in &[false, true] {
        for &hi in &[false, true] {
            let name = format!(
                "mul24_{}_{}",
                if hi { "hi" } else { "lo" },
                if signed { "s32" } else { "u32" },
            );
            tests.push(TestCase::new(
                name,
                crate::test::make_random(Mul24 { signed, hi }),
            ));
        }
    }
    tests
}
