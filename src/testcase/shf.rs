use crate::test::{make_random, RandomTest, TestCase, TestCommon, TestPtx};

static PTX: &str = include_str!("shf.ptx");

#[derive(Copy, Clone)]
enum ShfDir {
    Left,
    Right,
}

#[derive(Copy, Clone)]
enum ShfMode {
    Clamp,
    Wrap,
}

#[derive(Copy, Clone)]
struct Shf {
    dir: ShfDir,
    mode: ShfMode,
}

impl TestPtx for Shf {
    fn body(&self) -> String {
        let dir_str = match self.dir {
            ShfDir::Left => "l",
            ShfDir::Right => "r",
        };
        let mode_str = match self.mode {
            ShfMode::Clamp => "clamp",
            ShfMode::Wrap => "wrap",
        };

        PTX.replace("<DIR>", dir_str).replace("<MODE>", mode_str)
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "input_c", "output"]
    }
}

impl TestCommon for Shf {
    type Input = (u32, u32, u32);
    type Output = u32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b, c) = input;
        let shift = match self.mode {
            ShfMode::Clamp => c.min(32),
            ShfMode::Wrap => c & 0x1f,
        };

        let expected = match self.dir {
            ShfDir::Left => shl(b, shift) | shr(a, 32 - shift),
            ShfDir::Right => shl(b, 32 - shift) | shr(a, shift),
        };
        if output == expected {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

//Rust treats >> 32 as mod 32, so x >> 32 = x instead of 0, while
// PTX shifts it out to 0
fn shr(x: u32, s: u32) -> u32 {
    if s >= 32 {
        0
    } else {
        x >> s
    }
}

fn shl(x: u32, s: u32) -> u32 {
    if s >= 32 {
        0
    } else {
        x << s
    }
}

impl RandomTest for Shf {
    fn generate<R: rand::Rng>(&self, rng: &mut R) -> Self::Input {
        let a = rng.gen();
        let b = rng.gen();
        let c = rng.gen();
        (a, b, c)
    }
}

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "shf_l_clamp_b32".to_string(),
            make_random(Shf {
                dir: ShfDir::Left,
                mode: ShfMode::Clamp,
            }),
        ),
        TestCase::new(
            "shf_l_wrap_b32".to_string(),
            make_random(Shf {
                dir: ShfDir::Left,
                mode: ShfMode::Wrap,
            }),
        ),
        TestCase::new(
            "shf_r_clamp_b32".to_string(),
            make_random(Shf {
                dir: ShfDir::Right,
                mode: ShfMode::Clamp,
            }),
        ),
        TestCase::new(
            "shf_r_wrap_b32".to_string(),
            make_random(Shf {
                dir: ShfDir::Right,
                mode: ShfMode::Wrap,
            }),
        ),
    ]
}
