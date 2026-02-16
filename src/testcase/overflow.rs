use crate::test::{make_random, RandomTest, TestCase, TestCommon, TestPtx};

static ADDC_PTX: &str = include_str!("addc.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        //TestCase::new("add_cc".to_string(), make_random(AddCC)),
        TestCase::new("addc".to_string(), make_random(Addc { carry_out: false })),
        TestCase::new("addc_cc".to_string(), make_random(Addc { carry_out: true })),
    ]
}

struct Addc {
    carry_out: bool,
}

impl TestPtx for Addc {
    fn body(&self) -> String {
        ADDC_PTX
            .replace("<TYPE>", "u32")
            .replace("<CC>", if self.carry_out { ".cc" } else { "" })
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "input_c", "output"]
    }
}

impl TestCommon for Addc {
    type Input = (u32, u32, u32);
    type Output = u64;

    fn host_verify(&self, input: (u32, u32, u32), output: u64) -> Result<(), u64> {
        let (a, b, carry_in) = input;
        let (expected, carry_out_1) = a.overflowing_add(b);
        let (expected, carry_out_2) = expected.overflowing_add(carry_in);
        let mut result = expected as u64;
        let cc_cf = if self.carry_out {
            (carry_out_1 || carry_out_2) as u64
        } else {
            carry_in as u64
        };
        result |= cc_cf << 32;
        if result == output {
            Ok(())
        } else {
            Err(result)
        }
    }
}

impl RandomTest for Addc {
    fn generate<R: rand::Rng>(&self, rng: &mut R) -> Self::Input {
        let a = rng.gen::<u32>();
        let b = rng.gen::<u32>();
        let carry_in = rng.gen::<bool>();
        (a, b, carry_in as u32)
    }
}
