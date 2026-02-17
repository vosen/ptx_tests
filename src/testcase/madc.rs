use crate::test::{make_random, RandomTest, TestCase, TestCommon, TestPtx};

static PTX: &str = include_str!("madc.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "madc_u32".to_string(),
            make_random(Madc {
                carry_out: false,
                type_: "u32",
            }),
        ),
        TestCase::new(
            "madc_cc_u32".to_string(),
            make_random(Madc {
                carry_out: true,
                type_: "u32",
            }),
        ),
        TestCase::new(
            "madc_s32".to_string(),
            make_random(Madc {
                carry_out: false,
                type_: "s32",
            }),
        ),
        TestCase::new(
            "madc_cc_s32".to_string(),
            make_random(Madc {
                carry_out: true,
                type_: "s32",
            }),
        ),
    ]
}

struct Madc {
    carry_out: bool,
    type_: &'static str,
}

impl TestPtx for Madc {
    fn body(&self) -> String {
        PTX.replace("<TYPE>", self.type_)
            .replace("<CC>", if self.carry_out { ".cc" } else { "" })
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "input_c", "input_d", "output"]
    }
}

impl TestCommon for Madc {
    type Input = (u32, u32, u32, u32);
    type Output = u64;

    fn host_verify(&self, input: (u32, u32, u32, u32), output: u64) -> Result<(), u64> {
        let (a, b, c, carry_in) = input;
        let (lhs, _) = a.overflowing_mul(b);
        let (rhs, carry_out_2) = c.overflowing_add(carry_in);
        let (expected, carry_out_3) = lhs.overflowing_add(rhs);
        let mut result = expected as u64;
        let cc_cf = if self.carry_out {
            (carry_out_2 || carry_out_3) as u64
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

impl RandomTest for Madc {
    fn generate<R: rand::Rng>(&self, rng: &mut R) -> Self::Input {
        let a = rng.gen::<u32>();
        let b = rng.gen::<u32>();
        let c = rng.gen::<u32>();
        let carry_in = rng.gen::<bool>();
        (a, b, c, carry_in as u32)
    }
}
