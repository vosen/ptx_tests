use crate::test::{make_random, RandomTest, TestCase, TestCommon, TestPtx};

static ADDC_SUBC_PTX: &str = include_str!("addc_subc.ptx");

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "addc_u32".to_string(),
            make_random(AddcOrSubc {
                type_: "u32",
                carry_out: false,
                is_sub: false,
            }),
        ),
        TestCase::new(
            "addc_cc_u32".to_string(),
            make_random(AddcOrSubc {
                type_: "u32",
                carry_out: true,
                is_sub: false,
            }),
        ),
        TestCase::new(
            "subc_u32".to_string(),
            make_random(AddcOrSubc {
                type_: "u32",
                carry_out: false,
                is_sub: true,
            }),
        ),
        TestCase::new(
            "subc_cc_u32".to_string(),
            make_random(AddcOrSubc {
                type_: "u32",
                carry_out: true,
                is_sub: true,
            }),
        ),
        TestCase::new(
            "addc_s32".to_string(),
            make_random(AddcOrSubc {
                type_: "s32",
                carry_out: false,
                is_sub: false,
            }),
        ),
        TestCase::new(
            "addc_cc_s32".to_string(),
            make_random(AddcOrSubc {
                type_: "s32",
                carry_out: true,
                is_sub: false,
            }),
        ),
        TestCase::new(
            "subc_s32".to_string(),
            make_random(AddcOrSubc {
                type_: "s32",
                carry_out: false,
                is_sub: true,
            }),
        ),
        TestCase::new(
            "subc_cc_s32".to_string(),
            make_random(AddcOrSubc {
                type_: "s32",
                carry_out: true,
                is_sub: true,
            }),
        ),
    ]
}

struct AddcOrSubc {
    is_sub: bool,
    carry_out: bool,
    type_: &'static str,
}

impl TestPtx for AddcOrSubc {
    fn body(&self) -> String {
        ADDC_SUBC_PTX
            .replace("<OP>", if self.is_sub { "subc" } else { "addc" })
            .replace("<TYPE>", self.type_)
            .replace("<CC>", if self.carry_out { ".cc" } else { "" })
    }

    fn args(&self) -> &[&str] {
        &["input_a", "input_b", "input_c", "output"]
    }
}

impl TestCommon for AddcOrSubc {
    type Input = (u32, u32, u32);
    type Output = u64;

    fn host_verify(&self, input: (u32, u32, u32), output: u64) -> Result<(), u64> {
        let (a, b, carry_in) = input;
        let (op, carry_in_for_calculation) = if self.is_sub {
            (
                u32::overflowing_sub as fn(u32, u32) -> (u32, bool),
                !(carry_in != 0) as u32,
            )
        } else {
            (
                u32::overflowing_add as fn(u32, u32) -> (u32, bool),
                carry_in,
            )
        };
        let (rhs, carry_out_2) = b.overflowing_add(carry_in_for_calculation);
        let (expected, carry_out_1) = op(a, rhs);
        let mut result = expected as u64;
        let cc_cf = if self.carry_out {
            if self.is_sub {
                !(carry_out_1 || carry_out_2) as u64
            } else {
                (carry_out_1 || carry_out_2) as u64
            }
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

impl RandomTest for AddcOrSubc {
    fn generate<R: rand::Rng>(&self, rng: &mut R) -> Self::Input {
        let a = rng.gen::<u32>();
        let b = rng.gen::<u32>();
        let carry_in = rng.gen::<bool>();
        (a, b, carry_in as u32)
    }
}
