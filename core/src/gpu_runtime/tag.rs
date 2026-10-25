use self::diff_add_mult_prob::DiffAddMultProbProvenance;
use super::array::C_Array;
use crate::{
  common::foreign_tensor::{DynamicExternalTensor, FromTensor},
  runtime::provenance::{
    diff_min_max_prob::DiffMinMaxProbProvenance, diff_top_k_proofs::DiffTopKProofsProvenance,
    min_max_prob::MinMaxProbProvenance, top_k_proofs::TopKProofsProvenance, *,
  },
  utils::PointerFamily,
};
use std::ffi::c_void;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct C_Tag<Prov: Provenance> {
  pub tag: Prov::FFITag,
}

impl<Prov: Provenance> C_Tag<Prov> {
  pub fn new(tag: &Prov::Tag) -> Self {
    Self {
      tag: Prov::to_ffi_tag(tag),
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub enum C_Provenance {
  Unit(),
  MinMaxProb(C_MinMaxProb),
  DiffMinMaxProb(C_DiffMinMaxProb),
  DiffAddMultProb(C_DiffAddMultProb),
  DiffTopKProofs(C_DiffTopKProofs),
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_MinMaxProb {
  valid_threshold: f64,
}
impl C_MinMaxProb {
  pub fn new() -> Self {
    Self { valid_threshold: 0.5 }
  }
}

impl FromTensor for *const c_void {
  fn from_tensor(_tensor: DynamicExternalTensor) -> Option<Self> {
    None
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_Prob {
  pub prob: f64,
  pub deriv: isize,
}
impl std::fmt::Display for C_Prob {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "Prob: (prob: {}, deriv: {})", self.prob, self.deriv)
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_DiffMinMaxProb {
  valid_threshold: f64,
}
impl C_DiffMinMaxProb {
  pub unsafe fn new<T: FromTensor, Ptr: PointerFamily>(prov: &DiffMinMaxProbProvenance<T, Ptr>) -> Self {
    Self {
      valid_threshold: prov.valid_threshold,
    }
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_Gradient {
  pub indices: C_Array<usize>,
  pub values: C_Array<f64>,
}
impl std::fmt::Display for C_Gradient {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "Gradient: [")?;
    for (i, (index, value)) in self
      .indices
      .as_slice()
      .iter()
      .zip(self.values.as_slice().iter())
      .enumerate()
    {
      if i > 0 {
        write!(f, ", ")?;
      }
      write!(f, "({}, {})", index, value)?;
    }
    write!(f, "]")
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_DualNumber {
  pub real: f64,
  pub gradient: C_Gradient,
}
impl std::fmt::Display for C_DualNumber {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "DualNumber: (real: {}, gradient: {})", self.real, self.gradient)
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_DiffAddMultProb {
  valid_threshold: f64,
}
impl C_DiffAddMultProb {
  pub unsafe fn new<T: FromTensor, Ptr: PointerFamily>(prov: &DiffAddMultProbProvenance<T, Ptr>) -> Self {
    Self {
      valid_threshold: prov.valid_threshold,
    }
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_Proof {
  pub literals: C_Array<i32>,
  pub empty: bool,
}
impl C_Proof {
  // Zero is reserved. Device proofs store signed 16-bit literals.
  pub fn encode_literal(literal: &Literal) -> i32 {
    let (id, negative) = match literal {
      Literal::Pos(id) => (*id, false),
      Literal::Neg(id) => (*id, true),
    };
    assert!(id < i16::MAX as usize, "GPU proof fact id exceeds 32766");
    let encoded = id as i32 + 1;
    if negative { -encoded } else { encoded }
  }

  pub fn decode_literal(encoded: &i32) -> Literal {
    assert!(*encoded != 0 && *encoded >= -(i16::MAX as i32) && *encoded <= i16::MAX as i32,
            "Invalid GPU proof literal");
    let id = (encoded.abs() - 1) as usize;
    if *encoded < 0 { Literal::Neg(id) } else { Literal::Pos(id) }
  }
}

#[cfg(test)]
mod proof_literal_tests {
  use super::*;

  #[test]
  fn preserves_fact_zero_sign_and_device_boundary() {
    assert_eq!(C_Proof::encode_literal(&Literal::Pos(0)), 1);
    assert_eq!(C_Proof::encode_literal(&Literal::Neg(0)), -1);
    for id in [0, 1, 32766] {
      for literal in [Literal::Pos(id), Literal::Neg(id)] {
        assert_eq!(C_Proof::decode_literal(&C_Proof::encode_literal(&literal)), literal);
      }
    }
  }

  #[test]
  #[should_panic(expected = "exceeds 32766")]
  fn rejects_device_id_overflow() {
    C_Proof::encode_literal(&Literal::Pos(32767));
  }

  #[test]
  #[should_panic(expected = "Invalid GPU proof literal")]
  fn rejects_zero_literal() {
    C_Proof::decode_literal(&0);
  }

  #[test]
  #[should_panic(expected = "Invalid GPU proof literal")]
  fn rejects_invalid_signed_literal() {
    C_Proof::decode_literal(&i32::MIN);
  }
}

impl std::fmt::Display for C_Proof {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "Proof: [")?;
    for (i, lit) in self.literals.as_slice().iter().enumerate() {
      if i > 0 {
        write!(f, ", ")?;
      }
      write!(f, "{}", lit)?;
    }
    write!(f, "]")
  }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct C_DiffTopKProofs {
  // nested because of batching
  pub literal_probabilities: C_Array<C_Array<f32>>,
}
impl<T: FromTensor, Ptr: PointerFamily> From<&Vec<DiffTopKProofsProvenance<T, Ptr>>> for C_DiffTopKProofs {
  fn from(ctxes: &Vec<DiffTopKProofsProvenance<T, Ptr>>) -> Self {
    //Ptr::get_rc_cell(&prov.storage.storage, |d: &Vec<(f64, Option<T>)>| println!("Number of input variables: {}", d.len()));
    let probs = ctxes
      .iter()
      .map(|prov| {
        C_Array::new(Ptr::get_rc_cell(&prov.storage.storage, |d: &Vec<(f64, Option<T>)>| {
          d.iter().map(|(p, _)| p.clone() as f32).collect::<Vec<_>>()
        }))
      })
      .collect::<Vec<_>>();
    let literal_probabilities = C_Array::new(probs);

    Self { literal_probabilities }
  }
}

impl<Ptr: PointerFamily> From<&Vec<TopKProofsProvenance<Ptr>>> for C_DiffTopKProofs {
  fn from(ctxes: &Vec<TopKProofsProvenance<Ptr>>) -> Self {
    //Ptr::get_cell(&prov.probs, |d: &Vec<f64>| println!("Number of input variables: {}", d.len()));
    let probs = ctxes
      .iter()
      .map(|prov| {
        C_Array::new(Ptr::get_cell(&prov.probs, |d: &Vec<f64>| {
          d.iter().map(|p| p.clone() as f32).collect::<Vec<_>>()
        }))
      })
      .collect::<Vec<_>>();
    let literal_probabilities = C_Array::new(probs);

    Self { literal_probabilities }
  }
}
