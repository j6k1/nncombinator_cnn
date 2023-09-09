//! nncombinator_cnn is a neural network library that allows type-safe implementation.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate libc;
extern crate cuda_runtime_sys;
extern crate rcublas_sys;
extern crate rcublas;
extern crate rcudnn;
extern crate rcudnn_sys;
extern crate rayon;

extern crate nncombinator;

pub mod collection;
//pub mod device;
//pub mod layer;

pub enum Assert<const CHECK: bool> {}

pub trait IsTrue {}

impl IsTrue for Assert<true> {}

pub const fn assert_convolution<const H:usize,const W:usize, const FH: usize,const FW: usize,const PAD:usize,const S:usize>() -> bool {
    H + 2 * PAD >= FH && (H + 2 * PAD - FH) % S == 0 && W + 2 * PAD >= FW && (W + 2 * PAD - FW) % S == 0
}
